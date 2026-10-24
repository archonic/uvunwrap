//! Core unwrapping: charts, distortion, determinism and input validation.

mod common;
use common::*;

#[test]
fn flat_plane_is_one_undistorted_chart() {
    let m = plane(12);
    let atlas = run(&[&m], &Options::default());
    assert_eq!(atlas.chart_count, 1);
    // Every grid triangle has the same 3D area, so UV areas must match too.
    let out = &atlas.meshes[0];
    let areas: Vec<f64> = out
        .indices
        .chunks_exact(3)
        .map(|t| tri_area([0, 1, 2].map(|c| out.vertices[t[c] as usize].uv.map(f64::from))))
        .collect();
    let (lo, hi) = areas
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(hi / lo < 1.001, "distortion {}", hi / lo);
    assert!(atlas.utilization > 0.8, "utilization {}", atlas.utilization);
}

#[test]
fn cube_gets_six_charts() {
    let m = cube();
    let atlas = run(&[&m], &Options::default());
    assert_eq!(atlas.chart_count, 6);
    assert_eq!(atlas.meshes[0].vertices.len(), 24);
}

#[test]
fn closed_and_curved_surfaces() {
    for m in [torus(48, 24), sphere(32, 16), cylinder(32)] {
        let atlas = run(&[&m], &Options::default());
        assert!(atlas.chart_count >= 2);
        assert!(atlas.utilization > 0.3, "utilization {}", atlas.utilization);
    }
}

#[test]
fn multiple_meshes_share_an_atlas() {
    let (a, b) = (cube(), torus(24, 12));
    let atlas = run(&[&a, &b], &Options::default());
    let charts_a: std::collections::HashSet<_> =
        atlas.meshes[0].vertices.iter().map(|v| v.chart).collect();
    let charts_b: std::collections::HashSet<_> =
        atlas.meshes[1].vertices.iter().map(|v| v.chart).collect();
    assert!(charts_a.is_disjoint(&charts_b));
}

#[test]
fn degenerate_and_soup_input() {
    // Triangle soup (no shared vertices) plus degenerate and duplicate triangles.
    let src = torus(16, 8);
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for t in src.indices.chunks_exact(3) {
        for &i in t {
            indices.push(positions.len() as u32);
            positions.push(src.positions[i as usize]);
        }
    }
    positions.push([5.0, 5.0, 5.0]);
    let lonely = (positions.len() - 1) as u32;
    indices.extend([lonely, lonely, lonely, 0, 0, 1]);
    let m = Mesh { positions, indices };
    let atlas = run(&[&m], &Options::default());
    let out = &atlas.meshes[0];
    let last = out.vertices[*out.indices.last().unwrap() as usize];
    assert!(
        last.chart.is_some(),
        "degenerate triangle should reuse an existing UV"
    );
    assert!(out
        .vertices
        .iter()
        .any(|v| v.xref == lonely && v.chart.is_none()));
}

#[test]
fn inconsistently_wound_input_still_gets_ccw_uvs() {
    // Every 5th triangle reversed; `check` asserts each output triangle is
    // counter-clockwise in UV space in its own index order.
    let mut m = torus(24, 12);
    for (f, t) in m.indices.chunks_exact_mut(3).enumerate() {
        if f % 5 == 0 {
            t.swap(1, 2);
        }
    }
    run(&[&m], &Options::default());
}

#[test]
fn empty_input() {
    let atlas = generate(&[], &Options::default()).unwrap();
    assert_eq!((atlas.chart_count, atlas.page_count), (0, 0));
}

#[test]
fn input_errors() {
    let p = [[0.0f32; 3]; 3];
    let bad_count = MeshInput {
        positions: &p,
        indices: &[0, 1],
        ..Default::default()
    };
    let bad_index = MeshInput {
        positions: &p,
        indices: &[0, 1, 3],
        ..Default::default()
    };
    let nan = [[f32::NAN, 0.0, 0.0], [0.0; 3], [0.0; 3]];
    let bad_pos = MeshInput {
        positions: &nan,
        indices: &[0, 1, 2],
        ..Default::default()
    };
    let o = Options::default();
    assert!(matches!(
        generate(&[bad_count], &o),
        Err(Error::InvalidIndexCount { .. })
    ));
    assert!(matches!(
        generate(&[bad_index], &o),
        Err(Error::IndexOutOfRange { index: 3, .. })
    ));
    assert!(matches!(
        generate(&[bad_pos], &o),
        Err(Error::NonFinitePosition { vertex: 0, .. })
    ));
    let bad_opt = Options {
        max_chart_angle: 95.0,
        ..o
    };
    assert!(matches!(
        generate(&[], &bad_opt),
        Err(Error::InvalidOption { .. })
    ));
}

#[test]
fn output_is_deterministic() {
    // Bit-identical output across runs (fresh hash seeds each time) and thread
    // counts, on meshes that exercise segmentation, splitting and packing.
    let (a, b, c) = (torus(48, 24), sphere(32, 16), cube());
    let inputs = [
        a.input(),
        b.input(),
        MeshInput {
            instance_scales: &[1.0, 1.5],
            ..c.input()
        },
    ];
    let fixed = Options {
        density: Density::Fixed {
            texels_per_unit: 40.0,
            page_size: None,
        },
        ..Options::default()
    };
    let fast = Options {
        packing: Packing::Fast,
        density: Density::Fixed {
            texels_per_unit: 40.0,
            page_size: Some(512),
        },
        ..Options::default()
    };
    for opts in [Options::default(), fixed, fast] {
        let reference = generate(&inputs, &opts).unwrap();
        for threads in [1, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..2 {
                let atlas = pool.install(|| generate(&inputs, &opts).unwrap());
                assert!(atlas == reference, "output differs with {threads} threads");
            }
        }
    }
}

#[test]
fn area_distortion_stays_within_the_limit() {
    // Per chart, each triangle's share of the UV area over its share of the
    // surface area must lie in [1 / d, d].
    for d in [1.5f32, 4.0] {
        for m in [torus(48, 24), sphere(32, 16)] {
            let opts = Options {
                max_area_distortion: d,
                ..Options::default()
            };
            let atlas = run(&[&m], &opts);
            for (f, ratio, _) in area_ratios(&m, &atlas.meshes[0]) {
                let limit = d as f64 * 1.01; // f32 UV rounding
                assert!(
                    ratio <= limit && ratio * limit >= 1.0,
                    "d = {d}: face {f} ratio {ratio}"
                );
            }
        }
    }
}

#[test]
fn curved_charts_keep_texel_density_even() {
    // Area-weighted mean of max(r, 1/r), r as in `area_ratios`. LSCM alone
    // leaves 1.20 on both (its scale drifts across curved charts); ARAP brings
    // them to ~1.05 and ~1.08.
    for m in [torus(48, 24), sphere(32, 16)] {
        let atlas = run(&[&m], &Options::default());
        let (mut sum, mut weight) = (0.0, 0.0);
        for (_, r, a) in area_ratios(&m, &atlas.meshes[0]) {
            sum += a * r.max(1.0 / r);
            weight += a;
        }
        let mean = sum / weight;
        assert!(mean < 1.1, "mean area distortion {mean}");
    }
}

#[test]
fn chart_area_and_boundary_limits() {
    // Per chart: surface area, and the length of edges whose other side is
    // another chart, no chart or nothing (by position, as the input welds).
    // Degenerate faces (the sphere's poles) have no chart of their own.
    let measure = |m: &Mesh, out: &OutputMesh| {
        let mut charts: std::collections::HashMap<u32, (f64, f64)> = Default::default();
        let mut edges: std::collections::HashMap<[[u32; 3]; 2], Vec<usize>> = Default::default();
        let key = |v: u32| m.positions[v as usize].map(f32::to_bits);
        for f in 0..m.faces() {
            let Some(c) = face_chart(out, f).filter(|_| face_area(m, f) > 1e-9) else {
                continue;
            };
            charts.entry(c).or_default().0 += face_area(m, f);
            for k in 0..3 {
                let (a, b) = (
                    key(m.indices[f * 3 + k]),
                    key(m.indices[f * 3 + (k + 1) % 3]),
                );
                edges.entry([a.min(b), a.max(b)]).or_default().push(f);
            }
        }
        for (&[a, b], faces) in &edges {
            let len = (0..3)
                .map(|k| (f32::from_bits(a[k]) as f64 - f32::from_bits(b[k]) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            for &f in faces {
                let c = face_chart(out, f);
                if faces.iter().filter(|&&g| face_chart(out, g) == c).count() < 2 {
                    charts.get_mut(&c.unwrap()).unwrap().1 += len;
                }
            }
        }
        charts
    };
    let m = sphere(32, 16); // area ≈ 4π, no two faces coplanar
    let free = run(&[&m], &Options::default());
    let largest = |a: &Atlas, i: usize| {
        measure(&m, &a.meshes[0])
            .values()
            .map(|c| [c.0, c.1][i])
            .fold(0.0, f64::max)
    };
    for (i, limit) in [(0, 1.0f32), (1, 3.0)] {
        assert!(largest(&free, i) > limit as f64 * 1.5);
        let opts = Options {
            max_chart_area: if i == 0 { limit } else { 0.0 },
            max_boundary_length: if i == 1 { limit } else { 0.0 },
            ..Options::default()
        };
        let limited = run(&[&m], &opts);
        assert!(limited.chart_count > free.chart_count);
        let worst = largest(&limited, i);
        assert!(worst <= limit as f64 * 1.0001, "limit {limit}: {worst}");
    }
}
