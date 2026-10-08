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
        Err(Error::InvalidOption(_))
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
            instances: &[1.0, 1.5],
            ..c.input()
        },
    ];
    let fixed = Options {
        texels_per_unit: Some(40.0),
        resolution: 0,
        ..Options::default()
    };
    let rect = Options {
        packing: Packing::Rect,
        texels_per_unit: Some(40.0),
        resolution: 512,
        ..Options::default()
    };
    for opts in [Options::default(), fixed, rect] {
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
            let out = &atlas.meshes[0];
            let area3 = |f: usize| {
                let p =
                    [0, 1, 2].map(|c| m.positions[m.indices[f * 3 + c] as usize].map(f64::from));
                let (u, v) = (
                    [0, 1, 2].map(|k| p[1][k] - p[0][k]),
                    [0, 1, 2].map(|k| p[2][k] - p[0][k]),
                );
                let c = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
            };
            let mut totals: std::collections::HashMap<u32, (f64, f64)> =
                std::collections::HashMap::new();
            for f in 0..m.faces() {
                if let Some(c) = face_chart(out, f) {
                    let t = totals.entry(c).or_default();
                    t.0 += uv_area(out, f);
                    t.1 += area3(f);
                }
            }
            for f in 0..m.faces() {
                let Some(c) = face_chart(out, f) else {
                    continue;
                };
                let (uv_total, total) = totals[&c];
                let (a, b) = (uv_area(out, f), area3(f));
                if b < total * 1e-6 || a < uv_total * 1e-6 {
                    continue; // degenerate, or too small for f32 UVs to measure
                }
                let ratio = (a / uv_total) / (b / total);
                let limit = d as f64 * 1.01; // f32 UV rounding
                assert!(
                    ratio <= limit && ratio * limit >= 1.0,
                    "d = {d}: face {f} ratio {ratio}"
                );
            }
        }
    }
}
