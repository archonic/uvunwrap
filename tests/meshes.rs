//! End-to-end tests on generated meshes. Every atlas is checked for:
//! preserved topology (same triangles, valid xrefs), UVs inside [0, 1],
//! no flipped triangles and no overlapping triangles anywhere in the atlas.

use std::f32::consts::TAU;
use uvunwrap::{generate, Atlas, Error, MeshInput, Options};

struct Mesh {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl Mesh {
    fn input(&self) -> MeshInput<'_> {
        MeshInput { positions: &self.positions, indices: &self.indices }
    }
}

/// Parametric surface sampled on a grid, optionally wrapping in u and/or v.
fn surface(nu: u32, nv: u32, wrap_u: bool, wrap_v: bool, f: impl Fn(f32, f32) -> [f32; 3]) -> Mesh {
    let (cu, cv) = (if wrap_u { nu } else { nu + 1 }, if wrap_v { nv } else { nv + 1 });
    let mut positions = Vec::new();
    for j in 0..cv {
        for i in 0..cu {
            positions.push(f(i as f32 / nu as f32, j as f32 / nv as f32));
        }
    }
    let id = |i: u32, j: u32| (j % cv) * cu + (i % cu);
    let mut indices = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            indices.extend([a, b, c, a, c, d]);
        }
    }
    Mesh { positions, indices }
}

fn plane(n: u32) -> Mesh {
    surface(n, n, false, false, |u, v| [u * 2.0, v, 0.0])
}

fn torus(nu: u32, nv: u32) -> Mesh {
    surface(nu, nv, true, true, |u, v| {
        let (a, b) = (u * TAU, v * TAU);
        let r = 1.0 + 0.4 * b.cos();
        [r * a.cos(), r * a.sin(), 0.4 * b.sin()]
    })
}

fn sphere(nu: u32, nv: u32) -> Mesh {
    // Poles are duplicated per column, producing degenerate triangles there.
    surface(nu, nv, true, false, |u, v| {
        let (a, b) = (u * TAU, (v - 0.5) * std::f32::consts::PI);
        [b.cos() * a.cos(), b.cos() * a.sin(), b.sin()]
    })
}

fn cylinder(n: u32) -> Mesh {
    surface(n, 4, true, false, |u, v| [(u * TAU).cos(), (u * TAU).sin(), v * 3.0])
}

/// Cube with split vertices per face (24 vertices), like a typical exported mesh.
fn cube() -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let dirs: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]),
        ([-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]),
        ([0., 1., 0.], [0., 0., 1.], [1., 0., 0.]),
        ([0., -1., 0.], [1., 0., 0.], [0., 0., 1.]),
        ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
        ([0., 0., -1.], [0., 1., 0.], [1., 0., 0.]),
    ];
    for (n, t, b) in dirs {
        let base = positions.len() as u32;
        for (s, r) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
            positions.push([0, 1, 2].map(|k| n[k] + s * t[k] + r * b[k]));
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Mesh { positions, indices }
}

fn tri_area(p: [[f64; 2]; 3]) -> f64 {
    0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
}

fn overlap(a: &[[f64; 2]; 3], b: &[[f64; 2]; 3]) -> bool {
    for t in [a, b] {
        for i in 0..3 {
            let (p, q) = (t[i], t[(i + 1) % 3]);
            let ax = [p[1] - q[1], q[0] - p[0]];
            let proj = |t: &[[f64; 2]; 3]| {
                let d: Vec<f64> = t.iter().map(|v| v[0] * ax[0] + v[1] * ax[1]).collect();
                (d.iter().cloned().fold(f64::MAX, f64::min), d.iter().cloned().fold(f64::MIN, f64::max))
            };
            let ((alo, ahi), (blo, bhi)) = (proj(a), proj(b));
            let eps = 1e-6 * (ax[0].abs() + ax[1].abs());
            if ahi <= blo + eps || bhi <= alo + eps {
                return false;
            }
        }
    }
    true
}

/// Altitude below 1e-5 of the mesh scale (all test meshes are ~2-3 units).
fn is_sliver(p: [[f32; 3]; 3]) -> bool {
    let d = |a: [f32; 3], b: [f32; 3]| ((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>() as f64).sqrt();
    let (a, b, c) = (d(p[0], p[1]), d(p[1], p[2]), d(p[2], p[0]));
    let s = 0.5 * (a + b + c);
    let area = (s * (s - a) * (s - b) * (s - c)).max(0.0).sqrt();
    2.0 * area / a.max(b).max(c) < 3e-5
}

fn check(atlas: &Atlas, meshes: &[&Mesh]) {
    assert_eq!(atlas.meshes.len(), meshes.len());
    // (page, texel-space triangle, chart) for every non-degenerate triangle.
    let mut all = Vec::new();
    for (out, input) in atlas.meshes.iter().zip(meshes) {
        assert_eq!(out.indices.len(), input.indices.len());
        for (k, &i) in out.indices.iter().enumerate() {
            assert_eq!(out.vertices[i as usize].xref, input.indices[k], "topology changed");
        }
        for v in &out.vertices {
            assert!(v.uv.iter().all(|c| (0.0..=1.0).contains(c)), "uv out of range: {:?}", v.uv);
            assert!(v.page < atlas.page_count.max(1));
        }
        for t in out.indices.chunks_exact(3) {
            let vs = [0, 1, 2].map(|c| out.vertices[t[c] as usize]);
            let p = vs.map(|v| input.positions[v.xref as usize]);
            let e1 = [0, 1, 2].map(|k| p[1][k] - p[0][k]);
            let e2 = [0, 1, 2].map(|k| p[2][k] - p[0][k]);
            let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            if (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt() < 1e-9 {
                continue; // degenerate in 3D
            }
            assert!(vs[0].chart.is_some() && vs.iter().all(|v| v.chart == vs[0].chart && v.page == vs[0].page));
            let uv = vs.map(|v| [v.uv[0] as f64 * atlas.width as f64, v.uv[1] as f64 * atlas.height as f64]);
            if tri_area(uv) == 0.0 && is_sliver(p) {
                continue; // collapsed on purpose: too thin for f32 UVs
            }
            assert!(tri_area(uv) > 0.0, "flipped or collapsed triangle {p:?}");
            all.push((vs[0].page, uv));
        }
    }
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            if all[i].0 == all[j].0 {
                assert!(!overlap(&all[i].1, &all[j].1), "triangles {i} and {j} overlap");
            }
        }
    }
}

fn run(meshes: &[&Mesh], opts: &Options) -> Atlas {
    let inputs: Vec<MeshInput> = meshes.iter().map(|m| m.input()).collect();
    let atlas = generate(&inputs, opts).expect("generate failed");
    check(&atlas, meshes);
    atlas
}

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
    let (lo, hi) = areas.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
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
    let charts_a: std::collections::HashSet<_> = atlas.meshes[0].vertices.iter().map(|v| v.chart).collect();
    let charts_b: std::collections::HashSet<_> = atlas.meshes[1].vertices.iter().map(|v| v.chart).collect();
    assert!(charts_a.is_disjoint(&charts_b));
}

#[test]
fn fixed_texel_density() {
    let m = plane(4); // 2 x 1 units
    let opts = Options { texels_per_unit: Some(64.0), resolution: 0, padding: 1, ..Options::default() };
    let atlas = run(&[&m], &opts);
    assert_eq!(atlas.texels_per_unit, 64.0);
    assert_eq!((atlas.width.max(atlas.height), atlas.width.min(atlas.height)), (130, 66));
    assert_eq!(atlas.page_count, 1);
}

#[test]
fn fixed_density_spills_onto_pages() {
    let m = cube();
    let opts = Options { texels_per_unit: Some(30.0), resolution: 64, padding: 1, ..Options::default() };
    let atlas = run(&[&m], &opts);
    assert!(atlas.page_count > 1);
}

#[test]
fn chart_too_large() {
    let m = plane(2);
    let opts = Options { texels_per_unit: Some(1000.0), resolution: 64, ..Options::default() };
    assert!(matches!(generate(&[m.input()], &opts), Err(Error::ChartTooLarge { .. })));
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
    assert!(last.chart.is_some(), "degenerate triangle should reuse an existing UV");
    assert!(out.vertices.iter().any(|v| v.xref == lonely && v.chart.is_none()));
}

#[test]
fn empty_input() {
    let atlas = generate(&[], &Options::default()).unwrap();
    assert_eq!((atlas.chart_count, atlas.page_count), (0, 0));
}

#[test]
fn input_errors() {
    let p = [[0.0f32; 3]; 3];
    let bad_count = MeshInput { positions: &p, indices: &[0, 1] };
    let bad_index = MeshInput { positions: &p, indices: &[0, 1, 3] };
    let nan = [[f32::NAN, 0.0, 0.0], [0.0; 3], [0.0; 3]];
    let bad_pos = MeshInput { positions: &nan, indices: &[0, 1, 2] };
    let o = Options::default();
    assert!(matches!(generate(&[bad_count], &o), Err(Error::InvalidIndexCount { .. })));
    assert!(matches!(generate(&[bad_index], &o), Err(Error::IndexOutOfRange { index: 3, .. })));
    assert!(matches!(generate(&[bad_pos], &o), Err(Error::NonFinitePosition { vertex: 0, .. })));
    let bad_opt = Options { max_chart_angle: 95.0, ..o };
    assert!(matches!(generate(&[], &bad_opt), Err(Error::InvalidOption(_))));
}
