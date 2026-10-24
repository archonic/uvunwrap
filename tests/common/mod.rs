//! Shared test helpers: mesh generators, a seeded random generator, and
//! [`check`], which verifies every guarantee the crate makes about an atlas.
//!
//! Where each guarantee is tested:
//!
//! | Guarantee | Tests |
//! |---|---|
//! | topology kept, valid UVs, no overlaps, padding to charts and page edges | `check`, used by nearly every test; `robustness.rs` runs it on hostile and random input |
//! | area distortion within `max_area_distortion` | `unwrap.rs`: `area_distortion_stays_within_the_limit` |
//! | determinism across runs and thread counts | `unwrap.rs`: `output_is_deterministic` |
//! | authored seams, materials, ignored faces, scale, existing UVs | `inputs.rs` |
//! | polygons stay whole: one UV per polygon corner | `check`; `polygons.rs` |
//! | fixed density, pages, `max_chart_size`, block alignment, every packer option | `packing.rs` |
//! | instance rectangles, page limits, scale ratios | `instances.rs` |
//! | progress order and cancellation | `progress.rs` |

#![allow(dead_code)] // each test binary uses a different subset

use std::collections::HashMap;
use std::f32::consts::TAU;
#[allow(unused_imports)]
pub use uvunwrap::{
    generate, Atlas, Density, Error, InstanceRect, MeshInput, Options, OutputMesh, Packing,
};

pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn input(&self) -> MeshInput<'_> {
        MeshInput {
            positions: &self.positions,
            indices: &self.indices,
            ..Default::default()
        }
    }

    pub fn faces(&self) -> usize {
        self.indices.len() / 3
    }
}

// ---------------------------------------------------------------- generators

/// Parametric surface sampled on a grid, optionally wrapping in u and/or v.
pub fn surface(
    nu: u32,
    nv: u32,
    wrap_u: bool,
    wrap_v: bool,
    f: impl Fn(f32, f32) -> [f32; 3],
) -> Mesh {
    let (cu, cv) = (
        if wrap_u { nu } else { nu + 1 },
        if wrap_v { nv } else { nv + 1 },
    );
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

/// A flat 2 × 1 grid of `n × n` quads in the z = 0 plane.
pub fn plane(n: u32) -> Mesh {
    surface(n, n, false, false, |u, v| [u * 2.0, v, 0.0])
}

pub fn torus(nu: u32, nv: u32) -> Mesh {
    surface(nu, nv, true, true, |u, v| {
        let (a, b) = (u * TAU, v * TAU);
        let r = 1.0 + 0.4 * b.cos();
        [r * a.cos(), r * a.sin(), 0.4 * b.sin()]
    })
}

/// Poles are duplicated per column, producing degenerate triangles there.
pub fn sphere(nu: u32, nv: u32) -> Mesh {
    surface(nu, nv, true, false, |u, v| {
        let (a, b) = (u * TAU, (v - 0.5) * std::f32::consts::PI);
        [b.cos() * a.cos(), b.cos() * a.sin(), b.sin()]
    })
}

pub fn cylinder(n: u32) -> Mesh {
    surface(n, 4, true, false, |u, v| {
        [(u * TAU).cos(), (u * TAU).sin(), v * 3.0]
    })
}

/// Cube with split vertices per face (24 vertices), like a typical exported mesh.
pub fn cube() -> Mesh {
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

/// The same triangles with every corner its own vertex (flat shading).
pub fn soup(m: &Mesh) -> Mesh {
    Mesh {
        positions: m.indices.iter().map(|&i| m.positions[i as usize]).collect(),
        indices: (0..m.indices.len() as u32).collect(),
    }
}

/// Faces of `m` whose centroid is right of `x`.
pub fn right_of(m: &Mesh, x: f32) -> Vec<bool> {
    m.indices
        .chunks_exact(3)
        .map(|t| t.iter().map(|&i| m.positions[i as usize][0]).sum::<f32>() / 3.0 > x)
        .collect()
}

/// `plane(12)` with its vertices split down the middle (x = 1), as an
/// exporter does for a UV seam: the right half's faces use copies of the
/// seam vertices, appended after the originals.
pub fn split_plane() -> Mesh {
    let mut m = plane(12);
    let right = right_of(&m, 1.0);
    let mut copies = std::collections::HashMap::new();
    for (f, t) in m.indices.chunks_exact_mut(3).enumerate() {
        for i in t.iter_mut().filter(|_| right[f]) {
            if m.positions[*i as usize][0] == 1.0 {
                *i = *copies.entry(*i).or_insert_with(|| {
                    m.positions.push(m.positions[*i as usize]);
                    (m.positions.len() - 1) as u32
                });
            }
        }
    }
    m
}

/// xorshift64*: a small deterministic generator for randomized tests.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

// ------------------------------------------------------------- measurements

/// Signed area of a 2D triangle (positive = counter-clockwise in a y-up frame).
pub fn tri_area(p: [[f64; 2]; 3]) -> f64 {
    0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
}

/// UV area of one face of an output mesh.
pub fn uv_area(out: &OutputMesh, face: usize) -> f64 {
    tri_area([0, 1, 2].map(|c| {
        out.vertices[out.indices[face * 3 + c] as usize]
            .uv
            .map(f64::from)
    }))
}

/// Total UV area of an output mesh.
pub fn total_uv_area(out: &OutputMesh) -> f64 {
    (0..out.indices.len() / 3).map(|f| uv_area(out, f)).sum()
}

/// Per non-degenerate face of `m`: (face, r, 3D area), r being the face's share
/// of its chart's UV area over its share of the chart's surface area. Faces too
/// small for f32 UVs to measure are left out.
pub fn area_ratios(m: &Mesh, out: &OutputMesh) -> Vec<(usize, f64, f64)> {
    let area3 = |f: usize| {
        let p = [0, 1, 2].map(|c| m.positions[m.indices[f * 3 + c] as usize].map(f64::from));
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
    let mut totals: std::collections::HashMap<u32, (f64, f64)> = std::collections::HashMap::new();
    for f in 0..m.faces() {
        if let Some(c) = face_chart(out, f) {
            let t = totals.entry(c).or_default();
            t.0 += uv_area(out, f);
            t.1 += area3(f);
        }
    }
    (0..m.faces())
        .filter_map(|f| {
            let (uv_total, total) = totals[&face_chart(out, f)?];
            let (a, b) = (uv_area(out, f), area3(f));
            if b < total * 1e-6 || a < uv_total * 1e-6 {
                return None; // degenerate, or too small for f32 UVs to measure
            }
            Some((f, (a / uv_total) / (b / total), b))
        })
        .collect()
}

pub fn face_chart(out: &OutputMesh, face: usize) -> Option<u32> {
    out.vertices[out.indices[face * 3] as usize].chart
}

/// An instanced output mesh mapped into one instance's atlas rectangle.
pub fn instance_mesh(out: &OutputMesh, i: usize) -> OutputMesh {
    let r = out.instances[i];
    let mut m = out.clone();
    for v in &mut m.vertices {
        v.uv = [0, 1].map(|k| r.min[k] + v.uv[k] * (r.max[k] - r.min[k]));
        v.page = v.chart.map(|_| r.page);
    }
    m.instances.clear();
    m
}

/// Texel-space extent of a mesh's UVs: `(min, max)`.
pub fn texel_bounds(atlas: &Atlas, out: &OutputMesh) -> ([f64; 2], [f64; 2]) {
    let dims = [atlas.width as f64, atlas.height as f64];
    let mut lo = [f64::MAX; 2];
    let mut hi = [f64::MIN; 2];
    for v in &out.vertices {
        for k in 0..2 {
            lo[k] = lo[k].min(v.uv[k] as f64 * dims[k]);
            hi[k] = hi[k].max(v.uv[k] as f64 * dims[k]);
        }
    }
    (lo, hi)
}

// --------------------------------------------------------------------- check

fn dist(a: [f32; 3], b: [f32; 3]) -> f64 {
    ((0..3)
        .map(|k| (a[k] as f64 - b[k] as f64).powi(2))
        .sum::<f64>())
    .sqrt()
}

/// Altitude of a 3D triangle over its longest edge.
fn altitude(p: [[f32; 3]; 3]) -> f64 {
    let (a, b, c) = (dist(p[0], p[1]), dist(p[1], p[2]), dist(p[2], p[0]));
    let s = 0.5 * (a + b + c);
    let area = (s * (s - a) * (s - b) * (s - c)).max(0.0).sqrt();
    let longest = a.max(b).max(c);
    if longest > 0.0 {
        2.0 * area / longest
    } else {
        0.0
    }
}

fn bbox_diagonal(points: &[[f32; 3]]) -> f64 {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k] as f64);
            hi[k] = hi[k].max(p[k] as f64);
        }
    }
    if points.is_empty() {
        0.0
    } else {
        ((0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>()).sqrt()
    }
}

/// Convex hull (monotone chain), counter-clockwise.
fn hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut h: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = h.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(pts.iter())
        } else {
            Box::new(pts.iter().rev())
        };
        for &p in iter {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], p) <= 0.0 {
                h.pop();
            }
            h.push(p);
        }
        h.pop();
    }
    h
}

/// Whether two convex polygons' interiors overlap (touching doesn't count).
fn polygons_overlap(a: &[[f64; 2]], b: &[[f64; 2]]) -> bool {
    for poly in [a, b] {
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            let ax = [p[1] - q[1], q[0] - p[0]];
            let proj = |t: &[[f64; 2]]| {
                let d = t.iter().map(|v| v[0] * ax[0] + v[1] * ax[1]);
                d.clone().fold(f64::MAX, f64::min)..d.fold(f64::MIN, f64::max)
            };
            let (pa, pb) = (proj(a), proj(b));
            let eps = 1e-6 * (ax[0].abs() + ax[1].abs());
            if pa.end <= pb.start + eps || pb.end <= pa.start + eps {
                return false;
            }
        }
    }
    true
}

/// `t` grown by `r` in every axis direction (Minkowski sum with a square), so
/// overlapping it means being within Chebyshev distance `r` of `t`.
fn inflate(t: &[[f64; 2]; 3], r: f64) -> Vec<[f64; 2]> {
    hull(
        t.iter()
            .flat_map(|p| [[-r, -r], [r, -r], [r, r], [-r, r]].map(|d| [p[0] + d[0], p[1] + d[1]]))
            .collect(),
    )
}

/// A charted triangle in texel space.
struct Placed {
    page: u32,
    /// Distinct per (mesh or instance, chart).
    key: (usize, u32),
    tri: [[f64; 2]; 3],
    lo: [f64; 2],
    hi: [f64; 2],
}

/// Checks every guarantee of an atlas generated from `inputs` with `opts`:
///
/// - same triangles in the same order, `xref` pointing at the input vertex,
///   one rectangle per instance;
/// - every charted triangle has all corners on one chart and page and positive
///   UV area (unless it is too thin for f32 UVs), and lies at least `padding`
///   texels inside its page;
/// - triangles of the same chart don't overlap, and triangles of different
///   charts (or instances) are at least `2 * padding` texels apart;
/// - ignored faces have no UV area.
pub fn check(atlas: &Atlas, inputs: &[MeshInput], opts: &Options) {
    assert_eq!(atlas.meshes.len(), inputs.len());
    let pad = opts.padding as f64;
    let dims = [atlas.width as f64, atlas.height as f64];
    // f32 UVs and the rasterizer's 1e-3 texel growth.
    let eps = 0.01;
    let mut placed = Vec::new();
    let mut view_id = 0;
    for (out, input) in atlas.meshes.iter().zip(inputs) {
        assert_eq!(out.indices.len(), input.indices.len());
        for (k, &i) in out.indices.iter().enumerate() {
            assert_eq!(
                out.vertices[i as usize].xref, input.indices[k],
                "topology changed"
            );
        }
        assert_eq!(out.instances.len(), input.instance_scales.len());
        if let Some(polygons) = input.polygons {
            check_polygons(out, polygons);
        }
        // Faces the crate may leave uncharted or collapse: degenerate ones
        // (altitude below 1e-6 of the mesh size, as in topology.rs, with
        // margin) and ones too thin for f32 UVs.
        let diag = bbox_diagonal(input.positions);
        let views: Vec<Option<InstanceRect>> = if out.instances.is_empty() {
            vec![None]
        } else {
            out.instances.iter().copied().map(Some).collect()
        };
        for view in views {
            for (f, t) in out.indices.chunks_exact(3).enumerate() {
                let vs = [0, 1, 2].map(|c| out.vertices[t[c] as usize]);
                let uv = |v: &uvunwrap::OutputVertex| match view {
                    None => [v.uv[0] as f64 * dims[0], v.uv[1] as f64 * dims[1]],
                    Some(r) => [0, 1]
                        .map(|k| (r.min[k] + v.uv[k] * (r.max[k] - r.min[k])) as f64 * dims[k]),
                };
                let tri = vs.map(|v| uv(&v));
                if input.ignored_faces.is_some_and(|ig| ig[f]) {
                    assert_eq!(tri_area(tri), 0.0, "ignored face {f} has UV area");
                    continue;
                }
                let alt = altitude(vs.map(|v| input.positions[v.xref as usize]));
                if alt <= 2e-6 * diag {
                    continue;
                }
                let chart = vs[0].chart.expect("charted face");
                assert!(
                    vs.iter().all(|v| v.chart == Some(chart)),
                    "face {f} spans charts"
                );
                // Vertices of instanced meshes have no page of their own.
                let page = match view {
                    Some(r) => {
                        assert!(
                            vs.iter().all(|v| v.page.is_none()),
                            "face {f}: instanced vertex with a page"
                        );
                        r.page
                    }
                    None => {
                        let page = vs[0].page.expect("charted vertex without a page");
                        assert!(
                            vs.iter().all(|v| v.page == Some(page)),
                            "face {f} spans pages"
                        );
                        page
                    }
                };
                assert!(page < atlas.page_count.max(1));
                let area = tri_area(tri);
                if area == 0.0 && alt <= 1e-5 * diag {
                    continue; // collapsed on purpose: too thin for f32 UVs
                }
                assert!(area > 0.0, "face {f} flipped or collapsed: {tri:?}");
                for p in &tri {
                    for k in 0..2 {
                        assert!(
                            p[k] >= pad - eps && p[k] <= dims[k] - pad + eps,
                            "face {f} within {pad} texels of the page edge: {p:?}"
                        );
                    }
                }
                let lo = [0, 1].map(|k| tri.iter().map(|p| p[k]).fold(f64::MAX, f64::min));
                let hi = [0, 1].map(|k| tri.iter().map(|p| p[k]).fold(f64::MIN, f64::max));
                placed.push(Placed {
                    page,
                    key: (view_id, chart),
                    tri,
                    lo,
                    hi,
                });
            }
            view_id += 1;
        }
    }
    // Sweep over x: compare each triangle with those whose (grown) x ranges
    // overlap it.
    let gap = (2.0 * pad - eps).max(0.0);
    placed.sort_by(|a, b| (a.page, a.lo[0]).partial_cmp(&(b.page, b.lo[0])).unwrap());
    let mut active: Vec<usize> = Vec::new();
    for i in 0..placed.len() {
        let a = &placed[i];
        active.retain(|&j| placed[j].page == a.page && placed[j].hi[0] + gap > a.lo[0]);
        for &j in &active {
            let b = &placed[j];
            if b.hi[1] + gap <= a.lo[1] || a.hi[1] + gap <= b.lo[1] {
                continue;
            }
            if a.key == b.key {
                assert!(
                    !polygons_overlap(&a.tri, &b.tri),
                    "triangles of chart {:?} overlap",
                    a.key
                );
            } else {
                assert!(
                    !polygons_overlap(&inflate(&a.tri, gap), &b.tri),
                    "charts {:?} and {:?} are closer than {gap} texels on page {}",
                    a.key,
                    b.key,
                    a.page
                );
            }
        }
        active.push(i);
    }
}

/// Every polygon corner (polygon, input vertex) has exactly one UV, and a
/// polygon's charted triangles share one chart.
fn check_polygons(out: &OutputMesh, polygons: &[u32]) {
    let mut corner_uv: HashMap<(u32, u32), [u32; 2]> = HashMap::new();
    let mut chart: HashMap<u32, u32> = HashMap::new();
    for (f, t) in out.indices.chunks_exact(3).enumerate() {
        let p = polygons[f];
        for &i in t {
            let v = out.vertices[i as usize];
            let uv = v.uv.map(f32::to_bits);
            let seen = *corner_uv.entry((p, v.xref)).or_insert(uv);
            assert_eq!(
                seen, uv,
                "polygon {p}: vertex {} has two UVs (face {f})",
                v.xref
            );
            if let Some(c) = v.chart {
                let seen = *chart.entry(p).or_insert(c);
                assert_eq!(seen, c, "polygon {p} spans charts {seen} and {c}");
            }
        }
    }
}

/// Polygon ids pairing consecutive triangles: the quads of [`surface`],
/// [`cube`] and the generators built on them.
pub fn quads(m: &Mesh) -> Vec<u32> {
    (0..m.faces() as u32).map(|t| t / 2).collect()
}

/// Generates an atlas and checks it.
pub fn run_inputs(inputs: &[MeshInput], opts: &Options) -> Atlas {
    let atlas = generate(inputs, opts).expect("generate failed");
    check(&atlas, inputs, opts);
    atlas
}

/// Generates an atlas for plain meshes and checks it.
pub fn run(meshes: &[&Mesh], opts: &Options) -> Atlas {
    let inputs: Vec<MeshInput> = meshes.iter().map(|m| m.input()).collect();
    run_inputs(&inputs, opts)
}
