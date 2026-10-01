//! Chart parameterization.
//!
//! Each chart is flattened with planar projection (if it is already flat), else
//! Least Squares Conformal Maps (Lévy et al. 2002), falling back to planar
//! projection. A result is only accepted if it has no flipped triangles, no
//! self-overlap and bounded area distortion; otherwise the chart is split in
//! two and both halves are retried. Single triangles always succeed, so this
//! always terminates with a valid parameterization.

use crate::geom::*;
use crate::solver::{least_squares, SparseRows};
use crate::topology::{Topology, NONE};
use std::collections::{HashMap, VecDeque};

/// A parameterized chart. UVs are in world units (area-preserving on average),
/// rotated to their minimum-area bounding rectangle with the minimum at `(0, 0)`.
pub(crate) struct Chart {
    /// Face indices (into the mesh's faces).
    pub faces: Vec<u32>,
    /// Welded vertex ids, indexed by local vertex.
    pub verts: Vec<u32>,
    /// Per chart face, its local vertex indices.
    pub face_local: Vec<[u32; 3]>,
    /// Per local vertex UV.
    pub uvs: Vec<V2>,
    /// Extent of `uvs`.
    pub size: V2,
    /// Total surface area in world units.
    pub area: f64,
}

pub(crate) fn parameterize_all(topo: &Topology, groups: Vec<Vec<u32>>, max_distortion: f64) -> Vec<Chart> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        groups
            .into_par_iter()
            .flat_map_iter(|g| parameterize_group(topo, g, max_distortion))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        groups.into_iter().flat_map(|g| parameterize_group(topo, g, max_distortion)).collect()
    }
}

fn parameterize_group(topo: &Topology, faces: Vec<u32>, max_distortion: f64) -> Vec<Chart> {
    let mut out = Vec::new();
    let mut stack = vec![faces];
    while let Some(faces) = stack.pop() {
        let (verts, local) = build_local(topo, &faces);
        let planar = planar_projection(topo, &faces, &verts);
        if faces.len() == 1 || validate(topo, &faces, &local, &planar, 1.02) {
            out.push(finalize(topo, faces, verts, local, planar));
            continue;
        }
        let conformal = lscm(topo, &verts, &local, &planar);
        if validate(topo, &faces, &local, &conformal, max_distortion) {
            out.push(finalize(topo, faces, verts, local, conformal));
        } else if validate(topo, &faces, &local, &planar, max_distortion) {
            out.push(finalize(topo, faces, verts, local, planar));
        } else {
            let (a, b) = split(topo, &faces);
            stack.push(a);
            stack.push(b);
        }
    }
    out
}

fn build_local(topo: &Topology, faces: &[u32]) -> (Vec<u32>, Vec<[u32; 3]>) {
    let mut map: HashMap<u32, u32> = HashMap::with_capacity(faces.len());
    let mut verts = Vec::new();
    let local = faces
        .iter()
        .map(|&f| {
            topo.faces[f as usize].map(|v| {
                *map.entry(v).or_insert_with(|| {
                    verts.push(v);
                    (verts.len() - 1) as u32
                })
            })
        })
        .collect();
    (verts, local)
}

/// Orthographic projection onto the plane of the chart's average normal.
fn planar_projection(topo: &Topology, faces: &[u32], verts: &[u32]) -> Vec<V2> {
    let mut nsum = [0.0; 3];
    for &f in faces {
        nsum = add3(nsum, scale3(topo.normals[f as usize], topo.areas[f as usize]));
    }
    let n = normalize3(nsum).unwrap_or(topo.normals[faces[0] as usize]);
    let t = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = normalize3(sub3(t, scale3(n, dot3(t, n)))).unwrap_or([0.0, 0.0, 1.0]);
    let v = cross3(n, u); // (u, v, n) is right-handed, so front faces stay CCW
    verts
        .iter()
        .map(|&w| {
            let p = topo.positions[w as usize];
            [dot3(p, u), dot3(p, v)]
        })
        .collect()
}

/// Least Squares Conformal Maps. For each triangle with local 2D coordinates
/// z0 = 0, z1, z2 the conformality condition `z1 (U2 - U0) = z2 (U1 - U0)` on
/// complex UVs `U = u + iv` gives two real equations, weighted by 1/sqrt(2A).
/// Two vertices are pinned (to their planar-projected positions) to remove the
/// similarity-transform null space; the rest is a sparse least squares solve
/// (see [`least_squares`]), warm-started from the planar projection.
fn lscm(topo: &Topology, verts: &[u32], local: &[[u32; 3]], init: &[V2]) -> Vec<V2> {
    let n = verts.len();
    if n < 3 {
        return init.to_vec();
    }
    // Pin the two extreme vertices along the longest extent of the projection.
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in init {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let axis = if hi[0] - lo[0] >= hi[1] - lo[1] { 0 } else { 1 };
    let pin_a = (0..n).min_by(|&a, &b| init[a][axis].total_cmp(&init[b][axis])).unwrap();
    let pin_b = (0..n).max_by(|&a, &b| init[a][axis].total_cmp(&init[b][axis])).unwrap();
    if pin_a == pin_b {
        return init.to_vec();
    }

    // Column of u_i is col[i], of v_i is col[i] + 1; NONE for pinned vertices.
    let mut col = vec![NONE; n];
    let mut next = 0u32;
    for (i, c) in col.iter_mut().enumerate() {
        if i != pin_a && i != pin_b {
            *c = next;
            next += 2;
        }
    }
    let ncols = next as usize;
    let mut a = SparseRows::new(ncols);
    let mut rhs = Vec::with_capacity(local.len() * 2);

    for t in local {
        let [p0, p1, p2] = t.map(|l| topo.positions[verts[l as usize] as usize]);
        let e1 = sub3(p1, p0);
        let e2 = sub3(p2, p0);
        let nrm = cross3(e1, e2);
        let two_area = len3(nrm);
        let x1 = len3(e1);
        if !(two_area > 0.0) || !(x1 > 0.0) {
            continue;
        }
        let ax = scale3(e1, 1.0 / x1);
        let ay = cross3(scale3(nrm, 1.0 / two_area), ax);
        let (za, zb, zc, zd) = (x1, 0.0, dot3(e2, ax), dot3(e2, ay));
        let w = 1.0 / two_area.sqrt();
        let [i0, i1, i2] = t.map(|l| l as usize);
        let rows: [[(usize, usize, f64); 6]; 2] = [
            // Real part. Entries are (local vertex, component 0=u/1=v, coefficient).
            [(i0, 0, zc - za), (i0, 1, zb - zd), (i1, 0, -zc), (i1, 1, zd), (i2, 0, za), (i2, 1, -zb)],
            // Imaginary part.
            [(i0, 0, zd - zb), (i0, 1, zc - za), (i1, 0, -zd), (i1, 1, -zc), (i2, 0, zb), (i2, 1, za)],
        ];
        for row in rows {
            let mut b = 0.0;
            for (vi, comp, coef) in row {
                if coef == 0.0 {
                    continue;
                }
                if col[vi] == NONE {
                    b -= w * coef * init[vi][comp];
                } else {
                    a.push(col[vi] as usize + comp, w * coef);
                }
            }
            a.end_row();
            rhs.push(b);
        }
    }

    let mut x = vec![0.0; ncols];
    for i in 0..n {
        if col[i] != NONE {
            x[col[i] as usize] = init[i][0];
            x[col[i] as usize + 1] = init[i][1];
        }
    }
    least_squares(&mut a, &rhs, &mut x);

    (0..n)
        .map(|i| if col[i] == NONE { init[i] } else { [x[col[i] as usize], x[col[i] as usize + 1]] })
        .collect()
}

/// Accepts a parameterization only if no triangle is flipped or degenerate, the
/// per-triangle area ratio (after normalizing total area) stays within
/// `[1/max_distortion, max_distortion]`, and no two triangles overlap.
fn validate(topo: &Topology, faces: &[u32], local: &[[u32; 3]], uv: &[V2], max_distortion: f64) -> bool {
    if uv.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return false;
    }
    let areas: Vec<f64> = local.iter().map(|t| signed_area2(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize])).collect();
    let total_uv: f64 = areas.iter().sum();
    let total_3d: f64 = faces.iter().map(|&f| topo.areas[f as usize]).sum();
    if !(total_uv > 0.0) || !(total_3d > 0.0) {
        return false;
    }
    let min_area = total_uv * 1e-12;
    let scale = total_3d / total_uv;
    for (k, &f) in faces.iter().enumerate() {
        if areas[k] <= min_area {
            return false;
        }
        let ratio = areas[k] * scale / topo.areas[f as usize];
        if ratio > max_distortion || ratio * max_distortion < 1.0 {
            return false;
        }
    }
    !has_overlaps(local, uv, total_uv)
}

/// Grid-accelerated pairwise triangle overlap test.
fn has_overlaps(local: &[[u32; 3]], uv: &[V2], total_area: f64) -> bool {
    let n = local.len();
    if n < 2 {
        return false;
    }
    let tris: Vec<[V2; 3]> = local.iter().map(|t| t.map(|l| uv[l as usize])).collect();
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in uv {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let cell = (2.0 * (total_area / n as f64).sqrt()).max((hi[0] - lo[0]).max(hi[1] - lo[1]) / 1024.0);
    if !(cell > 0.0) {
        return true;
    }
    let gx = (((hi[0] - lo[0]) / cell) as usize + 1).min(1024);
    let gy = (((hi[1] - lo[1]) / cell) as usize + 1).min(1024);
    let cell_of = |v: f64, k: usize, g: usize| (((v - lo[k]) / cell) as usize).min(g - 1);
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); gx * gy];
    for (i, t) in tris.iter().enumerate() {
        let tlo = [t[0][0].min(t[1][0]).min(t[2][0]), t[0][1].min(t[1][1]).min(t[2][1])];
        let thi = [t[0][0].max(t[1][0]).max(t[2][0]), t[0][1].max(t[1][1]).max(t[2][1])];
        for y in cell_of(tlo[1], 1, gy)..=cell_of(thi[1], 1, gy) {
            for x in cell_of(tlo[0], 0, gx)..=cell_of(thi[0], 0, gx) {
                grid[y * gx + x].push(i as u32);
            }
        }
    }
    let mut pairs: Vec<u64> = Vec::new();
    for c in &grid {
        for (k, &a) in c.iter().enumerate() {
            for &b in &c[k + 1..] {
                pairs.push(((a.min(b) as u64) << 32) | a.max(b) as u64);
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let eps = total_area.sqrt() * 1e-9;
    pairs.iter().any(|&p| triangles_overlap(&tris[(p >> 32) as usize], &tris[(p & 0xffff_ffff) as usize], eps))
}

/// Splits a chart into two edge-connected halves: a graph-distance Voronoi
/// split between two mutually distant faces.
fn split(topo: &Topology, faces: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let index: HashMap<u32, usize> = faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let bfs = |start: usize| -> Vec<u32> {
        let mut dist = vec![u32::MAX; faces.len()];
        let mut queue = VecDeque::from([start]);
        dist[start] = 0;
        while let Some(i) = queue.pop_front() {
            for &nb in &topo.neighbors[faces[i] as usize] {
                if let Some(&j) = index.get(&nb) {
                    if dist[j] == u32::MAX {
                        dist[j] = dist[i] + 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        dist
    };
    let farthest = |d: &[u32]| (0..d.len()).filter(|&i| d[i] != u32::MAX).max_by_key(|&i| d[i]).unwrap_or(0);
    let a = farthest(&bfs(0));
    let da = bfs(a);
    let b = farthest(&da);
    let db = bfs(b);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for (i, &f) in faces.iter().enumerate() {
        if da[i] <= db[i] {
            left.push(f);
        } else {
            right.push(f);
        }
    }
    if left.is_empty() || right.is_empty() {
        // Disconnected or otherwise degenerate case: split the list.
        let mid = faces.len() / 2;
        return (faces[..mid].to_vec(), faces[mid..].to_vec());
    }
    (left, right)
}

/// Scales the chart to match its 3D area, rotates it to its minimum-area
/// bounding rectangle and moves it to the origin.
fn finalize(topo: &Topology, faces: Vec<u32>, verts: Vec<u32>, face_local: Vec<[u32; 3]>, mut uvs: Vec<V2>) -> Chart {
    let area: f64 = faces.iter().map(|&f| topo.areas[f as usize]).sum();
    let uv_area: f64 = face_local
        .iter()
        .map(|t| signed_area2(uvs[t[0] as usize], uvs[t[1] as usize], uvs[t[2] as usize]))
        .sum();
    let k = if uv_area > 0.0 { (area / uv_area).sqrt() } else { 1.0 };
    let theta = min_area_rect_angle(&convex_hull(&uvs));
    let (s, c) = theta.sin_cos();
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in uvs.iter_mut() {
        let r = rotate(*p, c, s);
        *p = [r[0] * k, r[1] * k];
        for i in 0..2 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    for p in uvs.iter_mut() {
        *p = [p[0] - lo[0], p[1] - lo[1]];
    }
    Chart { faces, verts, face_local, uvs, size: [hi[0] - lo[0], hi[1] - lo[1]], area }
}
