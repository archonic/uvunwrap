//! Chart parameterization.
//!
//! Each chart is flattened with planar projection (if it is already flat), else
//! Least Squares Conformal Maps (Lévy et al. 2002) refined by as-rigid-as-possible
//! iterations (Liu et al. 2008), falling back to the plain LSCM result and then
//! planar projection. A result is only accepted if it has no flipped triangles, no
//! self-overlap and bounded area distortion; otherwise the chart is split in
//! two and both halves are retried (large, strongly curved charts are split
//! without trying). Single triangles always succeed, so this
//! always terminates with a valid parameterization.

use crate::geom::{
    add3, angle, bounds2, convex_hull, cross3, dot3, len3, min_area_rect_dir, normalize3, rotate,
    scale3, signed_area, sub3, triangles_overlap, V2,
};
use crate::par;
use crate::progress::{Ctx, Stage};
use crate::solver::{least_squares, Factored, SparseRows};
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

/// What every chart of one mesh is parameterized with.
struct Job<'a> {
    topo: &'a Topology,
    max_distortion: f64,
    ctx: &'a Ctx<'a>,
    total: usize,
}

impl Job<'_> {
    /// Reports the chart's faces as done and finalizes it.
    fn done(
        &self,
        faces: Vec<u32>,
        verts: Vec<u32>,
        local: Vec<[u32; 3]>,
        uvs: Vec<V2>,
    ) -> Vec<Chart> {
        // A cancellation is noticed by the caller.
        let _ = self
            .ctx
            .advance(Stage::Parameterizing, faces.len(), Some(self.total));
        vec![finalize(self.topo, faces, verts, local, uvs)]
    }
}

/// Parameterizes every group in parallel, reporting finished faces to `ctx`
/// out of `total` (faces of all meshes). With `face_uvs` (the input UVs of
/// each face's corners), groups are the mesh's UV islands and are kept as
/// they are when valid. Returns nothing useful once cancelled.
pub(crate) fn parameterize(
    topo: &Topology,
    groups: Vec<Vec<u32>>,
    face_uvs: Option<&[[V2; 3]]>,
    max_distortion: f64,
    ctx: &Ctx,
    total: usize,
) -> Vec<Chart> {
    let job = Job {
        topo,
        max_distortion,
        ctx,
        total,
    };
    par::flat_map(groups, |g| match face_uvs {
        Some(uvs) => island(&job, g, uvs),
        None => parameterize_group(&job, g),
    })
}

/// Keeps a UV island as one chart (mirrored islands flipped back), unless it
/// is unusable as is (flipped or degenerate triangles, self-overlap), in
/// which case it is parameterized like any group.
fn island(job: &Job, faces: Vec<u32>, face_uvs: &[[V2; 3]]) -> Vec<Chart> {
    if job.ctx.is_cancelled() {
        return Vec::new();
    }
    // Local vertices by (welded vertex, UV): a vertex keeps one UV per island
    // even where the island touches itself at a vertex.
    let mut map: HashMap<(u32, [u64; 2]), u32> = HashMap::with_capacity(faces.len());
    let (mut verts, mut uvs) = (Vec::new(), Vec::new());
    let local: Vec<[u32; 3]> = faces
        .iter()
        .map(|&f| {
            let w = job.topo.faces[f as usize];
            [0, 1, 2].map(|c| {
                let uv = face_uvs[f as usize][c];
                // `+ 0.0` turns -0.0 into 0.0, so both key the same vertex.
                *map.entry((w[c], uv.map(|x| (x + 0.0).to_bits())))
                    .or_insert_with(|| {
                        verts.push(w[c]);
                        uvs.push(uv);
                        (verts.len() - 1) as u32
                    })
            })
        })
        .collect();
    if total_area(&local, &uvs) < 0.0 {
        // Mirrored island: flip it so its triangles wind counter-clockwise.
        uvs.iter_mut().for_each(|p| p[0] = -p[0]);
    }
    // Authored UVs are kept however distorted; only validity is required.
    if validate(job.topo, &faces, &local, &uvs, f64::INFINITY).is_ok() {
        return job.done(faces, verts, local, uvs);
    }
    parameterize_group(job, faces)
}

/// Why a parameterization was rejected.
enum Rejection {
    /// These two chart faces overlap: a good place to split.
    Overlap(u32, u32),
    /// Flipped or degenerate triangles, or too much distortion.
    Invalid,
}

/// Total signed UV area of a chart's triangles.
fn total_area(local: &[[u32; 3]], uvs: &[V2]) -> f64 {
    local
        .iter()
        .map(|t| signed_area(uvs[t[0] as usize], uvs[t[1] as usize], uvs[t[2] as usize]))
        .sum()
}

/// Charts at least this large are checked for curvature before solving.
const CURVATURE_CHECK_FACES: usize = 1000;
/// Total Gaussian curvature (angle deficit, radians) above which a large chart
/// is split without attempting LSCM: such charts almost never flatten within
/// the distortion limit, and solving them is the main cost on big meshes.
const MAX_CURVATURE: f64 = 3.0;

fn parameterize_group(job: &Job, faces: Vec<u32>) -> Vec<Chart> {
    if job.ctx.is_cancelled() {
        return Vec::new();
    }
    let (topo, max_distortion) = (job.topo, job.max_distortion);
    let (verts, local) = build_local(topo, &faces);
    let planar = planar_projection(topo, &faces, &verts);
    if faces.len() == 1 || validate(topo, &faces, &local, &planar, 1.02).is_ok() {
        return job.done(faces, verts, local, planar);
    }
    let hopeless = faces.len() >= CURVATURE_CHECK_FACES
        && total_curvature(topo, &verts, &local).abs() > MAX_CURVATURE;
    let mut overlap = None;
    if !hopeless {
        let conformal = lscm(topo, &verts, &local, &planar);
        if job.ctx.is_cancelled() {
            return Vec::new(); // the solve can't be interrupted, but validation can be skipped
        }
        let conformal_valid = validate(topo, &faces, &local, &conformal, max_distortion);
        if let Err(Rejection::Overlap(a, b)) = conformal_valid {
            overlap = Some((a, b));
        }
        // ARAP also gets LSCM results rejected for flips, overlaps or
        // distortion: it often fixes them, which saves a split.
        if conformal
            .iter()
            .all(|p| p[0].is_finite() && p[1].is_finite())
        {
            let accept = |uv: &[V2]| validate(topo, &faces, &local, uv, max_distortion).is_ok();
            match arap(topo, &verts, &local, &conformal, accept) {
                Ok(uvs) => return job.done(faces, verts, local, uvs),
                Err(last) if conformal_valid.is_ok() => {
                    if mean_distortion(topo, &faces, &local, &conformal) <= SPLIT_DISTORTION {
                        return job.done(faces, verts, local, conformal);
                    }
                    // Too distorted to keep: split where ARAP's result overlaps.
                    if let Err(Rejection::Overlap(a, b)) =
                        validate(topo, &faces, &local, &last, max_distortion)
                    {
                        overlap = Some((a, b));
                    }
                }
                Err(_) => {}
            }
        }
        if conformal_valid.is_err()
            && validate(topo, &faces, &local, &planar, max_distortion).is_ok()
        {
            return job.done(faces, verts, local, planar);
        }
    }
    // Overlaps usually come from a local fold in the surface, which no
    // flattening of this chart can fix: split between the two overlapping
    // triangles so a single split separates them.
    let (a, b) = split(topo, &faces, overlap);
    let (mut a, b) = par::join(|| parameterize_group(job, a), || parameterize_group(job, b));
    a.extend(b);
    a
}

/// Total Gaussian curvature of the chart: the sum of angle deficits
/// `2π - Σ angles` over its interior vertices.
fn total_curvature(topo: &Topology, verts: &[u32], local: &[[u32; 3]]) -> f64 {
    let n = verts.len();
    // Sum of the corner angles at each vertex.
    let mut angle_sum = vec![0.0; n];
    // Edge use count parity marks boundary edges (used by one face).
    let mut edges: HashMap<(u32, u32), bool> = HashMap::with_capacity(local.len() * 2);
    for t in local {
        let p = t.map(|l| topo.positions[verts[l as usize] as usize]);
        for k in 0..3 {
            let a = sub3(p[(k + 1) % 3], p[k]);
            let b = sub3(p[(k + 2) % 3], p[k]);
            angle_sum[t[k] as usize] += angle(len3(cross3(a, b)), dot3(a, b));
            let (u, v) = (t[k], t[(k + 1) % 3]);
            let e = edges.entry((u.min(v), u.max(v))).or_insert(false);
            *e = !*e;
        }
    }
    let mut boundary = vec![false; n];
    for (&(u, v), &odd) in &edges {
        if odd {
            boundary[u as usize] = true;
            boundary[v as usize] = true;
        }
    }
    (0..n)
        .filter(|&i| !boundary[i])
        .map(|i| std::f64::consts::TAU - angle_sum[i])
        .sum()
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
        nsum = add3(
            nsum,
            scale3(topo.normals[f as usize], topo.areas[f as usize]),
        );
    }
    let n = normalize3(nsum).unwrap_or(topo.normals[faces[0] as usize]);
    let t = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
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
    let [lo, hi] = bounds2(init.iter().copied());
    let axis = if hi[0] - lo[0] >= hi[1] - lo[1] { 0 } else { 1 };
    let pin_a = (0..n)
        .min_by(|&a, &b| init[a][axis].total_cmp(&init[b][axis]))
        .unwrap();
    let pin_b = (0..n)
        .max_by(|&a, &b| init[a][axis].total_cmp(&init[b][axis]))
        .unwrap();
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
            [
                (i0, 0, zc - za),
                (i0, 1, zb - zd),
                (i1, 0, -zc),
                (i1, 1, zd),
                (i2, 0, za),
                (i2, 1, -zb),
            ],
            // Imaginary part.
            [
                (i0, 0, zd - zb),
                (i0, 1, zc - za),
                (i1, 0, -zd),
                (i1, 1, -zc),
                (i2, 0, zb),
                (i2, 1, za),
            ],
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
        .map(|i| {
            if col[i] == NONE {
                init[i]
            } else {
                [x[col[i] as usize], x[col[i] as usize + 1]]
            }
        })
        .collect()
}

/// Mean area distortion ([`mean_distortion`]) above which a valid LSCM result
/// that [`arap`] can't improve validly is split rather than kept. Measured
/// results are the same anywhere in 1.05–1.1; chart counts barely change.
const SPLIT_DISTORTION: f64 = 1.05;

/// Area-weighted mean of max(r, 1/r) over the chart's faces, r being a face's
/// share of UV area over its share of surface area (1 = no area distortion).
fn mean_distortion(topo: &Topology, faces: &[u32], local: &[[u32; 3]], uv: &[V2]) -> f64 {
    let total_uv = total_area(local, uv);
    let total_3d: f64 = faces.iter().map(|&f| topo.areas[f as usize]).sum();
    let scale = total_3d / total_uv;
    let sum: f64 = local
        .iter()
        .zip(faces)
        .map(|(t, &f)| {
            let a = topo.areas[f as usize];
            let r =
                signed_area(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]) * scale / a;
            if a > 0.0 {
                a * r.max(1.0 / r)
            } else {
                0.0
            }
        })
        .sum();
    sum / total_3d
}

/// Local/global iterations of [`arap`].
const ARAP_ITERATIONS: usize = 10;
/// CGLS iterations per [`arap`] global step without the direct solver. Each
/// step is warm-started from the last and only has to lower the energy, not
/// converge: 100 gives the same results as solving to tolerance on the test
/// models, and 30 is measurably worse.
const ARAP_CGLS_ITERATIONS: usize = 100;

/// As-rigid-as-possible refinement (Liu et al. 2008) of a parameterization:
/// alternately fits each triangle's closest rotation to its current Jacobian
/// (local step) and solves for the UVs whose Jacobians best match those
/// rotations, area-weighted (global step: one factorization, shared by u and
/// v). LSCM preserves angles but lets scale drift across a curved chart; this
/// evens out the area distortion at a small cost in angle distortion.
/// Iterates can flip or overlap, so the result is the latest flip-free
/// iterate that `accept`s, trying only the latest and the first (validation
/// is the expensive part); if neither is accepted, `Err` with the last
/// iterate (to find where it overlaps).
fn arap(
    topo: &Topology,
    verts: &[u32],
    local: &[[u32; 3]],
    init: &[V2],
    accept: impl Fn(&[V2]) -> bool,
) -> Result<Vec<V2>, Vec<V2>> {
    let n = verts.len();
    if n < 3 {
        return Err(init.to_vec());
    }
    // Per triangle, the rows of its gradient operator in its own frame
    // (p0 = 0, p1 = (x1, 0), p2 = (x2, y2)), scaled by sqrt(area), and that
    // weight. The energy is Σ area · |J - R|², always positive semidefinite
    // (cotangent weights aren't, on obtuse triangles).
    let grads: Vec<([f64; 3], [f64; 3], f64)> = local
        .iter()
        .map(|t| {
            let [p0, p1, p2] = t.map(|l| topo.positions[verts[l as usize] as usize]);
            let e1 = sub3(p1, p0);
            let e2 = sub3(p2, p0);
            let two_area = len3(cross3(e1, e2));
            let x1 = len3(e1);
            if !(two_area > 0.0) || !(x1 > 0.0) {
                return ([0.0; 3], [0.0; 3], 0.0);
            }
            let x2 = dot3(e2, scale3(e1, 1.0 / x1));
            let y2 = two_area / x1;
            let w = (two_area / 2.0).sqrt();
            let gx = [-1.0 / x1, 1.0 / x1, 0.0];
            let gy = [(x2 - x1) / (x1 * y2), -x2 / (x1 * y2), 1.0 / y2];
            (gx.map(|g| g * w), gy.map(|g| g * w), w)
        })
        .collect();
    // Start from `init` at the 3D scale.
    let area_3d: f64 = grads.iter().map(|g| g.2 * g.2).sum();
    let area_uv = total_area(local, init);
    if !(area_uv > 0.0) {
        return Err(init.to_vec());
    }
    let k = (area_3d / area_uv).sqrt();
    let mut uv: Vec<V2> = init.iter().map(|p| [p[0] * k, p[1] * k]).collect();

    // Local vertex 0 is pinned (translation); the others are columns i - 1.
    let mut a = SparseRows::new(n - 1);
    for (t, g) in local.iter().zip(&grads) {
        for row in [&g.0, &g.1] {
            for (c, &coef) in row.iter().enumerate() {
                if t[c] != 0 && coef != 0.0 {
                    a.push(t[c] as usize - 1, coef);
                }
            }
            a.end_row();
        }
    }
    let solver = Factored::new(a, ARAP_CGLS_ITERATIONS);
    let (mut bu, mut bv) = (vec![0.0; local.len() * 2], vec![0.0; local.len() * 2]);
    let (mut xu, mut xv) = (vec![0.0; n - 1], vec![0.0; n - 1]);
    // Flip-free iterates: the first, and the latest.
    let (mut first, mut latest) = (None, None);
    for _ in 0..ARAP_ITERATIONS {
        for (f, (t, g)) in local.iter().zip(&grads).enumerate() {
            let q = t.map(|l| uv[l as usize]);
            let d =
                |row: &[f64; 3], c: usize| row[0] * q[0][c] + row[1] * q[1][c] + row[2] * q[2][c];
            // sqrt(area) · Jacobian [[du/dx, du/dy], [dv/dx, dv/dy]].
            let (ux, uy, vx, vy) = (d(&g.0, 0), d(&g.1, 0), d(&g.0, 1), d(&g.1, 1));
            // Its closest rotation [[c, -s], [s, c]].
            let (c, s) = (ux + vy, vx - uy);
            let r = (c * c + s * s).sqrt();
            let (c, s) = if r > 0.0 { (c / r, s / r) } else { (1.0, 0.0) };
            // The pinned vertex's part moves to the right-hand side.
            let pinned = |row: &[f64; 3], comp: usize| {
                (0..3)
                    .filter(|&k| t[k] == 0)
                    .map(|k| row[k] * uv[0][comp])
                    .sum::<f64>()
            };
            let w = g.2;
            bu[2 * f] = w * c - pinned(&g.0, 0);
            bu[2 * f + 1] = -w * s - pinned(&g.1, 0);
            bv[2 * f] = w * s - pinned(&g.0, 1);
            bv[2 * f + 1] = w * c - pinned(&g.1, 1);
        }
        for i in 1..n {
            (xu[i - 1], xv[i - 1]) = (uv[i][0], uv[i][1]);
        }
        solver.solve(&bu, &mut xu);
        solver.solve(&bv, &mut xv);
        for i in 1..n {
            uv[i] = [xu[i - 1], xv[i - 1]];
        }
        let flipped = local.iter().zip(&grads).any(|(t, g)| {
            g.2 > 0.0
                && !(signed_area(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]) > 0.0)
        });
        if !flipped {
            if first.is_none() {
                first = Some(uv.clone());
            } else {
                latest = Some(uv.clone());
            }
        }
    }
    [latest, first]
        .into_iter()
        .flatten()
        .find(|uv| accept(uv))
        .ok_or(uv)
}

/// Accepts a parameterization only if no triangle is flipped or degenerate, the
/// per-triangle area ratio (after normalizing total area) stays within
/// `[1/max_distortion, max_distortion]`, and no two triangles overlap.
fn validate(
    topo: &Topology,
    faces: &[u32],
    local: &[[u32; 3]],
    uv: &[V2],
    max_distortion: f64,
) -> Result<(), Rejection> {
    if uv.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return Err(Rejection::Invalid);
    }
    let areas: Vec<f64> = local
        .iter()
        .map(|t| signed_area(uv[t[0] as usize], uv[t[1] as usize], uv[t[2] as usize]))
        .collect();
    let total_uv: f64 = areas.iter().sum();
    let total_3d: f64 = faces.iter().map(|&f| topo.areas[f as usize]).sum();
    if !(total_uv > 0.0) || !(total_3d > 0.0) {
        return Err(Rejection::Invalid);
    }
    let min_area = total_uv * 1e-12;
    let scale = total_3d / total_uv;
    for (k, &f) in faces.iter().enumerate() {
        if areas[k] <= min_area {
            return Err(Rejection::Invalid);
        }
        let ratio = areas[k] * scale / topo.areas[f as usize];
        if ratio > max_distortion || ratio * max_distortion < 1.0 {
            return Err(Rejection::Invalid);
        }
    }
    first_overlap(local, uv, total_uv).map_or(Ok(()), Err)
}

/// Grid-accelerated pairwise triangle overlap test: the first overlapping pair
/// of chart faces, if any.
fn first_overlap(local: &[[u32; 3]], uv: &[V2], total_area: f64) -> Option<Rejection> {
    let n = local.len();
    if n < 2 {
        return None;
    }
    let tris: Vec<[V2; 3]> = local.iter().map(|t| t.map(|l| uv[l as usize])).collect();
    let boxes: Vec<[V2; 2]> = tris
        .iter()
        .map(|t| {
            [
                [
                    t[0][0].min(t[1][0]).min(t[2][0]),
                    t[0][1].min(t[1][1]).min(t[2][1]),
                ],
                [
                    t[0][0].max(t[1][0]).max(t[2][0]),
                    t[0][1].max(t[1][1]).max(t[2][1]),
                ],
            ]
        })
        .collect();
    let [lo, hi] = bounds2(uv.iter().copied());
    let cell =
        (2.0 * (total_area / n as f64).sqrt()).max((hi[0] - lo[0]).max(hi[1] - lo[1]) / 1024.0);
    if !(cell > 0.0) {
        return Some(Rejection::Invalid); // degenerate layout
    }
    let gx = (((hi[0] - lo[0]) / cell) as usize + 1).min(1024);
    let gy = (((hi[1] - lo[1]) / cell) as usize + 1).min(1024);
    let cell_of = |p: V2| {
        (
            (((p[0] - lo[0]) / cell) as usize).min(gx - 1),
            (((p[1] - lo[1]) / cell) as usize).min(gy - 1),
        )
    };
    // Bucket triangles by the grid cells their bounding boxes cover (CSR).
    let mut start = vec![0u32; gx * gy + 1];
    for b in &boxes {
        let ((x0, y0), (x1, y1)) = (cell_of(b[0]), cell_of(b[1]));
        for y in y0..=y1 {
            for x in x0..=x1 {
                start[y * gx + x + 1] += 1;
            }
        }
    }
    for c in 0..gx * gy {
        start[c + 1] += start[c];
    }
    let mut fill = start.clone();
    let mut items = vec![0u32; start[gx * gy] as usize];
    for (i, b) in boxes.iter().enumerate() {
        let ((x0, y0), (x1, y1)) = (cell_of(b[0]), cell_of(b[1]));
        for y in y0..=y1 {
            for x in x0..=x1 {
                items[fill[y * gx + x] as usize] = i as u32;
                fill[y * gx + x] += 1;
            }
        }
    }
    let eps = total_area.sqrt() * 1e-9;
    for y in 0..gy {
        for x in 0..gx {
            let c = &items[start[y * gx + x] as usize..start[y * gx + x + 1] as usize];
            for (k, &a) in c.iter().enumerate() {
                let ba = boxes[a as usize];
                for &b in &c[k + 1..] {
                    let bb = boxes[b as usize];
                    if ba[1][0] < bb[0][0]
                        || bb[1][0] < ba[0][0]
                        || ba[1][1] < bb[0][1]
                        || bb[1][1] < ba[0][1]
                    {
                        continue;
                    }
                    // Test each pair once: in the cell holding the minimum
                    // corner of the two boxes' intersection.
                    if cell_of([ba[0][0].max(bb[0][0]), ba[0][1].max(bb[0][1])]) != (x, y) {
                        continue;
                    }
                    if triangles_overlap(&tris[a as usize], &tris[b as usize], eps) {
                        return Some(Rejection::Overlap(a.min(b), a.max(b)));
                    }
                }
            }
        }
    }
    None
}

/// Splits a chart into two edge-connected halves: a graph-distance Voronoi
/// split between two seed faces (chart face indices), by default two mutually
/// distant ones. Polygons stay whole unless the chart is a single polygon.
fn split(topo: &Topology, faces: &[u32], seeds: Option<(u32, u32)>) -> (Vec<u32>, Vec<u32>) {
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
    let farthest = |d: &[u32]| {
        (0..d.len())
            .filter(|&i| d[i] != u32::MAX)
            .max_by_key(|&i| d[i])
            .unwrap_or(0)
    };
    let (da, db) = match seeds {
        Some((a, b)) if a != b => (bfs(a as usize), bfs(b as usize)),
        _ => {
            let a = farthest(&bfs(0));
            let da = bfs(a);
            let b = farthest(&da);
            let db = bfs(b);
            (da, db)
        }
    };
    // Each polygon goes to the side of its first face here.
    let mut polygon_side: HashMap<u32, bool> = HashMap::new();
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for (i, &f) in faces.iter().enumerate() {
        let side = *polygon_side
            .entry(topo.polygon[f as usize])
            .or_insert(da[i] <= db[i]);
        if side {
            left.push(f);
        } else {
            right.push(f);
        }
    }
    if left.is_empty() || right.is_empty() {
        // A single polygon, or a disconnected or otherwise degenerate case:
        // split the list.
        let mid = faces.len() / 2;
        return (faces[..mid].to_vec(), faces[mid..].to_vec());
    }
    (left, right)
}

/// Scales the chart to match its 3D area, rotates it to its minimum-area
/// bounding rectangle and moves it to the origin.
fn finalize(
    topo: &Topology,
    faces: Vec<u32>,
    verts: Vec<u32>,
    face_local: Vec<[u32; 3]>,
    mut uvs: Vec<V2>,
) -> Chart {
    let area: f64 = faces.iter().map(|&f| topo.areas[f as usize]).sum();
    let uv_area = total_area(&face_local, &uvs);
    let k = if uv_area > 0.0 {
        (area / uv_area).sqrt()
    } else {
        1.0
    };
    let (c, s) = min_area_rect_dir(&convex_hull(&uvs));
    for p in uvs.iter_mut() {
        let r = rotate(*p, c, s);
        *p = [r[0] * k, r[1] * k];
    }
    let [lo, hi] = bounds2(uvs.iter().copied());
    for p in uvs.iter_mut() {
        *p = [p[0] - lo[0], p[1] - lo[1]];
    }
    Chart {
        faces,
        verts,
        face_local,
        uvs,
        size: [hi[0] - lo[0], hi[1] - lo[1]],
        area,
    }
}
