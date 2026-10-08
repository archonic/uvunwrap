//! Welded mesh topology: per-face geometry and manifold edge adjacency.

use crate::geom::{cross3, len3, scale3, sub3, V3};
use crate::MeshInput;
use std::collections::HashMap;

/// Sentinel for "no neighbor" / "unassigned".
pub(crate) const NONE: u32 = u32::MAX;

pub(crate) struct Topology {
    /// Original vertex index -> welded vertex index.
    pub welded: Vec<u32>,
    /// Welded vertex positions.
    pub positions: Vec<V3>,
    /// Faces as welded vertex indices.
    pub faces: Vec<[u32; 3]>,
    /// Unit face normals (zero for degenerate faces).
    pub normals: Vec<V3>,
    pub areas: Vec<f64>,
    /// Degenerate or ignored faces, which get no chart.
    pub excluded: Vec<bool>,
    /// `neighbors[f][e]` is the face across edge `(e, e+1)` of `f`, or [`NONE`].
    /// Only manifold, consistently oriented edges between non-excluded faces
    /// of the same material produce adjacency; everything else is treated as
    /// a boundary (a seam). So are edges where the input mesh is split (hard
    /// edges, UV seams), unless most edges are split (e.g. flat shading), or,
    /// when the mesh has input UVs, edges where those UVs differ.
    pub neighbors: Vec<[u32; 3]>,
}

#[inline]
fn key_bits(x: f32) -> u32 {
    // Treat -0.0 and 0.0 as the same position.
    if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

impl Topology {
    pub fn build(mesh: &MeshInput<'_>) -> Self {
        let (positions, indices) = (mesh.positions, mesh.indices);
        let scale = mesh.scale as f64;
        // Weld vertices with bit-identical positions so split normals/UVs don't
        // break connectivity.
        let mut map: HashMap<[u32; 3], u32> = HashMap::with_capacity(positions.len());
        let mut wpos: Vec<V3> = Vec::with_capacity(positions.len());
        let welded: Vec<u32> = positions
            .iter()
            .map(|p| {
                let key = [key_bits(p[0]), key_bits(p[1]), key_bits(p[2])];
                *map.entry(key).or_insert_with(|| {
                    wpos.push([
                        p[0] as f64 * scale,
                        p[1] as f64 * scale,
                        p[2] as f64 * scale,
                    ]);
                    (wpos.len() - 1) as u32
                })
            })
            .collect();

        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in &wpos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let diag = if wpos.is_empty() {
            0.0
        } else {
            len3(sub3(hi, lo))
        };
        // Triangles thinner than this can't be represented in f32 UV space.
        let min_altitude = diag * 1e-6;

        let nf = indices.len() / 3;
        let mut faces = Vec::with_capacity(nf);
        let mut normals = Vec::with_capacity(nf);
        let mut areas = Vec::with_capacity(nf);
        let mut excluded = Vec::with_capacity(nf);
        for (fi, t) in indices.chunks_exact(3).enumerate() {
            let f = [
                welded[t[0] as usize],
                welded[t[1] as usize],
                welded[t[2] as usize],
            ];
            let [a, b, c] = f.map(|v| wpos[v as usize]);
            let n = cross3(sub3(b, a), sub3(c, a));
            let len = len3(n);
            let area = 0.5 * len;
            let longest = len3(sub3(b, a)).max(len3(sub3(c, b))).max(len3(sub3(a, c)));
            let degen = f[0] == f[1]
                || f[1] == f[2]
                || f[0] == f[2]
                || !(len > 0.0)
                || !(len / longest > min_altitude);
            let skip = degen || mesh.ignored_faces.is_some_and(|ig| ig[fi]);
            faces.push(f);
            normals.push(if skip { [0.0; 3] } else { scale3(n, 1.0 / len) });
            areas.push(if skip { 0.0 } else { area });
            excluded.push(skip);
        }

        // Edge -> incident (face, local edge) list.
        let mut edges: HashMap<(u32, u32), Vec<(u32, u8)>> = HashMap::with_capacity(nf * 2);
        for (fi, f) in faces.iter().enumerate() {
            if excluded[fi] {
                continue;
            }
            for e in 0..3u8 {
                let a = f[e as usize];
                let b = f[(e as usize + 1) % 3];
                edges
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push((fi as u32, e));
            }
        }
        let material = |f: u32| mesh.face_materials.map_or(0, |m| m[f as usize]);
        // Input (unwelded) vertex at corner `c` of face `f`.
        let corner = |f: u32, c: usize| indices[f as usize * 3 + c % 3];
        // Whether two input vertices at the same position are split: by their
        // UVs if given (UV seams), else by being different vertices.
        let differ = |a: u32, b: u32| match mesh.existing_uvs {
            Some(uv) => uv[a as usize] != uv[b as usize],
            None => a != b,
        };
        let mut neighbors = vec![[NONE; 3]; nf];
        // Adjacent pairs across split edges: cut afterwards if they're rare
        // enough to be authored seams rather than flat shading.
        let (mut split, mut joined) = (Vec::new(), 0usize);
        for incident in edges.values() {
            if let [(f1, e1), (f2, e2)] = incident[..] {
                if material(f1) != material(f2) {
                    continue;
                }
                let d = |f: u32, e: u8| {
                    let face = faces[f as usize];
                    (face[e as usize], face[(e as usize + 1) % 3])
                };
                let (a1, b1) = d(f1, e1);
                let (a2, b2) = d(f2, e2);
                // Consistent orientation: the two faces traverse the edge in opposite directions.
                if a1 == b2 && b1 == a2 {
                    neighbors[f1 as usize][e1 as usize] = f2;
                    neighbors[f2 as usize][e2 as usize] = f1;
                    joined += 1;
                    let (e1, e2) = (e1 as usize, e2 as usize);
                    if differ(corner(f1, e1), corner(f2, e2 + 1))
                        || differ(corner(f1, e1 + 1), corner(f2, e2))
                    {
                        split.push([(f1, e1), (f2, e2)]);
                    }
                }
            }
        }
        if mesh.existing_uvs.is_some() || split.len() * 2 <= joined {
            for (f, e) in split.into_iter().flatten() {
                neighbors[f as usize][e] = NONE;
            }
        }

        Topology {
            welded,
            positions: wpos,
            faces,
            normals,
            areas,
            excluded,
            neighbors,
        }
    }
}
