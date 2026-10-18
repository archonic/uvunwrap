//! The built-in torus and the output hash, shared by the `unwrap` and
//! `torus_hash` examples (CI compares the hash across platforms).

/// Cosine and sine of `k / n` turns from Taylor series, so the torus (and the
/// output hash CI compares) doesn't depend on the platform's libm.
fn cos_sin(k: u32, n: u32) -> (f64, f64) {
    use std::f64::consts::{PI, TAU};
    let mut x = k as f64 / n as f64 * TAU;
    if x > PI {
        x -= TAU;
    }
    let (mut c, mut s) = (0.0, 0.0);
    for m in (1..=30).rev() {
        c = 1.0 - x * x / ((2 * m - 1) * (2 * m)) as f64 * c;
        s = 1.0 - x * x / ((2 * m) * (2 * m + 1)) as f64 * s;
    }
    (c, x * s)
}

/// A 64×32 torus (4096 triangles).
pub fn torus() -> (Vec<[f32; 3]>, Vec<u32>) {
    let (nu, nv) = (64u32, 32u32);
    let mut positions = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let ((ca, sa), (cb, sb)) = (cos_sin(i, nu), cos_sin(j, nv));
            let r = 1.0 + 0.4 * cb;
            positions.push([(r * ca) as f32, (r * sa) as f32, (0.4 * sb) as f32]);
        }
    }
    let id = |i: u32, j: u32| (j % nv) * nu + (i % nu);
    let mut indices = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            indices.extend([
                id(i, j),
                id(i + 1, j),
                id(i + 1, j + 1),
                id(i, j),
                id(i + 1, j + 1),
                id(i, j + 1),
            ]);
        }
    }
    (positions, indices)
}

/// FNV-1a over the first mesh's whole output: equal on every platform and
/// thread count.
pub fn hash(atlas: &uvunwrap::Atlas) -> u64 {
    let out = &atlas.meshes[0];
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |x: u32| {
        for b in x.to_le_bytes() {
            hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
        }
    };
    for v in &out.vertices {
        [
            v.xref,
            v.uv[0].to_bits(),
            v.uv[1].to_bits(),
            v.chart.unwrap_or(u32::MAX),
            v.page.unwrap_or(0),
        ]
        .into_iter()
        .for_each(&mut feed);
    }
    out.indices.iter().for_each(|&i| feed(i));
    [atlas.width, atlas.height, atlas.texels_per_unit.to_bits()]
        .into_iter()
        .for_each(&mut feed);
    hash
}
