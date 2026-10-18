//! Conservative chart rasterization shared by both packers: a chart's
//! padded bitmap ([`rasterize_chart`]), which the shape packer stores as span
//! masks ([`crate::mask`]) and the fast packer reduces to profiles
//! ([`crate::fold`]), plus the texel geometry used to place charts.

use crate::geom::V2;

/// Texel extent of a chart's content box at density `s`. Chart UVs (in world
/// units, minimum at the origin) map to `[0, w] × [0, h]` texels.
pub(crate) fn content_size(size: V2, s: f64) -> (u32, u32) {
    let side = |v: f64| (v * s).ceil().clamp(1.0, (u32::MAX / 4) as f64) as u32;
    (side(size[0]), side(size[1]))
}

/// Maps a chart point already scaled to texels (`p = uv * s`) into its content
/// box rotated by `rotation` quarter turns counter-clockwise, given the
/// unrotated content size `(cw, ch)`. All four are proper rotations, so
/// triangle orientation is preserved.
#[inline]
pub(crate) fn rotate_texel(p: V2, rotation: u8, cw: u32, ch: u32) -> V2 {
    let (w, h) = (cw as f64, ch as f64);
    match rotation & 3 {
        0 => p,
        1 => [h - p[1], p[0]],
        2 => [w - p[0], h - p[1]],
        _ => [p[1], w - p[0]],
    }
}

/// Dense bitmap, one `u64` word run per row.
pub(crate) struct Bits {
    pub w: u32,
    pub h: u32,
    words: usize,
    data: Vec<u64>,
}

impl Bits {
    pub(crate) fn new(w: u32, h: u32) -> Bits {
        let words = (w as usize).div_ceil(64);
        Bits {
            w,
            h,
            words,
            data: vec![0; words * h as usize],
        }
    }

    #[inline]
    pub(crate) fn set(&mut self, x: u32, y: u32) {
        self.data[y as usize * self.words + (x / 64) as usize] |= 1 << (x % 64);
    }

    /// Sets `[a, b)` in row `y`.
    pub(crate) fn fill(&mut self, y: u32, a: u32, b: u32) {
        fill_row(
            &mut self.data[y as usize * self.words..(y as usize + 1) * self.words],
            a,
            b,
        );
    }

    /// The words of row `y`; bits past `w` are clear.
    pub(crate) fn row(&self, y: u32) -> &[u64] {
        &self.data[y as usize * self.words..(y as usize + 1) * self.words]
    }

    /// First `x' >= x` whose bit equals `value`, or `w`.
    pub(crate) fn next(&self, y: u32, x: u32, value: bool) -> u32 {
        next_in_row(self.row(y), x, value).min(self.w)
    }
}

/// Sets bits `[a, b)` of a bitmap row.
pub(crate) fn fill_row(row: &mut [u64], a: u32, b: u32) {
    if a >= b {
        return;
    }
    let (wa, wb) = ((a / 64) as usize, ((b - 1) / 64) as usize);
    let lo = !0u64 << (a % 64);
    let hi = !0u64 >> (63 - (b - 1) % 64);
    if wa == wb {
        row[wa] |= lo & hi;
    } else {
        row[wa] |= lo;
        row[wa + 1..wb].iter_mut().for_each(|w| *w = !0);
        row[wb] |= hi;
    }
}

/// First index `>= x` whose bit equals `value`; `row.len() * 64` if none.
#[inline]
pub(crate) fn next_in_row(row: &[u64], x: u32, value: bool) -> u32 {
    let mut i = (x / 64) as usize;
    if i >= row.len() {
        return x;
    }
    let flip = if value { 0 } else { !0 };
    let mut word = (row[i] ^ flip) & (!0u64 << (x % 64));
    loop {
        if word != 0 {
            return i as u32 * 64 + word.trailing_zeros();
        }
        i += 1;
        if i == row.len() {
            return i as u32 * 64;
        }
        word = row[i] ^ flip;
    }
}

/// A chart (UVs in world units, minimum at the origin) rasterized
/// conservatively at density `s`, every texel its triangles touch, then
/// dilated by `padding`: its padded bitmap, `padding` texels larger than its
/// content box on every side.
pub(crate) fn rasterize_chart(
    uvs: &[V2],
    tris: &[[u32; 3]],
    size: V2,
    s: f64,
    padding: u32,
) -> Bits {
    let (cw, ch) = content_size(size, s);
    let (w, h) = (
        cw.saturating_add(padding.saturating_mul(2)),
        ch.saturating_add(padding.saturating_mul(2)),
    );
    let mut bits = Bits::new(w, h);
    let off = padding as f64;
    // Grow each triangle slightly so f32 rounding of the final UVs can never
    // move a triangle outside its mask.
    let eps = 1e-3;
    for t in tris {
        let p = t.map(|i| {
            let q = uvs[i as usize];
            [q[0] * s + off, q[1] * s + off]
        });
        let ymin = p[0][1].min(p[1][1]).min(p[2][1]) - eps;
        let ymax = p[0][1].max(p[1][1]).max(p[2][1]) + eps;
        let r0 = ((ymin.ceil() - 1.0).max(0.0) as u32).min(h - 1);
        let r1 = (ymax.floor().max(0.0) as u32).min(h - 1);
        for r in r0..=r1 {
            // X extent of the triangle within the closed row slab [r, r + 1].
            let Some((xmin, xmax)) = slab_extent(&p, r as f64 - eps, r as f64 + 1.0 + eps) else {
                continue;
            };
            let c0 = ((xmin - eps).ceil() - 1.0).max(0.0) as u32;
            let c1 = ((xmax + eps).floor().max(0.0) as u32).min(w - 1);
            if c0 <= c1 {
                bits.fill(r, c0, c1 + 1);
            }
        }
    }
    dilate(&bits, padding)
}

/// X range of a triangle clipped to the horizontal slab `y0 <= y <= y1`.
fn slab_extent(p: &[V2; 3], y0: f64, y1: f64) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut add = |x: f64| {
        lo = lo.min(x);
        hi = hi.max(x);
    };
    for i in 0..3 {
        let (a, b) = (p[i], p[(i + 1) % 3]);
        if a[1] >= y0 && a[1] <= y1 {
            add(a[0]);
        }
        // Crossings of edge a-b with the slab boundaries.
        for y in [y0, y1] {
            if (a[1] - y) * (b[1] - y) < 0.0 {
                let t = (y - a[1]) / (b[1] - a[1]);
                add(a[0] + t * (b[0] - a[0]));
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Square (Chebyshev) dilation by `r` texels.
fn dilate(src: &Bits, r: u32) -> Bits {
    if r == 0 {
        return Bits {
            w: src.w,
            h: src.h,
            words: src.words,
            data: src.data.clone(),
        };
    }
    let mut horiz = Bits::new(src.w, src.h);
    for y in 0..src.h {
        let mut x = 0;
        while x < src.w {
            let a = src.next(y, x, true);
            if a >= src.w {
                break;
            }
            let b = src.next(y, a, false);
            horiz.fill(y, a.saturating_sub(r), (b + r).min(src.w));
            x = b;
        }
    }
    let mut out = Bits::new(src.w, src.h);
    let n = src.words;
    for y in 0..src.h as usize {
        let (y0, y1) = (
            y.saturating_sub(r as usize),
            (y + r as usize).min(src.h as usize - 1),
        );
        for yy in y0..=y1 {
            for k in 0..n {
                out.data[y * n + k] |= horiz.data[yy * n + k];
            }
        }
    }
    out
}
