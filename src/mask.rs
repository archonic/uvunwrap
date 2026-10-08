//! Chart masks and the bitmap page used for shape-aware packing.
//!
//! A chart is rasterized conservatively (every texel its triangles touch) at
//! the packing density, dilated by the padding, and stored as per-row spans.
//! Packing then searches a page bitset for the lowest position where no span
//! hits an occupied texel, so charts can nest into each other's concavities
//! and holes instead of reserving their whole bounding rectangle.

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

/// Half-open texel spans per row.
#[derive(Clone, Debug, Default)]
pub(crate) struct Mask {
    pub w: u32,
    pub h: u32,
    /// Spans of row `r` are `spans[row_start[r]..row_start[r + 1]]`.
    row_start: Vec<u32>,
    spans: Vec<[u32; 2]>,
    /// Row with the longest span, and that span's length: a cheap first test.
    long_row: u32,
    long_start: u32,
    long_len: u32,
    /// Longest span of each row.
    row_long: Vec<u32>,
    /// Rows in coarse-to-fine order (ends, middle, quarters, ...): a collision
    /// is usually found within the first few.
    order: Vec<u32>,
    /// Number of covered texels.
    pub area: u64,
}

impl Mask {
    fn row(&self, r: u32) -> &[[u32; 2]] {
        &self.spans[self.row_start[r as usize] as usize..self.row_start[r as usize + 1] as usize]
    }

    fn from_bits(bits: &Bits) -> Mask {
        let mut m = Mask { w: bits.w, h: bits.h, row_start: Vec::with_capacity(bits.h as usize + 1), ..Mask::default() };
        m.row_start.push(0);
        for r in 0..bits.h {
            let mut x = 0;
            let mut long = 0;
            while x < bits.w {
                let a = bits.next(r, x, true);
                if a >= bits.w {
                    break;
                }
                let b = bits.next(r, a, false);
                m.spans.push([a, b]);
                m.area += (b - a) as u64;
                long = long.max(b - a);
                if b - a > m.long_len {
                    m.long_len = b - a;
                    m.long_start = a;
                    m.long_row = r;
                }
                x = b;
            }
            m.row_long.push(long);
            m.row_start.push(m.spans.len() as u32);
        }
        m.order = coarse_to_fine(m.h);
        m
    }

    /// A fully covered `w × h` rectangle.
    pub(crate) fn rect(w: u32, h: u32) -> Mask {
        let mut bits = Bits::new(w, h);
        for y in 0..h {
            bits.fill(y, 0, w);
        }
        Mask::from_bits(&bits)
    }

    /// The mask in units of `b × b` texel blocks: a block is covered if any
    /// of its texels is.
    pub(crate) fn downsample(&self, b: u32) -> Mask {
        let up = |x: u32| x / b + u32::from(x % b != 0);
        let mut bits = Bits::new(up(self.w), up(self.h));
        for r in 0..self.h {
            for &[a, e] in self.row(r) {
                bits.fill(r / b, a / b, up(e));
            }
        }
        Mask::from_bits(&bits)
    }

    /// The mask rotated by 90° counter-clockwise (texel `(c, r)` moves to
    /// `(h - 1 - r, c)`), matching [`rotate_texel`] with rotation 1.
    fn rotate90(&self) -> Mask {
        let mut bits = Bits::new(self.h, self.w);
        for r in 0..self.h {
            for &[a, b] in self.row(r) {
                for c in a..b {
                    bits.set(self.h - 1 - r, c);
                }
            }
        }
        Mask::from_bits(&bits)
    }

    /// The mask rotated by 180°.
    fn rotate180(&self) -> Mask {
        let mut m = Mask { w: self.w, h: self.h, area: self.area, ..Mask::default() };
        m.row_start.push(0);
        for r in (0..self.h).rev() {
            for &[a, b] in self.row(r).iter().rev() {
                m.spans.push([self.w - b, self.w - a]);
            }
            m.row_start.push(m.spans.len() as u32);
        }
        m.long_row = self.h - 1 - self.long_row;
        m.long_len = self.long_len;
        m.long_start = self.w - self.long_start - self.long_len;
        m.row_long = self.row_long.iter().rev().copied().collect();
        m.order = coarse_to_fine(m.h);
        m
    }
}

/// `0..h` ordered by breadth-first bisection: `0, h-1, h/2, h/4, 3h/4, ...`.
fn coarse_to_fine(h: u32) -> Vec<u32> {
    let mut order = Vec::with_capacity(h as usize);
    if h == 0 {
        return order;
    }
    order.push(0);
    if h > 1 {
        order.push(h - 1);
    }
    let mut queue = std::collections::VecDeque::from([(0u32, h - 1)]);
    while let Some((a, b)) = queue.pop_front() {
        if b - a < 2 {
            continue;
        }
        let mid = a + (b - a) / 2;
        order.push(mid);
        queue.push_back((a, mid));
        queue.push_back((mid, b));
    }
    order
}

/// Dense bitmap, one `u64` word run per row.
struct Bits {
    w: u32,
    h: u32,
    words: usize,
    data: Vec<u64>,
}

impl Bits {
    fn new(w: u32, h: u32) -> Bits {
        let words = (w as usize + 63) / 64;
        Bits { w, h, words, data: vec![0; words * h as usize] }
    }

    #[inline]
    fn set(&mut self, x: u32, y: u32) {
        self.data[y as usize * self.words + (x / 64) as usize] |= 1 << (x % 64);
    }

    /// Sets `[a, b)` in row `y`.
    fn fill(&mut self, y: u32, a: u32, b: u32) {
        fill_row(&mut self.data[y as usize * self.words..(y as usize + 1) * self.words], a, b);
    }

    /// First `x' >= x` whose bit equals `value`, or `w`.
    fn next(&self, y: u32, x: u32, value: bool) -> u32 {
        next_in_row(&self.data[y as usize * self.words..(y as usize + 1) * self.words], x, value).min(self.w)
    }
}

fn fill_row(row: &mut [u64], a: u32, b: u32) {
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
fn next_in_row(row: &[u64], x: u32, value: bool) -> u32 {
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

/// All four rotations of a chart's padded mask, indexed by quarter turns.
pub(crate) struct ChartMasks {
    pub rot: Vec<Mask>,
}

/// Rasterizes a chart (UVs in world units, scaled by `s`) conservatively into
/// its padded mask and derives the requested rotations from it.
pub(crate) fn chart_masks(uvs: &[V2], tris: &[[u32; 3]], size: V2, s: f64, padding: u32, rotations: u8) -> ChartMasks {
    let (cw, ch) = content_size(size, s);
    let (w, h) = (cw + 2 * padding, ch + 2 * padding);
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
            let Some((xmin, xmax)) = slab_extent(&p, r as f64 - eps, r as f64 + 1.0 + eps) else { continue };
            let c0 = ((xmin - eps).ceil() - 1.0).max(0.0) as u32;
            let c1 = ((xmax + eps).floor().max(0.0) as u32).min(w - 1);
            if c0 <= c1 {
                bits.fill(r, c0, c1 + 1);
            }
        }
    }
    let bits = dilate(&bits, padding);
    let m0 = Mask::from_bits(&bits);
    let rot = match rotations {
        1 => vec![m0],
        2 => {
            let m1 = m0.rotate90();
            vec![m0, m1]
        }
        _ => {
            let m1 = m0.rotate90();
            let (m2, m3) = (m0.rotate180(), m1.rotate180());
            vec![m0, m1, m2, m3]
        }
    };
    ChartMasks { rot }
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
        return Bits { w: src.w, h: src.h, words: src.words, data: src.data.clone() };
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
        let (y0, y1) = (y.saturating_sub(r as usize), (y + r as usize).min(src.h as usize - 1));
        for yy in y0..=y1 {
            for k in 0..n {
                out.data[y * n + k] |= horiz.data[yy * n + k];
            }
        }
    }
    out
}

/// An atlas page as a bitset of occupied texels. Rows past `data` are free,
/// which lets a page grow downwards when its height is unbounded.
pub(crate) struct Page {
    w: u32,
    max_h: u32,
    words: usize,
    data: Vec<u64>,
    /// Longest free run per allocated row.
    free_run: Vec<u32>,
    /// Free runs `[start, end)` per allocated row.
    runs: Vec<Vec<[u32; 2]>>,
    pub used_width: u32,
    pub used_height: u32,
}

impl Page {
    pub fn new(w: u32, max_h: u32) -> Page {
        Page { w, max_h, words: (w as usize + 63) / 64, data: Vec::new(), free_run: Vec::new(), runs: Vec::new(), used_width: 0, used_height: 0 }
    }

    fn rows(&self) -> u32 {
        self.free_run.len() as u32
    }

    #[inline]
    fn row(&self, y: u32) -> &[u64] {
        &self.data[y as usize * self.words..(y as usize + 1) * self.words]
    }

    /// First `x' >= x` in row `y` whose occupancy equals `occupied`, or `w`.
    #[inline]
    fn next(&self, y: u32, x: u32, occupied: bool) -> u32 {
        if y >= self.rows() {
            return if occupied { self.w } else { x };
        }
        next_in_row(self.row(y), x, occupied).min(self.w)
    }

    /// Tests the mask at `(x, y)`, starting with mask row `*hint` (the row
    /// that collided last time, which likely collides again). On a collision
    /// returns the next `x` worth trying: the colliding span can't fit anywhere
    /// before it.
    fn test(&self, m: &Mask, x: u32, y: u32, hint: &mut u32) -> Result<(), u32> {
        let check = |r: u32| -> Result<(), u32> {
            let row = y + r;
            if row >= self.rows() {
                return Ok(());
            }
            for &[a, b] in m.row(r) {
                let hit = self.next(row, x + a, true);
                if hit < x + b {
                    return Err(self.next(row, hit, false).saturating_sub(a).max(x + 1));
                }
            }
            Ok(())
        };
        let first = *hint;
        check(first)?;
        for &r in &m.order {
            if r != first {
                if let Err(next) = check(r) {
                    *hint = r;
                    return Err(next);
                }
            }
        }
        Ok(())
    }

    /// Lowest (then leftmost) position for the mask with `y + h < limit`.
    pub fn find(&self, m: &Mask, limit: u64) -> Option<(u32, u32)> {
        if m.w > self.w || m.h > self.max_h {
            return None;
        }
        let y_end = (self.max_h - m.h) as u64;
        let mut hint = m.long_row;
        let mut reject = m.long_row as usize;
        let whole = [[0, self.w]];
        let mut y = 0u32;
        while (y as u64) <= y_end && y as u64 + (m.h as u64) < limit {
            if y >= self.rows() {
                return Some((0, y)); // everything from here down is free
            }
            // Cheap rejection: some row's longest free run is too short. The
            // row that failed last time is checked first; it usually fails again.
            let rows = (self.rows() - y).min(m.h) as usize;
            let free = &self.free_run[y as usize..y as usize + rows];
            let too_short = |r: usize| r < rows && free[r] < m.row_long[r];
            if too_short(reject) {
                y += 1;
                continue;
            }
            if let Some(r) = (0..rows).find(|&r| too_short(r)) {
                reject = r;
                y += 1;
                continue;
            }
            // Only positions where the mask's longest span lies inside a free
            // run of its page row are candidates.
            let lr = y + m.long_row;
            let runs: &[[u32; 2]] = if lr < self.rows() { &self.runs[lr as usize] } else { &whole };
            let (a, len) = (m.long_start, m.long_len);
            for &[s, e] in runs {
                // x + a in [s, e - len], and x + w <= page width.
                if e - s < len || e - len < a {
                    continue;
                }
                let mut x = s.saturating_sub(a);
                let x_end = (e - len - a).min(self.w - m.w);
                while x <= x_end {
                    match self.test(m, x, y, &mut hint) {
                        Ok(()) => return Some((x, y)),
                        Err(next) => x = next,
                    }
                }
            }
            y += 1;
        }
        None
    }

    pub fn insert(&mut self, m: &Mask, x: u32, y: u32) {
        let need = (y + m.h) as usize;
        if self.free_run.len() < need {
            self.data.resize(need * self.words, 0);
            // Bits past the page width count as occupied.
            for row in self.free_run.len()..need {
                let last = (row + 1) * self.words - 1;
                let used = self.w as usize - (self.words - 1) * 64;
                if used < 64 {
                    self.data[last] = !0u64 << used;
                }
            }
            self.free_run.resize(need, self.w);
            self.runs.resize(need, vec![[0, self.w]]);
        }
        for r in 0..m.h {
            let row = y + r;
            let start = row as usize * self.words;
            for &[a, b] in m.row(r) {
                fill_row(&mut self.data[start..start + self.words], x + a, x + b);
            }
            self.update_runs(row);
        }
        self.used_width = self.used_width.max(x + m.w);
        self.used_height = self.used_height.max(y + m.h);
    }

    fn update_runs(&mut self, y: u32) {
        let mut runs = std::mem::take(&mut self.runs[y as usize]);
        runs.clear();
        let mut x = 0;
        while x < self.w {
            let a = self.next(y, x, false);
            if a >= self.w {
                break;
            }
            let b = self.next(y, a, true);
            runs.push([a, b]);
            x = b;
        }
        self.free_run[y as usize] = runs.iter().map(|r| r[1] - r[0]).max().unwrap_or(0);
        self.runs[y as usize] = runs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_match_transform() {
        // An L-shaped chart: rasterize at each rotation directly and compare
        // with the derived masks.
        let uvs = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        let tris = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]];
        let s = 3.7;
        let derived = chart_masks(&uvs, &tris, [3.0, 2.0], s, 1, 4);
        let (cw, ch) = content_size([3.0, 2.0], s);
        for k in 0..4u8 {
            let moved: Vec<V2> = uvs.iter().map(|&q| rotate_texel([q[0] * s, q[1] * s], k, cw, ch)).collect();
            let size = if k % 2 == 0 { [cw as f64, ch as f64] } else { [ch as f64, cw as f64] };
            let direct = &chart_masks(&moved, &tris, size, 1.0, 1, 1).rot[0];
            let d = &derived.rot[k as usize];
            assert_eq!((d.w, d.h), (direct.w, direct.h), "rotation {k}");
            for r in 0..d.h {
                assert_eq!(d.row(r), direct.row(r), "rotation {k}, row {r}");
            }
        }
    }

    #[test]
    fn coarse_to_fine_is_a_permutation() {
        for h in 0..40 {
            let mut o = coarse_to_fine(h);
            assert_eq!(o[..2.min(h as usize)], [0, h.saturating_sub(1)][..2.min(h as usize)]);
            o.sort();
            assert_eq!(o, (0..h).collect::<Vec<_>>(), "h = {h}");
        }
    }

    #[test]
    fn nested_masks_dont_collide() {
        let ring_outer = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let mut uvs = ring_outer.to_vec();
        uvs.extend([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]]);
        // A square frame around a 4x4 hole.
        let tris = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]];
        let frame = chart_masks(&uvs, &tris, [10.0, 10.0], 1.0, 0, 1);
        let small = chart_masks(&[[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]], &[[0, 1, 2]], [1.5, 1.5], 1.0, 0, 1);
        let mut page = Page::new(10, 10);
        let (x, y) = page.find(&frame.rot[0], u64::MAX).unwrap();
        page.insert(&frame.rot[0], x, y);
        // The small chart must land inside the hole.
        let (x, y) = page.find(&small.rot[0], u64::MAX).unwrap();
        assert!((3..=5).contains(&x) && (3..=5).contains(&y), "placed at {x},{y}");
    }
}
