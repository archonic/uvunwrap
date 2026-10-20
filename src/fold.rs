//! Fold-and-push packing, the fast [`Packing::Fast`] method, after TABI (Gu,
//! Vining & Sheffer, "TABI: Tight and Balanced Interactive Atlas Packing",
//! 2026).
//!
//! Items sorted by height are folded into rows, and each row is pushed up
//! against a frontline: per column, one past the lowest occupied texel. Items
//! are described by [`Profile`]s read exactly off their padded masks (TABI
//! approximates them with local AABBs and OBBs because it never rasterizes):
//! columns' top and bottom ends shape the vertical push, rows' left and right
//! ends the horizontal compacting within a row.
//!
//! Each row tries both directions, the full page width or the concavity next
//! to a tall item followed by much shorter ones (a *knee*), and with or
//! without horizontal compacting, and keeps the lowest result (TABI Sec. 4.2,
//! Alg. 4). Items are never mirrored, which would flip UV winding: TABI's
//! reflections become a choice between an orientation and its 180° turn.
//!
//! A frontline can't reach the space under an overhang, so (beyond TABI)
//! each placed row leaves a [`Pocket`]: per column, the gap between the
//! frontline it was pushed against and its items' tops. The largest of the
//! remaining items that fit go into it, each pushed up as far as it goes;
//! most are small ones from the end of the order.
//!
//! [`Packing::Fast`]: crate::Packing::Fast

use crate::mask::coarse_to_fine;
use crate::pack::{Layout, Placement};
use crate::raster::Bits;
use std::cmp::Reverse;

/// Per-column and per-row extents of a padded mask. Conservative: items
/// whose profiles don't intersect don't overlap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Profile {
    pub w: u32,
    pub h: u32,
    /// Per column, the first covered row (`h` if none).
    top: Vec<u32>,
    /// Per column, one past the last covered row (0 if none).
    bot: Vec<u32>,
    /// Per row, the first covered column (`w` if none).
    left: Vec<u32>,
    /// Per row, one past the last covered column (0 if none).
    right: Vec<u32>,
    /// Quarter turns counter-clockwise from the mask it was read from.
    pub rotation: u8,
}

impl Profile {
    /// The profile of a chart's padded bitmap
    /// ([`crate::raster::rasterize_chart`]), read a word at a time.
    pub(crate) fn from_bits(bits: &Bits) -> Profile {
        let (w, h) = (bits.w, bits.h);
        let mut left = vec![w; h as usize];
        let mut right = vec![0; h as usize];
        let mut top = vec![h; w as usize];
        let mut bot = vec![0; w as usize];
        for y in 0..h {
            let row = bits.row(y);
            if let Some(k) = row.iter().position(|&x| x != 0) {
                left[y as usize] = k as u32 * 64 + row[k].trailing_zeros();
                let k = row.iter().rposition(|&x| x != 0).expect("a word is set");
                right[y as usize] = k as u32 * 64 + 64 - row[k].leading_zeros();
            }
        }
        // A column's top (bottom) is the first row, from above (below), where
        // its bit is set and wasn't in any row before.
        let first_rows = |rows: &mut dyn Iterator<Item = u32>, out: &mut [u32], end: bool| {
            let mut seen = vec![0u64; (w as usize).div_ceil(64)];
            for y in rows {
                for (k, (s, &word)) in seen.iter_mut().zip(bits.row(y)).enumerate() {
                    let mut new = word & !*s;
                    *s |= new;
                    while new != 0 {
                        out[k * 64 + new.trailing_zeros() as usize] = y + end as u32;
                        new &= new - 1;
                    }
                }
            }
        };
        first_rows(&mut (0..h), &mut top, false);
        first_rows(&mut (0..h).rev(), &mut bot, true);
        Profile {
            w,
            h,
            top,
            bot,
            left,
            right,
            rotation: 0,
        }
    }

    /// A fully covered `w × h` rectangle.
    pub(crate) fn rect(w: u32, h: u32) -> Profile {
        Profile {
            w,
            h,
            top: vec![0; w as usize],
            bot: vec![h; w as usize],
            left: vec![0; h as usize],
            right: vec![w; h as usize],
            rotation: 0,
        }
    }

    /// The profile in units of `b × b` texel blocks, as read off the
    /// downsampled mask ([`Mask::downsample`](crate::mask::Mask::downsample)).
    pub(crate) fn downsample(&self, b: u32) -> Profile {
        if b == 1 {
            return self.clone();
        }
        // Over each block's texel columns (rows): the first start, and the
        // last end rounded up to a whole block.
        let shrink = |starts: &[u32], ends: &[u32], empty: u32| -> (Vec<u32>, Vec<u32>) {
            starts
                .chunks(b as usize)
                .zip(ends.chunks(b as usize))
                .map(|(s, e)| {
                    let end = e.iter().copied().max().unwrap_or(0);
                    if end == 0 {
                        (empty, 0)
                    } else {
                        (s.iter().copied().min().unwrap_or(0) / b, end.div_ceil(b))
                    }
                })
                .unzip()
        };
        let (w, h) = (self.w.div_ceil(b), self.h.div_ceil(b));
        let (top, bot) = shrink(&self.top, &self.bot, h);
        let (left, right) = shrink(&self.left, &self.right, w);
        Profile {
            w,
            h,
            top,
            bot,
            left,
            right,
            rotation: self.rotation,
        }
    }

    /// The profile of the mask turned 90° counter-clockwise (texel `(c, r)`
    /// to `(h - 1 - r, c)`, as [`crate::raster::rotate_texel`]).
    fn rotate90(&self) -> Profile {
        let h = self.h;
        Profile {
            w: self.h,
            h: self.w,
            top: self.left.iter().rev().copied().collect(),
            bot: self.right.iter().rev().copied().collect(),
            left: self.bot.iter().map(|&b| h - b).collect(),
            right: self.top.iter().map(|&t| h - t).collect(),
            rotation: (self.rotation + 1) & 3,
        }
    }

    /// Turned taller than wide, then by 180° if that leaves more empty space
    /// at the bottom and right, where the next row and the next item in the
    /// row can use it (TABI's chart orientation, without reflections).
    pub(crate) fn oriented(self, rotatable: bool) -> Profile {
        if !rotatable {
            return self;
        }
        let p = if self.w > self.h {
            self.rotate90()
        } else {
            self
        };
        let turned = p.rotate90().rotate90();
        if turned.free_below_right() > p.free_below_right() {
            turned
        } else {
            p
        }
    }

    /// Empty area between the profile and the bounding box's bottom and
    /// right edges.
    fn free_below_right(&self) -> u64 {
        let below: u64 = self.bot.iter().map(|&b| (self.h - b) as u64).sum();
        let right: u64 = self.right.iter().map(|&r| (self.w - r) as u64).sum();
        below + right
    }

    /// Whether this profile at `(x, y)` intersects `o` at `(ox, oy)`.
    fn intersects(&self, x: u32, y: u32, o: &Profile, ox: u32, oy: u32) -> bool {
        if x >= ox + o.w || ox >= x + self.w {
            return false;
        }
        let (y0, y1) = (y.max(oy), (y + self.h).min(oy + o.h));
        (y0..y1).any(|row| {
            let (r, s) = ((row - y) as usize, (row - oy) as usize);
            let (a0, a1) = (x + self.left[r], x + self.right[r]);
            let (b0, b1) = (ox + o.left[s], ox + o.right[s]);
            a0 < a1 && b0 < b1 && a0 < b1 && b0 < a1
        })
    }
}

/// A tall item followed by much shorter ones in a row: the concavity below
/// the short ones can be filled by the next rows (TABI Sec. 4.2).
#[derive(Clone, Copy, Debug)]
struct Knee {
    /// The row ran left to right, so the concavity is right of the tall
    /// item (else left of it).
    ltr: bool,
    /// The tall item's columns `[left, right)`, narrowed by
    /// [`Knee::update`] as rows fill the concavity.
    left: u32,
    right: u32,
}

impl Knee {
    /// Moves the concavity's edge past the tall item's columns that no longer
    /// reach below the frontline next to it (TABI Alg. 2).
    fn update(&mut self, frontline: &[u32]) {
        let w = frontline.len() as u32;
        if self.ltr {
            if self.right >= w {
                return;
            }
            let floor = frontline[self.right as usize];
            self.right = (self.left..self.right)
                .rev()
                .find(|&x| frontline[x as usize] >= floor)
                .map_or(self.left, |x| x + 1);
        } else {
            if self.left == 0 {
                return;
            }
            let floor = frontline[self.left as usize - 1];
            self.left = (self.left..self.right)
                .find(|&x| frontline[x as usize] >= floor)
                .unwrap_or(self.right);
        }
    }

    /// Columns `[x0, x1)` of the concavity.
    fn region(&self, w: u32) -> (u32, u32) {
        if self.ltr {
            (self.right, w)
        } else {
            (0, self.left)
        }
    }
}

/// One way to place the next row.
struct Row {
    /// Items `order[start..end]` are in the row.
    end: usize,
    /// Top-left corner of each item's bounding box.
    pos: Vec<(u32, u32)>,
    ltr: bool,
    frontline: Vec<u32>,
    /// The frontline's deepest point (largest y) over the whole page, and
    /// over the columns the row was folded into.
    height: u32,
    region_height: u32,
    /// Sum of the frontline over the columns the row was folded into.
    region_sum: u64,
}

impl Row {
    /// Lower is better: the deepest point (balance), then the area above
    /// the frontline (tightness).
    fn rank(&self) -> (u32, u64) {
        (self.region_height, self.region_sum)
    }
}

/// One page being filled.
struct Shelf {
    frontline: Vec<u32>,
    knee: Option<Knee>,
    used_width: u32,
    /// One past the lowest row used (the frontline's deepest point, or
    /// deeper once items fill the space below it).
    used_height: u32,
    empty: bool,
}

/// Fold-and-push state over all pages.
struct Folder<'a> {
    items: &'a [Profile],
    /// Item indices, tallest first.
    order: Vec<usize>,
    width: u32,
    /// Smallest height difference that makes a knee.
    knee_min: u32,
    /// Per item, what pockets need to know.
    fits: Vec<FitInfo>,
    /// Page height (`u32::MAX`: unbounded).
    height: u32,
}

impl Folder<'_> {
    /// Fills `pocket` on `shelf` with the largest of `candidates` that fit,
    /// one at a time, placing them on `page`; returns the others.
    fn fill(
        &self,
        shelf: &mut Shelf,
        pocket: &mut Pocket,
        candidates: &[usize],
        page: u32,
        placements: &mut [Placement],
    ) -> Vec<usize> {
        let mut rest = Vec::with_capacity(candidates.len());
        for &i in candidates {
            let p = &self.items[i];
            match pocket.fit(p, &self.fits[i], self.height) {
                Some((x, y)) => {
                    pocket.insert(p, x, y);
                    placements[i] = Placement {
                        page,
                        x,
                        y,
                        rotation: p.rotation,
                    };
                    shelf.used_width = shelf.used_width.max(x + p.w);
                    shelf.used_height = shelf.used_height.max(y + p.h);
                    shelf.empty = false;
                }
                None => rest.push(i),
            }
        }
        rest
    }

    /// Folds items from `start` into one row within columns `[x0, x1)`,
    /// left to right or right to left, with each item touching the previous
    /// ones' bounding boxes or (`compact`) their row profiles. Returns the
    /// end of the row and the items' x positions.
    fn fold(&self, start: usize, x0: u32, x1: u32, ltr: bool, compact: bool) -> (usize, Vec<u32>) {
        let rows = self.items[self.order[start]].h as usize;
        // Per row, the row's occupied extent so far (compacting), or the
        // bounding boxes' edge.
        let mut edge = vec![if ltr { x0 } else { x1 } as i64; rows];
        let mut cursor = if ltr { x0 } else { x1 } as i64;
        let mut xs = Vec::new();
        for &i in &self.order[start..] {
            let p = &self.items[i];
            let filled = (0..p.h as usize).filter(|&r| p.left[r] < p.right[r]);
            let x = if !compact {
                if ltr {
                    cursor
                } else {
                    cursor - p.w as i64
                }
            } else if ltr {
                filled
                    .map(|r| edge[r] - p.left[r] as i64)
                    .max()
                    .unwrap_or(cursor)
            } else {
                filled
                    .map(|r| edge[r] - p.right[r] as i64)
                    .min()
                    .unwrap_or(cursor - p.w as i64)
            };
            // The bounding box must be on the page, the content within
            // the columns.
            let lo = p.left.iter().copied().min().unwrap_or(0);
            let hi = p.right.iter().copied().max().unwrap_or(p.w);
            if x < 0
                || x + p.w as i64 > self.width as i64
                || x + (lo as i64) < x0 as i64
                || x + hi as i64 > x1 as i64
            {
                break;
            }
            for ((e, &l), &r) in edge.iter_mut().zip(&p.left).zip(&p.right) {
                if l < r {
                    *e = if ltr {
                        (*e).max(x + r as i64)
                    } else {
                        (*e).min(x + l as i64)
                    };
                }
            }
            cursor = if ltr { x + p.w as i64 } else { x };
            xs.push(x as u32);
        }
        (start + xs.len(), xs)
    }

    /// Pushes a folded row up against `frontline` and raises the frontline
    /// under it.
    fn push(
        &self,
        start: usize,
        xs: Vec<u32>,
        ltr: bool,
        frontline: &[u32],
        x0: u32,
        x1: u32,
    ) -> Row {
        let items = &self.order[start..start + xs.len()];
        let mut ys: Vec<u32> = items
            .iter()
            .zip(&xs)
            .map(|(&i, &x)| {
                let p = &self.items[i];
                (0..p.w as usize)
                    .filter(|&c| p.top[c] < p.bot[c])
                    .map(|c| frontline[x as usize + c].saturating_sub(p.top[c]))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        // Items pushed independently can collide with their neighbours in the
        // row; lower the higher of each colliding pair to the other's level,
        // where the folding guarantees they are apart (TABI Alg. 1, made
        // exact for any pair).
        let mut by_x: Vec<usize> = (0..xs.len()).collect();
        by_x.sort_by_key(|&k| (xs[k], k));
        loop {
            let mut changed = false;
            for (n, &a) in by_x.iter().enumerate() {
                let pa = &self.items[items[a]];
                for &b in &by_x[n + 1..] {
                    if xs[b] >= xs[a] + pa.w {
                        break;
                    }
                    let pb = &self.items[items[b]];
                    if ys[a] != ys[b] && pa.intersects(xs[a], ys[a], pb, xs[b], ys[b]) {
                        let y = ys[a].max(ys[b]);
                        (ys[a], ys[b]) = (y, y);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut frontline = frontline.to_vec();
        for ((&i, &x), &y) in items.iter().zip(&xs).zip(&ys) {
            let p = &self.items[i];
            for c in 0..p.w as usize {
                if p.top[c] < p.bot[c] {
                    let f = &mut frontline[x as usize + c];
                    *f = (*f).max(y + p.bot[c]);
                }
            }
        }
        let height = frontline.iter().copied().max().unwrap_or(0);
        let region = &frontline[x0 as usize..x1 as usize];
        let region_height = region.iter().copied().max().unwrap_or(0);
        let region_sum = region.iter().map(|&f| f as u64).sum();
        Row {
            end: start + xs.len(),
            pos: xs.into_iter().zip(ys).collect(),
            ltr,
            frontline,
            height,
            region_height,
            region_sum,
        }
    }

    /// The best row within columns `[x0, x1)`: per direction, the fold that
    /// fits more items, compacted or not, or if they fit as many, the one
    /// that pushes lower; then the direction that pushes lower.
    fn best_in(&self, shelf: &Shelf, start: usize, x0: u32, x1: u32) -> Option<Row> {
        let push = |(end, xs): (usize, Vec<u32>), ltr| {
            (end > start).then(|| self.push(start, xs, ltr, &shelf.frontline, x0, x1))
        };
        let lower = |a: Option<Row>, b: Option<Row>| match (a, b) {
            (Some(a), Some(b)) => Some(if b.rank() < a.rank() { b } else { a }),
            (a, b) => a.or(b),
        };
        let mut best = None;
        for ltr in [true, false] {
            let loose = self.fold(start, x0, x1, ltr, false);
            let tight = self.fold(start, x0, x1, ltr, true);
            let row = match tight.0.cmp(&loose.0) {
                std::cmp::Ordering::Greater => push(tight, ltr),
                std::cmp::Ordering::Less => push(loose, ltr),
                std::cmp::Ordering::Equal => lower(push(tight, ltr), push(loose, ltr)),
            };
            best = lower(best, row);
        }
        best
    }

    /// The tallest-to-shorter step in a placed row, if it is a knee: at
    /// least `knee_min` texels and a fifth of the taller item.
    fn find_knee(&self, start: usize, row: &Row) -> Option<Knee> {
        let items = &self.order[start..row.end];
        (1..items.len())
            .filter_map(|k| {
                let (a, b) = (&self.items[items[k - 1]], &self.items[items[k]]);
                let d = a.h - b.h;
                (d >= self.knee_min && 5 * d >= a.h).then_some((d, k - 1))
            })
            .max_by_key(|&(d, k)| (d, Reverse(k)))
            .map(|(_, k)| {
                let x = row.pos[k].0;
                Knee {
                    ltr: row.ltr,
                    left: x,
                    right: x + self.items[items[k]].w,
                }
            })
    }

    /// The next row on `shelf`, folded at the page width or into the current
    /// knee, whichever leaves the page's frontline lower; and the knee for
    /// the rows after it.
    fn next_row(&self, shelf: &Shelf, start: usize) -> (Row, Option<Knee>) {
        let knee = shelf.knee.map(|mut k| {
            k.update(&shelf.frontline);
            k
        });
        let full = self
            .best_in(shelf, start, 0, self.width)
            .expect("every item fits the page width");
        let at_knee = knee.and_then(|k| {
            let (x0, x1) = k.region(self.width);
            (x0 < x1)
                .then(|| self.best_in(shelf, start, x0, x1))
                .flatten()
        });
        match at_knee {
            Some(row) if row.height < full.height => (row, knee),
            _ => {
                let knee = self.find_knee(start, &full);
                (full, knee)
            }
        }
    }
}

/// What [`Pocket::fit`] needs of an item besides its profile.
struct FitInfo {
    /// The height of its tallest column, and that column.
    tall: u32,
    key: usize,
    /// Its covered columns, ends and middle first: they usually decide (or
    /// rule out) a position within the first few.
    cols: Vec<u32>,
}

impl FitInfo {
    fn new(p: &Profile) -> FitInfo {
        let cols: Vec<u32> = coarse_to_fine(p.w)
            .into_iter()
            .filter(|&c| p.top[c as usize] < p.bot[c as usize])
            .collect();
        let (tall, key) = (0..p.w as usize)
            .map(|c| (p.bot[c].saturating_sub(p.top[c]), c))
            .max_by_key(|&(t, c)| (t, Reverse(c)))
            .unwrap_or((0, 0));
        FitInfo { tall, key, cols }
    }
}

/// Free space on a page: per column from `x0`, the rows `[lo, hi)`. Items
/// are pushed up against `lo` and must stay above `hi`.
///
/// Above a row, between it and the frontline it was pushed against (empty
/// where the row has no item): everything placed before lies above `lo` and
/// the row's items lie below `hi`, so an item within it overlaps nothing,
/// and later rows, pushed against the frontline below the row, never reach
/// it.
struct Pocket {
    x0: u32,
    lo: Vec<u32>,
    hi: Vec<u32>,
    /// The largest `hi - lo`.
    deepest: u32,
    /// Columns by decreasing initial depth, with that depth. Depths only
    /// shrink, so the columns that started at least as deep as an item's
    /// tallest column are the only ones its tallest column can go in.
    by_depth: Vec<(u32, u32)>,
}

impl Pocket {
    /// The pocket above a row of `items` at `pos`, pushed against
    /// `frontline`.
    fn new(profiles: &[Profile], items: &[usize], pos: &[(u32, u32)], frontline: &[u32]) -> Pocket {
        let x0 = pos.iter().map(|p| p.0).min().unwrap_or(0);
        let x1 = items
            .iter()
            .zip(pos)
            .map(|(&i, p)| p.0 + profiles[i].w)
            .max()
            .unwrap_or(x0);
        let lo = frontline[x0 as usize..x1 as usize].to_vec();
        let mut hi = vec![u32::MAX; lo.len()];
        for (&i, &(x, y)) in items.iter().zip(pos) {
            let p = &profiles[i];
            for c in 0..p.w as usize {
                if p.top[c] < p.bot[c] {
                    let h = &mut hi[(x - x0) as usize + c];
                    *h = (*h).min(y + p.top[c]);
                }
            }
        }
        for (h, &l) in hi.iter_mut().zip(&lo) {
            *h = if *h == u32::MAX { l } else { (*h).max(l) };
        }
        Pocket::between(x0, lo, hi)
    }

    fn between(x0: u32, lo: Vec<u32>, hi: Vec<u32>) -> Pocket {
        let mut by_depth: Vec<(u32, u32)> = lo
            .iter()
            .zip(&hi)
            .enumerate()
            .map(|(c, (&l, &h))| (h - l, c as u32))
            .filter(|&(d, _)| d > 0)
            .collect();
        by_depth.sort_unstable_by_key(|&(d, c)| (Reverse(d), c));
        let mut pocket = Pocket {
            x0,
            lo,
            hi,
            deepest: 0,
            by_depth,
        };
        pocket.measure();
        pocket
    }

    fn measure(&mut self) {
        self.deepest = self
            .lo
            .iter()
            .zip(&self.hi)
            .map(|(&l, &h)| h - l)
            .max()
            .unwrap_or(0);
    }

    /// The highest (then leftmost) position for `p`, whose tallest column is
    /// `tall`, with its bounding box above `height`.
    fn fit(&self, p: &Profile, fit: &FitInfo, height: u32) -> Option<(u32, u32)> {
        let (tall, key, cols) = (fit.tall, fit.key, &fit.cols[..]);
        if cols.is_empty() || tall > self.deepest || p.w as usize > self.lo.len() || p.h > height {
            return None;
        }
        let last = self.lo.len() - p.w as usize;
        let mut best: Option<(u32, u32)> = None;
        // The tallest column can only go in a column that is still as deep.
        let deep = self.by_depth.partition_point(|&(d, _)| d >= tall);
        'columns: for &(_, k) in &self.by_depth[..deep] {
            let Some(x) = (k as usize).checked_sub(key).filter(|&x| x <= last) else {
                continue;
            };
            let (lo, hi) = (self.lo[k as usize], self.hi[k as usize]);
            if hi - lo < tall {
                continue;
            }
            // Pushed up against `lo`, the item must stay above `hi` (in the
            // tallest column, first), the page bottom, and beat `best`.
            let at = self.x0 + x as u32;
            let Some(limit) = hi.checked_sub(p.bot[key]) else {
                continue;
            };
            let mut limit = limit.min(height - p.h);
            if let Some((bx, by)) = best {
                limit = limit.min(if at < bx { by } else { by.wrapping_sub(1) });
                if by == 0 && at >= bx {
                    continue;
                }
            }
            // It fits if the lowest `y` any column needs is at most the
            // highest any column allows.
            let mut y = lo.saturating_sub(p.top[key]);
            for &c in cols {
                let c = c as usize;
                y = y.max(self.lo[x + c].saturating_sub(p.top[c]));
                match self.hi[x + c].checked_sub(p.bot[c]) {
                    Some(l) if y <= l.min(limit) => limit = limit.min(l),
                    _ => continue 'columns,
                }
            }
            best = Some((at, y));
        }
        best
    }

    /// Marks the columns above `p` at `(x, y)` as used.
    fn insert(&mut self, p: &Profile, x: u32, y: u32) {
        let x = (x - self.x0) as usize;
        for c in 0..p.w as usize {
            if p.top[c] < p.bot[c] {
                let l = &mut self.lo[x + c];
                *l = (*l).max(y + p.bot[c]).min(self.hi[x + c]);
            }
        }
        self.measure();
    }
}

/// Packs profiles (in their own orientation) by folding them into rows on at
/// most `max_pages` pages of `width × height` (`height` may be `u32::MAX`:
/// unbounded). `None` if they don't fit.
pub(crate) fn pack_folded(
    items: &[Profile],
    width: u32,
    height: u32,
    max_pages: usize,
) -> Option<Layout> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| (Reverse(items[i].h), Reverse(items[i].w), i));
    let reference = if height == u32::MAX { width } else { height };
    let mut folder = Folder {
        items,
        order,
        width,
        knee_min: (reference / 10).max(1),
        fits: items.iter().map(FitInfo::new).collect(),
        height,
    };
    let new_shelf = || Shelf {
        frontline: vec![0; width as usize],
        knee: None,
        used_width: 0,
        used_height: 0,
        empty: true,
    };
    let mut placements = vec![Placement::default(); items.len()];
    let mut shelves: Vec<Shelf> = Vec::new();
    let mut start = 0;
    while start < folder.order.len() {
        if shelves.is_empty() {
            shelves.push(new_shelf());
        }
        let page = shelves.len() as u32 - 1;
        let last_page = shelves.len() >= max_pages;
        let shelf = shelves.last_mut().expect("pushed above");
        let (row, knee) = folder.next_row(shelf, start);
        if row.height > height {
            if shelf.empty || last_page {
                return None;
            }
            shelves.push(new_shelf());
            continue;
        }
        let row_items = &folder.order[start..row.end];
        for (&i, &(x, y)) in row_items.iter().zip(&row.pos) {
            placements[i] = Placement {
                page,
                x,
                y,
                rotation: items[i].rotation,
            };
            shelf.used_width = shelf.used_width.max(x + items[i].w);
        }
        // Fill the pocket the row left above itself with the largest of the
        // remaining items that fit, which then leave the folding order.
        let mut pocket = Pocket::new(items, row_items, &row.pos, &shelf.frontline);
        let rest = folder.fill(
            shelf,
            &mut pocket,
            &folder.order[row.end..],
            page,
            &mut placements,
        );
        shelf.frontline = row.frontline;
        shelf.knee = knee;
        shelf.used_height = shelf.used_height.max(row.height);
        shelf.empty = false;
        folder.order.truncate(row.end);
        folder.order.extend(rest);
        start = row.end;
    }
    Some(Layout {
        placements,
        page_sizes: shelves
            .iter()
            .map(|s| (s.used_width, s.used_height))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::V2;
    use crate::mask::{chart_masks, Mask};
    use crate::raster::rasterize_chart;

    /// The profile read off a mask, texel by texel.
    fn profile_of(m: &Mask) -> Profile {
        let (w, h) = (m.w as usize, m.h as usize);
        let mut p = Profile {
            w: m.w,
            h: m.h,
            top: vec![m.h; w],
            bot: vec![0; w],
            left: vec![m.w; h],
            right: vec![0; h],
            rotation: 0,
        };
        for r in 0..m.h {
            let spans = m.row(r);
            if let (Some(first), Some(last)) = (spans.first(), spans.last()) {
                p.left[r as usize] = first[0];
                p.right[r as usize] = last[1];
            }
            for &[a, b] in spans {
                for c in a as usize..b as usize {
                    p.top[c] = p.top[c].min(r);
                    p.bot[c] = r + 1;
                }
            }
        }
        p
    }

    /// An L-shaped chart: profiles of every rotation must match the profiles
    /// read off the rotated masks.
    #[test]
    fn rotated_profiles_match_rotated_masks() {
        let uvs: Vec<V2> = vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 5.0],
            [0.0, 5.0],
        ];
        let tris = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]];
        let masks = chart_masks(&uvs, &tris, [3.0, 5.0], 4.3, 1, true);
        let mut p = Profile::from_bits(&rasterize_chart(&uvs, &tris, [3.0, 5.0], 4.3, 1));
        assert_eq!(p, profile_of(&masks.rot[0]));
        for (k, m) in masks.rot.iter().enumerate() {
            let mut expected = profile_of(m);
            expected.rotation = k as u8;
            assert_eq!(p, expected, "rotation {k}");
            // Downsampling commutes with reading the profile.
            for b in [2, 3, 4] {
                let mut expected = profile_of(&m.downsample(b));
                expected.rotation = k as u8;
                assert_eq!(p.downsample(b), expected, "rotation {k}, block {b}");
            }
            p = p.rotate90();
        }
    }

    /// An L: `w × h` with the bottom-right quadrant from row `h0` and column
    /// `w0` empty.
    fn l_shape(w: u32, h: u32, w0: u32, h0: u32) -> Profile {
        let mut p = Profile::rect(w, h);
        for c in w0..w {
            p.bot[c as usize] = h0;
        }
        for r in h0..h {
            p.right[r as usize] = w0;
        }
        p
    }

    /// A row pushed against an overhang leaves a pocket under it, which
    /// takes a small item that would otherwise start a row of its own.
    #[test]
    fn small_items_fill_pockets_under_overhangs() {
        // The L's right half ends at row 4; the full-width rect below it is
        // held at row 10 by its left half, leaving rows 4..10 free on the
        // right.
        let items = vec![
            l_shape(20, 10, 10, 4),
            Profile::rect(20, 10),
            Profile::rect(6, 4),
        ];
        let layout = pack_folded(&items, 20, 30, 1).unwrap();
        let p = &layout.placements;
        assert_eq!((p[0].x, p[0].y), (0, 0));
        assert_eq!((p[1].x, p[1].y), (0, 10));
        assert!(p[2].x >= 10 && p[2].y == 4, "({}, {})", p[2].x, p[2].y);
        assert_eq!(layout.page_sizes, vec![(20, 20)]);
        assert_disjoint(&items, &layout, 30);
    }

    /// A tall item followed by short ones leaves a knee: the next row is
    /// folded into the concavity beside the tall item rather than across the
    /// full width, where its leftmost items would rest on the tall one.
    #[test]
    fn rows_fold_into_a_knee() {
        let mut items = vec![Profile::rect(6, 60)];
        items.extend((0..60).map(|_| Profile::rect(4, 8)));
        let layout = pack_folded(&items, 100, 100, 1).unwrap();
        let p = &layout.placements;
        // Row 1: the tall item and 23 short ones (6 + 23 × 4 = 98 columns).
        assert!(p[..24].iter().all(|q| q.y == 0));
        // Row 2: 23 short items in the knee's columns [6, 100), at y = 8.
        for (i, q) in p[24..47].iter().enumerate() {
            assert!(q.x >= 6 && q.y == 8, "item {}: ({}, {})", 24 + i, q.x, q.y);
        }
        assert_disjoint(&items, &layout, 100);
    }

    /// Asserts that no two items' profiles overlap and all stay within their
    /// `size × size` page.
    fn assert_disjoint(items: &[Profile], layout: &Layout, size: u32) {
        let pages = layout.page_sizes.len();
        let mut grid = vec![vec![false; (size * size) as usize]; pages];
        for (p, pl) in items.iter().zip(&layout.placements) {
            assert!(pl.x + p.w <= size && pl.y + p.h <= size);
            for r in 0..p.h {
                for c in p.left[r as usize]..p.right[r as usize] {
                    let t = &mut grid[pl.page as usize][((pl.y + r) * size + pl.x + c) as usize];
                    assert!(!*t, "overlap at ({}, {})", pl.x + c, pl.y + r);
                    *t = true;
                }
            }
        }
    }

    /// Random L shapes and rectangles: no two items' masks overlap, and all
    /// stay on their page.
    #[test]
    fn folded_items_dont_overlap() {
        let mut seed = 1u64;
        let mut next = |n: u32| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % n as u64) as u32
        };
        let mut masks = Vec::new();
        for _ in 0..300 {
            let (w, h) = (2 + next(30), 2 + next(30));
            let (a, b) = (1 + next(w - 1), 1 + next(h - 1));
            // An L: full bottom-left column and top row, cut at (a, b).
            let mut bits = vec![false; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    bits[(y * w + x) as usize] = x < a || y < b;
                }
            }
            masks.push((w, h, bits));
        }
        let profiles: Vec<Profile> = masks
            .iter()
            .map(|(w, h, bits)| {
                let mut p = Profile::rect(*w, *h);
                for c in 0..*w as usize {
                    let rows = (0..*h as usize).filter(|&r| bits[r * *w as usize + c]);
                    p.top[c] = rows.clone().min().unwrap() as u32;
                    p.bot[c] = rows.max().unwrap() as u32 + 1;
                }
                for r in 0..*h as usize {
                    let cols = (0..*w as usize).filter(|&c| bits[r * *w as usize + c]);
                    p.left[r] = cols.clone().min().unwrap() as u32;
                    p.right[r] = cols.max().unwrap() as u32 + 1;
                }
                p.oriented(true)
            })
            .collect();
        let layout = pack_folded(&profiles, 160, 160, usize::MAX).unwrap();
        assert_disjoint(&profiles, &layout, 160);
    }
}
