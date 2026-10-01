//! Atlas packing of chart bounding rectangles with a bottom-left skyline packer.

use crate::Error;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Placement {
    pub page: u32,
    /// Top-left corner of the padded rectangle, in texels.
    pub x: u32,
    pub y: u32,
    /// Chart content is rotated by 90° counter-clockwise.
    pub rotated: bool,
}

pub(crate) struct PackResult {
    pub placements: Vec<Placement>,
    pub width: u32,
    pub height: u32,
    pub pages: u32,
    pub texels_per_unit: f64,
}

#[derive(Clone, Copy)]
struct Node {
    x: u32,
    y: u32,
    w: u32,
}

struct Skyline {
    width: u32,
    height: u32,
    nodes: Vec<Node>,
    used_height: u32,
}

impl Skyline {
    fn new(width: u32, height: u32) -> Self {
        Skyline { width, height, nodes: vec![Node { x: 0, y: 0, w: width }], used_height: 0 }
    }

    /// Best bottom-left position for a `w × h` rect: `(score, node, x, y)`.
    fn find(&self, w: u32, h: u32) -> Option<(u64, usize, u32, u32)> {
        if w > self.width || h > self.height {
            return None;
        }
        let mut best: Option<(u64, usize, u32, u32)> = None;
        for i in 0..self.nodes.len() {
            let x = self.nodes[i].x;
            if x as u64 + w as u64 > self.width as u64 {
                break;
            }
            let mut y = 0;
            let mut remaining = w as i64;
            let mut j = i;
            while remaining > 0 {
                y = y.max(self.nodes[j].y);
                remaining -= self.nodes[j].w as i64;
                j += 1;
            }
            if y as u64 + h as u64 > self.height as u64 {
                continue;
            }
            let score = ((y as u64 + h as u64) << 32) | x as u64;
            if best.map_or(true, |b| score < b.0) {
                best = Some((score, i, x, y));
            }
        }
        best
    }

    fn insert(&mut self, i: usize, x: u32, y: u32, w: u32, h: u32) {
        self.nodes.insert(i, Node { x, y: y + h, w });
        let j = i + 1;
        while j < self.nodes.len() {
            let prev_end = self.nodes[j - 1].x + self.nodes[j - 1].w;
            if self.nodes[j].x >= prev_end {
                break;
            }
            let shrink = prev_end - self.nodes[j].x;
            if self.nodes[j].w <= shrink {
                self.nodes.remove(j);
            } else {
                self.nodes[j].x += shrink;
                self.nodes[j].w -= shrink;
                break;
            }
        }
        let mut k = 0;
        while k + 1 < self.nodes.len() {
            if self.nodes[k].y == self.nodes[k + 1].y {
                self.nodes[k].w += self.nodes[k + 1].w;
                self.nodes.remove(k + 1);
            } else {
                k += 1;
            }
        }
        self.used_height = self.used_height.max(y + h);
    }

    /// Places a rect, optionally trying the 90° rotation. Returns `(x, y, rotated)`.
    fn place(&mut self, w: u32, h: u32, allow_rotation: bool) -> Option<(u32, u32, bool)> {
        let normal = self.find(w, h).map(|c| (c, false));
        let rotated = if allow_rotation && w != h { self.find(h, w).map(|c| (c, true)) } else { None };
        let ((_, i, x, y), rot) = match (normal, rotated) {
            (Some(a), Some(b)) => {
                if b.0 .0 < a.0 .0 {
                    b
                } else {
                    a
                }
            }
            (a, b) => a.or(b)?,
        };
        let (pw, ph) = if rot { (h, w) } else { (w, h) };
        self.insert(i, x, y, pw, ph);
        Some((x, y, rot))
    }
}

fn rects(sizes: &[[f64; 2]], s: f64, padding: u32) -> Vec<(u32, u32)> {
    let side = |v: f64| ((v * s).ceil().clamp(1.0, (u32::MAX / 4) as f64) as u32).saturating_add(2 * padding);
    sizes.iter().map(|sz| (side(sz[0]), side(sz[1]))).collect()
}

fn order(r: &[(u32, u32)]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..r.len()).collect();
    o.sort_by_key(|&i| (std::cmp::Reverse(r[i].0.max(r[i].1)), std::cmp::Reverse(r[i].0.min(r[i].1)), i));
    o
}

fn try_pack(r: &[(u32, u32)], width: u32, height: u32, max_pages: usize, allow_rotation: bool) -> Option<(Vec<Placement>, Vec<Skyline>)> {
    let mut placements = vec![Placement::default(); r.len()];
    let mut pages: Vec<Skyline> = Vec::new();
    for i in order(r) {
        let (w, h) = r[i];
        let mut done = false;
        for (pi, page) in pages.iter_mut().enumerate() {
            if let Some((x, y, rotated)) = page.place(w, h, allow_rotation) {
                placements[i] = Placement { page: pi as u32, x, y, rotated };
                done = true;
                break;
            }
        }
        if !done {
            if pages.len() >= max_pages {
                return None;
            }
            let mut page = Skyline::new(width, height);
            let (x, y, rotated) = page.place(w, h, allow_rotation)?;
            placements[i] = Placement { page: pages.len() as u32, x, y, rotated };
            pages.push(page);
        }
    }
    Some((placements, pages))
}

pub(crate) fn pack(
    sizes: &[[f64; 2]],
    texels_per_unit: Option<f64>,
    resolution: u32,
    padding: u32,
    allow_rotation: bool,
) -> Result<PackResult, Error> {
    let bbox_area: f64 = sizes.iter().map(|s| s[0] * s[1]).sum();
    if sizes.is_empty() || !(bbox_area > 0.0) {
        let res = resolution.max(1);
        return Ok(PackResult {
            placements: vec![Placement::default(); sizes.len()],
            width: res,
            height: res,
            pages: if sizes.is_empty() { 0 } else { 1 },
            texels_per_unit: texels_per_unit.unwrap_or(1.0),
        });
    }

    match (texels_per_unit, resolution) {
        // Fixed density, unbounded single page: pick a near-square width.
        (Some(s), 0) => {
            let r = rects(sizes, s, padding);
            let max_side = r.iter().map(|&(w, h)| w.max(h)).max().unwrap_or(1);
            let area: f64 = r.iter().map(|&(w, h)| w as f64 * h as f64).sum();
            let width = max_side.max((area * 1.15).sqrt().ceil() as u32);
            let (placements, pages) = try_pack(&r, width, u32::MAX, 1, allow_rotation).expect("unbounded page always fits");
            Ok(PackResult { placements, width, height: pages[0].used_height.max(1), pages: 1, texels_per_unit: s })
        }
        // Fixed density, fixed page size: as many pages as needed.
        (Some(s), res) => {
            let r = rects(sizes, s, padding);
            if let Some(&(w, h)) = r.iter().find(|&&(w, h)| w.max(h) > res) {
                return Err(Error::ChartTooLarge { width: w, height: h, resolution: res });
            }
            let (placements, pages) = try_pack(&r, res, res, usize::MAX, allow_rotation)
                .ok_or(Error::ChartTooLarge { width: res + 1, height: res + 1, resolution: res })?;
            Ok(PackResult { placements, width: res, height: res, pages: pages.len() as u32, texels_per_unit: s })
        }
        // Automatic density: the largest density that fits a single page.
        (None, res) => {
            let res = if res == 0 { 1024 } else { res };
            let mut hi = (res as f64 * res as f64 / bbox_area).sqrt();
            let mut lo = 0.0;
            let mut best = None;
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                match try_pack(&rects(sizes, mid, padding), res, res, 1, allow_rotation) {
                    Some((placements, pages)) => {
                        lo = mid;
                        best = Some((placements, mid, pages[0].used_height));
                    }
                    None => hi = mid,
                }
            }
            match best {
                // Crop the page to the rows actually used.
                Some((placements, s, used)) => {
                    Ok(PackResult { placements, width: res, height: used.max(1), pages: 1, texels_per_unit: s })
                }
                // Too many charts for one page even at minimal density.
                None => {
                    let s = 0.5 * (res as f64 * res as f64 / bbox_area).sqrt();
                    pack(sizes, Some(s), res, padding, allow_rotation)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skyline_packs_without_overlap() {
        let r: Vec<(u32, u32)> = (0..200).map(|i| (3 + (i * 7) % 23, 2 + (i * 13) % 17)).collect();
        let (pl, _) = try_pack(&r, 128, 128, usize::MAX, true).unwrap();
        let boxes: Vec<_> = pl
            .iter()
            .zip(&r)
            .map(|(p, &(w, h))| {
                let (w, h) = if p.rotated { (h, w) } else { (w, h) };
                (p.page, p.x, p.y, p.x + w, p.y + h)
            })
            .collect();
        for (i, a) in boxes.iter().enumerate() {
            assert!(a.3 <= 128 && a.4 <= 128);
            for b in &boxes[i + 1..] {
                let overlap = a.0 == b.0 && a.1 < b.3 && b.1 < a.3 && a.2 < b.4 && b.2 < a.4;
                assert!(!overlap, "{a:?} overlaps {b:?}");
            }
        }
    }
}
