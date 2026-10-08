//! Rectangle packing with a bottom-left skyline: the fast [`Packing::Rect`]
//! method, and the lower bound for the shape packer's density search.
//!
//! [`Packing::Rect`]: crate::Packing::Rect

use crate::pack::{Layout, Placement};

/// A horizontal segment of the skyline: the free space above `y` over
/// `[x, x + w)`.
#[derive(Clone, Copy)]
struct Node {
    x: u32,
    y: u32,
    w: u32,
}

/// Where a rectangle would go.
#[derive(Clone, Copy)]
struct Spot {
    /// Top edge, then x: lower is better (bottom-left first).
    score: u64,
    /// Index of the node the rectangle starts on.
    node: usize,
    x: u32,
    y: u32,
}

/// One page's skyline.
struct Skyline {
    width: u32,
    height: u32,
    nodes: Vec<Node>,
    used_height: u32,
}

impl Skyline {
    fn new(width: u32, height: u32) -> Self {
        Skyline {
            width,
            height,
            nodes: vec![Node {
                x: 0,
                y: 0,
                w: width,
            }],
            used_height: 0,
        }
    }

    /// Best bottom-left spot for a `w × h` rectangle.
    fn find(&self, w: u32, h: u32) -> Option<Spot> {
        if w > self.width || h > self.height {
            return None;
        }
        let mut best: Option<Spot> = None;
        for i in 0..self.nodes.len() {
            let x = self.nodes[i].x;
            if x as u64 + w as u64 > self.width as u64 {
                break;
            }
            // The rectangle rests on the highest node it spans.
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
            if best.is_none_or(|b| score < b.score) {
                best = Some(Spot {
                    score,
                    node: i,
                    x,
                    y,
                });
            }
        }
        best
    }

    /// Raises the skyline under a `w × h` rectangle placed at `spot`.
    fn insert(&mut self, spot: Spot, w: u32, h: u32) {
        let i = spot.node;
        self.nodes.insert(
            i,
            Node {
                x: spot.x,
                y: spot.y + h,
                w,
            },
        );
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
        self.used_height = self.used_height.max(spot.y + h);
    }

    /// Places a rectangle, trying its 90° rotation too if allowed. Returns
    /// `(x, y, rotated)`.
    fn place(&mut self, w: u32, h: u32, allow_rotation: bool) -> Option<(u32, u32, bool)> {
        let normal = self.find(w, h).map(|s| (s, false));
        let rotated = if allow_rotation && w != h {
            self.find(h, w).map(|s| (s, true))
        } else {
            None
        };
        let (spot, rot) = match (normal, rotated) {
            (Some(a), Some(b)) => {
                if b.0.score < a.0.score {
                    b
                } else {
                    a
                }
            }
            (a, b) => a.or(b)?,
        };
        let (pw, ph) = if rot { (h, w) } else { (w, h) };
        self.insert(spot, pw, ph);
        Some((spot.x, spot.y, rot))
    }
}

/// Packs `(w, h)` rectangles, largest first, onto at most `max_pages`
/// `width × height` pages. `None` if they don't fit.
pub(crate) fn pack_rects(
    rects: &[(u32, u32)],
    rotatable: &[bool],
    width: u32,
    height: u32,
    max_pages: usize,
) -> Option<Layout> {
    let mut order: Vec<usize> = (0..rects.len()).collect();
    order.sort_by_key(|&i| {
        let (w, h) = rects[i];
        (std::cmp::Reverse(w.max(h)), std::cmp::Reverse(w.min(h)), i)
    });
    let mut placements = vec![Placement::default(); rects.len()];
    let mut pages: Vec<Skyline> = Vec::new();
    for i in order {
        let (w, h) = rects[i];
        let mut placed = None;
        for (pi, page) in pages.iter_mut().enumerate() {
            if let Some(spot) = page.place(w, h, rotatable[i]) {
                placed = Some((pi, spot));
                break;
            }
        }
        if placed.is_none() {
            if pages.len() >= max_pages {
                return None;
            }
            let mut page = Skyline::new(width, height);
            placed = Some((pages.len(), page.place(w, h, rotatable[i])?));
            pages.push(page);
        }
        let (pi, (x, y, rotated)) = placed.expect("placed above");
        placements[i] = Placement {
            page: pi as u32,
            x,
            y,
            rotation: rotated as u8,
        };
    }
    Some(Layout {
        placements,
        page_sizes: pages.iter().map(|p| (p.width, p.used_height)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skyline_packs_without_overlap() {
        let r: Vec<(u32, u32)> = (0..200)
            .map(|i| (3 + (i * 7) % 23, 2 + (i * 13) % 17))
            .collect();
        let layout = pack_rects(&r, &vec![true; r.len()], 128, 128, usize::MAX).unwrap();
        let boxes: Vec<_> = layout
            .placements
            .iter()
            .zip(&r)
            .map(|(p, &(w, h))| {
                let (w, h) = if p.rotation == 1 { (h, w) } else { (w, h) };
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
