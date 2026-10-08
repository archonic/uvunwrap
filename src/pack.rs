//! Atlas packing. Charts are placed either by their rasterized shape (see
//! [`crate::mask`]) or by their bounding rectangle with a bottom-left skyline
//! packer; both share the texel density search.

use crate::geom::V2;
use crate::mask::{chart_masks, content_size, ChartMasks, Mask, Page};
use crate::progress::{Ctx, Stage, QUIET};
use crate::{Error, Packing};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Placement {
    pub page: u32,
    /// Top-left corner of the padded rectangle, in texels.
    pub x: u32,
    pub y: u32,
    /// Quarter turns counter-clockwise applied to the chart content.
    pub rotation: u8,
}

pub(crate) struct PackResult {
    /// One per chart in [`Packer::charts`].
    pub placements: Vec<Placement>,
    /// Per tile, one placement per instance (never rotated).
    pub instances: Vec<Vec<Placement>>,
    /// Per tile, the layout of its charts.
    pub tiles: Vec<TileLayout>,
    pub width: u32,
    pub height: u32,
    pub pages: u32,
    pub texels_per_unit: f64,
}

/// A mesh drawn several times, each instance needing its own atlas region:
/// its charts are packed once into a local layout, and each instance gets a
/// rectangle holding that layout at its own scale.
#[derive(Clone, Copy)]
pub(crate) struct TileSpec<'a> {
    pub charts: &'a [ChartGeom<'a>],
    /// Instance scales (> 0), relative to the chart geometry.
    pub scales: &'a [f64],
}

/// The local layout of a tile's charts.
#[derive(Clone, Debug)]
pub(crate) struct TileLayout {
    /// Chart placements in the layout, in texels at `density`.
    pub placements: Vec<Placement>,
    /// Texels per unit of the layout: the atlas density at the smallest
    /// instance scale, so every instance gets at least the full padding.
    pub density: f64,
    /// Largest chart side in layout texels (0: no limit), so no chart exceeds
    /// `max_chart_size` in the largest instance.
    pub chart_limit: u32,
    /// Size of the layout's content box (inside the outer padding), in chart
    /// units at scale 1. An instance of scale `k` occupies `inner * k * s`
    /// texels at atlas density `s`.
    pub inner: V2,
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
            if best.is_none_or(|b| score < b.0) {
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
        let rotated = if allow_rotation && w != h {
            self.find(h, w).map(|c| (c, true))
        } else {
            None
        };
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

/// `x` rounded up to a multiple of `b`.
fn round_up(x: u32, b: u32) -> u32 {
    x.div_euclid(b)
        .saturating_add(u32::from(x % b != 0))
        .saturating_mul(b)
}

/// Texel density for one chart: `s`, reduced so the chart's larger content
/// side stays within `max_size` texels (0: no limit).
pub(crate) fn chart_density(size: V2, s: f64, max_size: u32) -> f64 {
    let side = size[0].max(size[1]);
    if max_size == 0 || !(side * s > max_size as f64) {
        return s;
    }
    // Just under the limit, so rounding up in `content_size` can't exceed it.
    s.min(max_size as f64 / side * (1.0 - 1e-9))
}

/// What the packer needs to know about a chart.
#[derive(Clone, Copy)]
pub(crate) struct ChartGeom<'a> {
    /// UVs in world units, minimum at the origin.
    pub uvs: &'a [V2],
    pub tris: &'a [[u32; 3]],
    pub size: V2,
    pub area: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct Packer<'a> {
    pub charts: &'a [ChartGeom<'a>],
    pub tiles: &'a [TileSpec<'a>],
    pub padding: u32,
    pub allow_rotation: bool,
    pub method: Packing,
    /// Largest chart side in texels (0: no limit); bigger charts are scaled down.
    pub max_chart_size: u32,
    /// Charts occupy whole `block × block` texel blocks, never sharing one
    /// (1: no alignment).
    pub block: u32,
    /// Progress (one unit per packing trial) and cancellation.
    pub ctx: &'a Ctx<'a>,
}

/// Most texels (padded item area) a pack may rasterize: 2^30, a 32K × 32K
/// atlas.
const MAX_TEXELS: u64 = 1 << 30;

/// Placements (charts, then instances) plus the used `(width, height)` of
/// each page.
type Trial = (Vec<Placement>, Vec<(u32, u32)>);

/// Everything to place at one density: the charts, then one rectangle per
/// tile instance.
struct Prepared {
    /// Padded sizes in texels, rounded up to whole blocks.
    sizes: Vec<(u32, u32)>,
    /// Whether each item may be rotated.
    rotatable: Vec<bool>,
    /// Masks in block units (shape packing only).
    masks: Vec<ChartMasks>,
    tiles: Vec<TileLayout>,
}

impl Packer<'_> {
    /// Lays out a tile's charts at the density of its smallest instance, so
    /// that its largest instance fits within `page` texels if possible.
    fn layout_tile(
        &self,
        t: &TileSpec<'_>,
        s: f64,
        page: Option<u32>,
    ) -> Result<TileLayout, Error> {
        let k_min = t.scales.iter().copied().fold(f64::INFINITY, f64::min);
        let k_max = t.scales.iter().copied().fold(0.0, f64::max);
        let density = s * k_min;
        let pad = 2 * self.padding;
        // Instance `k` is the layout scaled by `k / k_min`, padding included:
        // limit the layout (and its charts) so the largest instance fits.
        let shrink = |texels: u32| ((texels as f64 * k_min / k_max).floor() as u32).max(1);
        let chart_limit = if self.max_chart_size == 0 {
            0
        } else {
            shrink(self.max_chart_size)
        };
        let cap = page.map_or(u32::MAX, |p| {
            shrink(p.saturating_sub(pad)).saturating_add(pad)
        });
        // Nested packing is neither reported nor cancelled; the trials
        // around it are.
        let sub = Packer {
            charts: t.charts,
            tiles: &[],
            max_chart_size: chart_limit,
            block: 1,
            ctx: &QUIET,
            ..*self
        };
        let r = pack_unbounded(&sub, density, cap)?;
        let inner = [
            r.width.saturating_sub(pad) as f64 / density,
            r.height.saturating_sub(pad) as f64 / density,
        ];
        Ok(TileLayout {
            placements: r.placements,
            density,
            chart_limit,
            inner,
        })
    }

    /// Sizes (and tile layouts) of everything to place at density `s`, for
    /// pages of `page` texels if bounded. Masks are rasterized separately
    /// ([`Packer::rasterize`]), once the sizes are known to be reasonable.
    fn prepare(&self, s: f64, page: Option<u32>) -> Result<Prepared, Error> {
        let tiles = self
            .tiles
            .iter()
            .map(|t| self.layout_tile(t, s, page))
            .collect::<Result<Vec<_>, _>>()?;
        let pad = 2 * self.padding;
        let b = self.block;
        let padded = |(w, h): (u32, u32)| {
            (
                round_up(w.saturating_add(pad), b),
                round_up(h.saturating_add(pad), b),
            )
        };
        let mut sizes: Vec<(u32, u32)> = self
            .charts
            .iter()
            .map(|c| {
                padded(content_size(
                    c.size,
                    chart_density(c.size, s, self.max_chart_size),
                ))
            })
            .collect();
        let mut rotatable = vec![self.allow_rotation; sizes.len()];
        for (t, layout) in self.tiles.iter().zip(&tiles) {
            for &k in t.scales {
                sizes.push(padded(content_size(layout.inner, k * s)));
                rotatable.push(false);
            }
        }
        Ok(Prepared {
            sizes,
            rotatable,
            masks: Vec::new(),
            tiles,
        })
    }

    /// Fails if the items' total area is beyond what can reasonably be
    /// rasterized (a density far too high for the meshes).
    fn check_budget(prep: &Prepared) -> Result<(), Error> {
        let texels: u64 = prep.sizes.iter().map(|&(w, h)| w as u64 * h as u64).sum();
        if texels > MAX_TEXELS {
            return Err(Error::AtlasTooLarge { texels });
        }
        Ok(())
    }

    /// The largest item, if any doesn't fit a `page × page` page.
    fn oversized(prep: &Prepared, page: u32) -> Option<(u32, u32)> {
        prep.sizes
            .iter()
            .copied()
            .filter(|&(w, h)| w.max(h) > page)
            .max_by_key(|&(w, h)| w.max(h))
    }

    /// Rasterizes the masks for shape packing.
    fn rasterize(&self, prep: &mut Prepared, s: f64) {
        if self.method != Packing::Shape || !prep.masks.is_empty() {
            return;
        }
        let b = self.block;
        prep.masks = self.masks(s);
        prep.masks.extend(
            prep.sizes[self.charts.len()..]
                .iter()
                .map(|&(w, h)| ChartMasks {
                    rot: vec![Mask::rect(w / b, h / b)],
                }),
        );
    }

    /// Chart masks in blocks.
    fn masks(&self, s: f64) -> Vec<ChartMasks> {
        let rotations = if self.allow_rotation { 4 } else { 1 };
        let one = |c: &ChartGeom| {
            let m = chart_masks(
                c.uvs,
                c.tris,
                c.size,
                chart_density(c.size, s, self.max_chart_size),
                self.padding,
                rotations,
            );
            if self.block > 1 {
                ChartMasks {
                    rot: m.rot.iter().map(|r| r.downsample(self.block)).collect(),
                }
            } else {
                m
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.charts.par_iter().map(one).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.charts.iter().map(one).collect()
        }
    }

    /// Splits trial placements into charts and per-tile instances.
    fn finish(
        &self,
        prep: Prepared,
        (mut placements, _): Trial,
        width: u32,
        height: u32,
        pages: u32,
        s: f64,
    ) -> PackResult {
        let mut rest = placements.split_off(self.charts.len()).into_iter();
        let instances = self
            .tiles
            .iter()
            .map(|t| rest.by_ref().take(t.scales.len()).collect())
            .collect();
        PackResult {
            placements,
            instances,
            tiles: prep.tiles,
            width,
            height,
            pages,
            texels_per_unit: s,
        }
    }

    fn place(&self, prep: &Prepared, width: u32, height: u32, max_pages: usize) -> Option<Trial> {
        // Pack in block units, then scale back to texels.
        let b = self.block;
        let (bw, bh) = (
            width / b,
            if height == u32::MAX {
                u32::MAX
            } else {
                height / b
            },
        );
        let (mut placements, mut used) = match self.method {
            Packing::Rect => {
                let r: Vec<(u32, u32)> = prep.sizes.iter().map(|&(w, h)| (w / b, h / b)).collect();
                let (placements, pages) = try_pack(&r, &prep.rotatable, bw, bh, max_pages)?;
                (
                    placements,
                    pages
                        .iter()
                        .map(|p| (p.width, p.used_height))
                        .collect::<Vec<_>>(),
                )
            }
            Packing::Shape => shape_pack(&prep.masks, bw, bh, max_pages)?,
        };
        if b > 1 {
            for p in &mut placements {
                (p.x, p.y) = (p.x * b, p.y * b);
            }
            for u in &mut used {
                *u = (u.0 * b, u.1 * b);
            }
        }
        Some((placements, used))
    }
}

fn shape_pack(masks: &[ChartMasks], width: u32, height: u32, max_pages: usize) -> Option<Trial> {
    let mut order: Vec<usize> = (0..masks.len()).collect();
    // Largest first; perimeter ordering packs slightly denser than area.
    order.sort_by_key(|&i| (std::cmp::Reverse(masks[i].rot[0].w + masks[i].rot[0].h), i));
    let mut placements = vec![Placement::default(); masks.len()];
    let mut pages: Vec<Page> = Vec::new();
    for i in order {
        let mut placed = None;
        for (pi, page) in pages.iter().enumerate() {
            if let Some((x, y, rotation)) = best_fit(page, &masks[i]) {
                placed = Some((pi, x, y, rotation));
                break;
            }
        }
        if placed.is_none() {
            if pages.len() >= max_pages {
                return None;
            }
            pages.push(Page::new(width, height));
            let (x, y, rotation) = best_fit(pages.last().unwrap(), &masks[i])?;
            placed = Some((pages.len() - 1, x, y, rotation));
        }
        let (pi, x, y, rotation) = placed.unwrap();
        pages[pi].insert(&masks[i].rot[rotation as usize], x, y);
        placements[i] = Placement {
            page: pi as u32,
            x,
            y,
            rotation,
        };
    }
    Some((
        placements,
        pages
            .iter()
            .map(|p| (p.used_width, p.used_height))
            .collect(),
    ))
}

/// The rotation whose lowest placement has the lowest top edge.
fn best_fit(page: &Page, masks: &ChartMasks) -> Option<(u32, u32, u8)> {
    let mut best: Option<(u64, u32, u32, u8)> = None;
    for (k, m) in masks.rot.iter().enumerate() {
        let limit = best.map_or(u64::MAX, |b| b.0);
        if let Some((x, y)) = page.find(m, limit) {
            let top = y as u64 + m.h as u64;
            if best.is_none_or(|b| top < b.0) {
                best = Some((top, x, y, k as u8));
            }
        }
    }
    best.map(|(_, x, y, k)| (x, y, k))
}

fn order(r: &[(u32, u32)]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..r.len()).collect();
    o.sort_by_key(|&i| {
        (
            std::cmp::Reverse(r[i].0.max(r[i].1)),
            std::cmp::Reverse(r[i].0.min(r[i].1)),
            i,
        )
    });
    o
}

fn try_pack(
    r: &[(u32, u32)],
    rotatable: &[bool],
    width: u32,
    height: u32,
    max_pages: usize,
) -> Option<(Vec<Placement>, Vec<Skyline>)> {
    let mut placements = vec![Placement::default(); r.len()];
    let mut pages: Vec<Skyline> = Vec::new();
    for i in order(r) {
        let (w, h) = r[i];
        let allow_rotation = rotatable[i];
        let mut done = false;
        for (pi, page) in pages.iter_mut().enumerate() {
            if let Some((x, y, rotated)) = page.place(w, h, allow_rotation) {
                placements[i] = Placement {
                    page: pi as u32,
                    x,
                    y,
                    rotation: rotated as u8,
                };
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
            placements[i] = Placement {
                page: pages.len() as u32,
                x,
                y,
                rotation: rotated as u8,
            };
            pages.push(page);
        }
    }
    Some((placements, pages))
}

pub(crate) fn pack(
    p: &Packer<'_>,
    texels_per_unit: Option<f64>,
    resolution: u32,
) -> Result<PackResult, Error> {
    let tile_area = |t: &TileSpec| {
        t.charts.iter().map(|c| c.area).sum::<f64>() * t.scales.iter().map(|k| k * k).sum::<f64>()
    };
    let area: f64 =
        p.charts.iter().map(|c| c.area).sum::<f64>() + p.tiles.iter().map(tile_area).sum::<f64>();
    if !(area > 0.0) {
        // Nothing with surface area: everything collapses to the origin.
        let res = resolution.max(1);
        let s = texels_per_unit.unwrap_or(1.0);
        let items = p.charts.len() + p.tiles.iter().map(|t| t.scales.len()).sum::<usize>();
        let trial = (vec![Placement::default(); items], Vec::new());
        return Ok(p.finish(
            p.prepare(s, None)?,
            trial,
            res,
            res,
            u32::from(items > 0),
            s,
        ));
    }

    match (texels_per_unit, resolution) {
        // Fixed density, unbounded single page.
        (Some(s), 0) => {
            p.ctx.advance(Stage::Packing, 1, 0)?;
            pack_unbounded(p, s, u32::MAX)
        }
        // Fixed density, fixed page size: as many pages as needed.
        (Some(s), res) => {
            p.ctx.advance(Stage::Packing, 1, 0)?;
            pack_pages(p, s, res)
        }
        // Automatic density: the largest density that fits a single page.
        (None, res) => {
            let res = if res == 0 { 1024 } else { res };
            // `Ok(None)`: doesn't fit one page at density `s`.
            let fits = |s: f64| -> Result<_, Error> {
                let mut prep = p.prepare(s, Some(res))?;
                if Packer::oversized(&prep, res).is_some() {
                    return Ok(None);
                }
                Packer::check_budget(&prep)?;
                p.rasterize(&mut prep, s);
                Ok(p.place(&prep, res, res, 1).map(|trial| {
                    let used = trial.1[0].1;
                    (prep, trial, s, used)
                }))
            };
            let tile_bbox = |t: &TileSpec| {
                t.charts.iter().map(|c| c.size[0] * c.size[1]).sum::<f64>()
                    * t.scales.iter().map(|k| k * k).sum::<f64>()
            };
            let bbox: f64 = p.charts.iter().map(|c| c.size[0] * c.size[1]).sum::<f64>()
                + p.tiles.iter().map(tile_bbox).sum::<f64>();
            let full = res as f64 * res as f64;
            let mut best = None;
            let (mut lo, mut hi);
            if p.method == Packing::Rect {
                (lo, hi) = (0.0, (full / bbox).sqrt());
            } else {
                // Rectangles are a valid shape packing, so the skyline result
                // is a good lower bound; 100% coverage bounds it from above.
                let rect = Packer {
                    method: Packing::Rect,
                    ..*p
                };
                lo = match pack(&rect, None, res) {
                    Ok(r) if r.pages == 1 => r.texels_per_unit,
                    _ => 0.0,
                };
                hi = (full / area).sqrt();
                while lo > 0.0 {
                    p.ctx.advance(Stage::Packing, 1, 0)?;
                    match fits(lo)? {
                        Some(b) => {
                            best = Some(b);
                            break;
                        }
                        None => lo *= 0.95,
                    }
                    if lo < hi * 1e-3 {
                        lo = 0.0;
                    }
                }
            }
            // Bisect until within 0.25%.
            for _ in 0..24 {
                if best.is_some() && hi < lo * 1.0025 {
                    break;
                }
                let mid = if lo > 0.0 { (lo * hi).sqrt() } else { 0.5 * hi };
                p.ctx.advance(Stage::Packing, 1, 0)?;
                match fits(mid)? {
                    Some(b) => {
                        lo = mid;
                        best = Some(b);
                    }
                    None => hi = mid,
                }
            }
            match best {
                // Crop the page to the rows actually used.
                Some((prep, trial, s, used)) => Ok(p.finish(prep, trial, res, used.max(1), 1, s)),
                // Too many charts for one page: several pages, at a density
                // low enough that every item fits a page.
                None => {
                    let mut s = 0.5 * (full / bbox).sqrt();
                    for _ in 0..64 {
                        match pack_pages(p, s, res) {
                            Err(Error::ChartTooLarge { .. }) => s *= 0.5,
                            result => return result,
                        }
                    }
                    pack_pages(p, s, res)
                }
            }
        }
    }
}

/// Fixed density `s` on one page that grows to fit: tries a few near-square
/// widths (at most `cap`) and keeps the smallest page within `cap × cap`, or
/// failing that the one with the shortest longer side.
fn pack_unbounded(p: &Packer<'_>, s: f64, cap: u32) -> Result<PackResult, Error> {
    let mut prep = p.prepare(s, None)?;
    Packer::check_budget(&prep)?;
    p.rasterize(&mut prep, s);
    let r = &prep.sizes;
    let bbox: f64 = r.iter().map(|&(w, h)| w as f64 * h as f64).sum();
    let fill = if p.method == Packing::Shape {
        0.8
    } else {
        1.15
    };
    let max_side = r.iter().map(|&(w, h)| w.max(h)).max().unwrap_or(1);
    let one = |k: usize| {
        // Seven widths around the estimate, plus the cap itself.
        let side = if k == 7 {
            cap as f64
        } else {
            (bbox * fill).sqrt() * (0.85 + 0.05 * k as f64)
        };
        let width = round_up(max_side.max((side.ceil() as u32).min(cap)), p.block);
        let (placements, used) = p
            .place(&prep, width, u32::MAX, 1)
            .expect("an unbounded page always fits");
        let side = used[0].0.max(used[0].1);
        let over = side > cap;
        let size = if over {
            side as u64
        } else {
            used[0].0 as u64 * used[0].1 as u64
        };
        ((over, size, k), placements, used[0])
    };
    let widths = if cap == u32::MAX { 7 } else { 8 };
    #[cfg(feature = "parallel")]
    let trials: Vec<_> = {
        use rayon::prelude::*;
        (0..widths).into_par_iter().map(one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let trials: Vec<_> = (0..widths).map(one).collect();
    let (_, placements, used) = trials
        .into_iter()
        .min_by_key(|t| t.0)
        .expect("at least seven trials");
    Ok(p.finish(
        prep,
        (placements, Vec::new()),
        used.0.max(1),
        used.1.max(1),
        1,
        s,
    ))
}

/// Fixed density `s` on as many `res × res` pages as needed.
fn pack_pages(p: &Packer<'_>, s: f64, res: u32) -> Result<PackResult, Error> {
    let mut prep = p.prepare(s, Some(res))?;
    if let Some((width, height)) = Packer::oversized(&prep, res) {
        return Err(Error::ChartTooLarge {
            width,
            height,
            resolution: res,
        });
    }
    Packer::check_budget(&prep)?;
    p.rasterize(&mut prep, s);
    let trial = p
        .place(&prep, res, res, usize::MAX)
        .expect("every item fits an empty page");
    let pages = trial.1.len() as u32;
    Ok(p.finish(prep, trial, res, res, pages, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skyline_packs_without_overlap() {
        let r: Vec<(u32, u32)> = (0..200)
            .map(|i| (3 + (i * 7) % 23, 2 + (i * 13) % 17))
            .collect();
        let (pl, _) = try_pack(&r, &vec![true; r.len()], 128, 128, usize::MAX).unwrap();
        let boxes: Vec<_> = pl
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
