//! Atlas packing. Charts are placed either by their rasterized shape (see
//! [`crate::mask`]) or folded into rows by their profiles ([`crate::fold`]);
//! instanced meshes are placed as one rectangle per instance
//! ([`crate::tile`]). Three modes choose the density and pages:
//!
//! - [`pack_auto`]: the highest density that fits one page (bisection);
//! - [`pack_unbounded`]: a fixed density on one page that grows to fit;
//! - [`pack_pages`]: a fixed density on as many fixed-size pages as needed.

use crate::fold::{pack_folded, Profile};
use crate::geom::V2;
use crate::mask::{chart_masks, chart_profile, content_size, ChartMasks, Mask, Page};
use crate::par;
use crate::progress::{Ctx, Stage};
use crate::tile::{layout_tile, TileLayout, TileSpec};
use crate::{Error, Packing};

/// Where an item (chart or instance rectangle) was placed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Placement {
    pub page: u32,
    /// Top-left corner of the padded rectangle, in texels.
    pub x: u32,
    pub y: u32,
    /// Quarter turns counter-clockwise applied to the chart content.
    pub rotation: u8,
}

/// The packed atlas.
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

/// What the packer needs to know about a chart.
#[derive(Clone, Copy)]
pub(crate) struct ChartGeom<'a> {
    /// UVs in world units, minimum at the origin.
    pub uvs: &'a [V2],
    pub tris: &'a [[u32; 3]],
    pub size: V2,
    pub area: f64,
}

/// Everything to pack, and how.
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

/// `x` rounded up to a multiple of `b`.
fn round_up(x: u32, b: u32) -> u32 {
    x.div_ceil(b).saturating_mul(b)
}

/// Most texels (padded item area) a pack may rasterize: 2^30, a 32K × 32K
/// atlas.
const MAX_TEXELS: u64 = 1 << 30;

/// Placed items (charts, then instance rectangles) and the used size of each
/// page.
pub(crate) struct Layout {
    pub placements: Vec<Placement>,
    /// `(width, height)` actually used on each page.
    pub page_sizes: Vec<(u32, u32)>,
}

/// Everything to place at one density: the charts, then one rectangle per
/// tile instance.
struct Prepared {
    /// Padded sizes in texels, rounded up to whole blocks.
    sizes: Vec<(u32, u32)>,
    /// Masks in block units (shape packing only).
    masks: Vec<ChartMasks>,
    /// Profiles in block units, oriented for folding (fast packing only).
    profiles: Vec<Profile>,
    tiles: Vec<TileLayout>,
}

impl Packer<'_> {
    /// Total surface area to pack, over all charts and instances.
    fn area(&self) -> f64 {
        self.charts.iter().map(|c| c.area).sum::<f64>()
            + self.tiles.iter().map(TileSpec::area).sum::<f64>()
    }

    /// Total chart bounding-box area to pack.
    fn bbox_area(&self) -> f64 {
        self.charts
            .iter()
            .map(|c| c.size[0] * c.size[1])
            .sum::<f64>()
            + self.tiles.iter().map(TileSpec::bbox_area).sum::<f64>()
    }

    /// Sizes (and tile layouts) of everything to place at density `s`, for
    /// pages of `page` texels if bounded. Masks are rasterized separately
    /// ([`Packer::rasterize`]), once the sizes are known to be reasonable.
    fn prepare(&self, s: f64, page: Option<u32>) -> Result<Prepared, Error> {
        let tiles = self
            .tiles
            .iter()
            .map(|t| layout_tile(self, t, s, page))
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
        for (t, layout) in self.tiles.iter().zip(&tiles) {
            for &k in t.scales {
                sizes.push(padded(content_size(layout.inner, k * s)));
            }
        }
        Ok(Prepared {
            sizes,
            masks: Vec::new(),
            profiles: Vec::new(),
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

    /// Rasterizes the charts for placement, in block units: all rotations'
    /// masks for shape packing, oriented profiles for fast packing; then a
    /// rectangle per instance. Charts are rotated in texels, then
    /// downsampled, so blocks line up with the rotated chart.
    fn rasterize(&self, prep: &mut Prepared, s: f64) {
        if !prep.masks.is_empty() || !prep.profiles.is_empty() {
            return;
        }
        let b = self.block;
        let instances = prep.sizes[self.charts.len()..]
            .iter()
            .map(|&(w, h)| (w / b, h / b));
        match self.method {
            Packing::Shape => {
                prep.masks = par::map_ref(self.charts, |c| {
                    let m = chart_masks(
                        c.uvs,
                        c.tris,
                        c.size,
                        chart_density(c.size, s, self.max_chart_size),
                        self.padding,
                        self.allow_rotation,
                    );
                    if b > 1 {
                        ChartMasks {
                            rot: m.rot.iter().map(|r| r.downsample(b)).collect(),
                        }
                    } else {
                        m
                    }
                });
                prep.masks.extend(instances.map(|(w, h)| ChartMasks {
                    rot: vec![Mask::rect(w, h)],
                }));
            }
            Packing::Fast => {
                prep.profiles = par::map_ref(self.charts, |c| {
                    let s = chart_density(c.size, s, self.max_chart_size);
                    chart_profile(c.uvs, c.tris, c.size, s, self.padding)
                        .oriented(self.allow_rotation)
                        .downsample(b)
                });
                prep.profiles
                    .extend(instances.map(|(w, h)| Profile::rect(w, h)));
            }
        }
    }

    /// Places everything on at most `max_pages` pages of `width × height`
    /// texels (`height` may be `u32::MAX`: unbounded). `None` if it doesn't
    /// fit.
    fn place(&self, prep: &Prepared, width: u32, height: u32, max_pages: usize) -> Option<Layout> {
        // Pack in block units, then scale back to texels.
        let b = self.block;
        let height = if height == u32::MAX {
            u32::MAX
        } else {
            height / b
        };
        let mut layout = match self.method {
            Packing::Fast => pack_folded(&prep.profiles, width / b, height, max_pages)?,
            Packing::Shape => pack_shapes(&prep.masks, width / b, height, max_pages)?,
        };
        if b > 1 {
            for p in &mut layout.placements {
                (p.x, p.y) = (p.x * b, p.y * b);
            }
            for size in &mut layout.page_sizes {
                *size = (size.0 * b, size.1 * b);
            }
        }
        Some(layout)
    }

    /// The result: placements split into charts and per-tile instances, with
    /// pages of `width × height` at density `s`.
    fn finish(
        &self,
        prep: Prepared,
        layout: Layout,
        width: u32,
        height: u32,
        s: f64,
    ) -> PackResult {
        let pages = layout.page_sizes.len() as u32;
        let mut placements = layout.placements;
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
}

/// Packs chart masks (largest first), each at the lowest position on the
/// first page it fits, onto at most `max_pages` pages.
fn pack_shapes(masks: &[ChartMasks], width: u32, height: u32, max_pages: usize) -> Option<Layout> {
    let mut order: Vec<usize> = (0..masks.len()).collect();
    // Largest first; perimeter ordering packs slightly denser than area.
    order.sort_by_key(|&i| (std::cmp::Reverse(masks[i].rot[0].w + masks[i].rot[0].h), i));
    let mut placements = vec![Placement::default(); masks.len()];
    let mut pages: Vec<Page> = Vec::new();
    for i in order {
        let mut placed = None;
        for (pi, page) in pages.iter().enumerate() {
            if let Some(spot) = best_fit(page, &masks[i]) {
                placed = Some((pi, spot));
                break;
            }
        }
        if placed.is_none() {
            if pages.len() >= max_pages {
                return None;
            }
            pages.push(Page::new(width, height));
            placed = Some((pages.len() - 1, best_fit(pages.last()?, &masks[i])?));
        }
        let (pi, (x, y, rotation)) = placed.expect("placed above");
        pages[pi].insert(&masks[i].rot[rotation as usize], x, y);
        placements[i] = Placement {
            page: pi as u32,
            x,
            y,
            rotation,
        };
    }
    Some(Layout {
        placements,
        page_sizes: pages
            .iter()
            .map(|p| (p.used_width, p.used_height))
            .collect(),
    })
}

/// The rotation whose lowest placement has the lowest top edge:
/// `(x, y, rotation)`.
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

/// Packs at a fixed `texels_per_unit` (`None`: automatic) on pages of
/// `page_size` texels (`0`: one unbounded page).
pub(crate) fn pack(
    p: &Packer<'_>,
    texels_per_unit: Option<f64>,
    page_size: u32,
) -> Result<PackResult, Error> {
    if !(p.area() > 0.0) {
        // Nothing with surface area: everything collapses to the origin.
        let size = page_size.max(1);
        let s = texels_per_unit.unwrap_or(1.0);
        let items = p.charts.len() + p.tiles.iter().map(|t| t.scales.len()).sum::<usize>();
        let layout = Layout {
            placements: vec![Placement::default(); items],
            page_sizes: if items > 0 {
                vec![(size, size)]
            } else {
                Vec::new()
            },
        };
        return Ok(p.finish(p.prepare(s, None)?, layout, size, size, s));
    }
    match (texels_per_unit, page_size) {
        (Some(s), 0) => {
            p.ctx.advance(Stage::Packing, 1, None)?;
            pack_unbounded(p, s, u32::MAX)
        }
        (Some(s), size) => {
            p.ctx.advance(Stage::Packing, 1, None)?;
            pack_pages(p, s, size)
        }
        (None, size) => pack_auto(p, size),
    }
}

/// A packing that fits one page.
struct Fit {
    prep: Prepared,
    layout: Layout,
    density: f64,
}

/// The highest density at which everything fits one `size × size` page,
/// cropped to the rows used. If nothing fits one page, several pages at a
/// density low enough for every item.
fn pack_auto(p: &Packer<'_>, size: u32) -> Result<PackResult, Error> {
    // One packing trial at density `s`: the packing if it fits one page,
    // and for fast packing the height it needed. Folding onto an unbounded
    // page places the same rows as onto a bounded one until a row
    // overflows, so one fold answers both.
    let trial = |s: f64| -> Result<(Option<Fit>, Option<u32>), Error> {
        let mut prep = p.prepare(s, Some(size))?;
        if Packer::oversized(&prep, size).is_some() {
            return Ok((None, None));
        }
        Packer::check_budget(&prep)?;
        p.rasterize(&mut prep, s);
        let (layout, used) = match p.method {
            Packing::Fast => {
                let layout = p
                    .place(&prep, size, u32::MAX, 1)
                    .expect("an unbounded page always fits");
                let used = layout.page_sizes[0].1;
                ((used <= size).then_some(layout), Some(used))
            }
            Packing::Shape => (p.place(&prep, size, size, 1), None),
        };
        let fit = layout.map(|layout| Fit {
            prep,
            layout,
            density: s,
        });
        Ok((fit, used))
    };
    let full = size as f64 * size as f64;
    let mut best = None;
    // 100% coverage bounds the density from above.
    let (mut lo, mut hi) = (0.0, (full / p.area()).sqrt());
    if p.method == Packing::Shape {
        // A fast packing is a valid shape packing, so its density is a good
        // lower bound.
        let fast = Packer {
            method: Packing::Fast,
            ..*p
        };
        lo = match pack_auto(&fast, size) {
            Ok(r) if r.pages == 1 => r.texels_per_unit,
            _ => 0.0,
        };
        // Greedy placement isn't monotone in density, so shapes can fail
        // where folding fits: back off a little a few times, then leave it
        // to the bisection.
        for _ in 0..8 {
            if !(lo > 0.0) {
                break;
            }
            p.ctx.advance(Stage::Packing, 1, None)?;
            if let (Some(fit), _) = trial(lo)? {
                best = Some(fit);
                break;
            }
            lo *= 0.95;
        }
        if best.is_none() {
            lo = 0.0;
        }
    }
    // Narrow [lo, hi) until within 0.25%: bisect, or for fast packing jump
    // to the density predicted to just fill the page (used area grows with
    // the density squared) when that is inside the bracket. That halves
    // the trials to reach the page edge; heights step by whole rows near
    // it, so the last few trials are bisection either way.
    let mut next = 0.0;
    for _ in 0..24 {
        if best.is_some() && hi < lo * 1.0025 {
            break;
        }
        let mid = if lo > 0.0 { (lo * hi).sqrt() } else { 0.5 * hi };
        let s = if next > lo * 1.001 && next < hi / 1.001 {
            next
        } else {
            mid
        };
        p.ctx.advance(Stage::Packing, 1, None)?;
        let (fit, used) = trial(s)?;
        match fit {
            Some(fit) => {
                lo = s;
                best = Some(fit);
            }
            None => hi = s,
        }
        next = match used {
            Some(used) if used > 0 => s * (size as f64 / used as f64).sqrt(),
            _ => 0.0,
        };
    }
    if let Some(fit) = best {
        // Crop the page to the rows actually used.
        let used = fit.layout.page_sizes[0].1.max(1);
        return Ok(p.finish(fit.prep, fit.layout, size, used, fit.density));
    }
    // Too many charts for one page: several pages, at a density low enough
    // that every item fits a page.
    let mut s = 0.5 * (full / p.bbox_area()).sqrt();
    for _ in 0..64 {
        match pack_pages(p, s, size) {
            Err(Error::ChartTooLarge { .. }) => s *= 0.5,
            result => return result,
        }
    }
    pack_pages(p, s, size)
}

/// Fixed density `s` on one page that grows to fit: tries a few near-square
/// widths (at most `cap`) and keeps the smallest page within `cap × cap`, or
/// failing that the one with the shortest longer side.
pub(crate) fn pack_unbounded(p: &Packer<'_>, s: f64, cap: u32) -> Result<PackResult, Error> {
    let mut prep = p.prepare(s, None)?;
    Packer::check_budget(&prep)?;
    p.rasterize(&mut prep, s);
    let r = &prep.sizes;
    let bbox: f64 = r.iter().map(|&(w, h)| w as f64 * h as f64).sum();
    let fill = if p.method == Packing::Shape {
        0.8
    } else {
        0.95
    };
    let max_side = r.iter().map(|&(w, h)| w.max(h)).max().unwrap_or(1);
    let trial = |k: usize| {
        // Seven widths around the estimate, plus the cap itself.
        let side = if k == 7 {
            cap as f64
        } else {
            (bbox * fill).sqrt() * (0.85 + 0.05 * k as f64)
        };
        let width = round_up(max_side.max((side.ceil() as u32).min(cap)), p.block);
        let layout = p
            .place(&prep, width, u32::MAX, 1)
            .expect("an unbounded page always fits");
        let (w, h) = layout.page_sizes[0];
        let over = w.max(h) > cap;
        let size = if over {
            w.max(h) as u64
        } else {
            w as u64 * h as u64
        };
        ((over, size, k), layout)
    };
    let widths = if cap == u32::MAX { 7 } else { 8 };
    let (_, layout) = par::map((0..widths).collect(), trial)
        .into_iter()
        .min_by_key(|t| t.0)
        .expect("at least seven trials");
    let (w, h) = layout.page_sizes[0];
    Ok(p.finish(prep, layout, w.max(1), h.max(1), s))
}

/// Fixed density `s` on as many `size × size` pages as needed.
pub(crate) fn pack_pages(p: &Packer<'_>, s: f64, size: u32) -> Result<PackResult, Error> {
    let mut prep = p.prepare(s, Some(size))?;
    if let Some((width, height)) = Packer::oversized(&prep, size) {
        return Err(Error::ChartTooLarge {
            width,
            height,
            page_size: size,
        });
    }
    Packer::check_budget(&prep)?;
    p.rasterize(&mut prep, s);
    let layout = p
        .place(&prep, size, size, usize::MAX)
        .expect("every item fits an empty page");
    Ok(p.finish(prep, layout, size, size, s))
}
