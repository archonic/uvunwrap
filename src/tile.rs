//! Tiles: meshes drawn several times (instances). Their charts are packed
//! once into a local layout, and each instance gets a rectangle holding that
//! layout at its own scale, so all instances share one set of UVs.

use crate::geom::V2;
use crate::pack::{pack_unbounded, ChartGeom, Packer, Placement};
use crate::progress::QUIET;
use crate::Error;

/// An instanced mesh to lay out.
#[derive(Clone, Copy)]
pub(crate) struct TileSpec<'a> {
    pub charts: &'a [ChartGeom<'a>],
    /// Instance scales (> 0), relative to the chart geometry.
    pub scales: &'a [f64],
}

impl TileSpec<'_> {
    /// Total chart surface area over all instances.
    pub fn area(&self) -> f64 {
        self.charts.iter().map(|c| c.area).sum::<f64>() * self.sum_sq_scales()
    }

    /// Total chart bounding-box area over all instances.
    pub fn bbox_area(&self) -> f64 {
        self.charts
            .iter()
            .map(|c| c.size[0] * c.size[1])
            .sum::<f64>()
            * self.sum_sq_scales()
    }

    fn sum_sq_scales(&self) -> f64 {
        self.scales.iter().map(|k| k * k).sum()
    }
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

/// Lays out a tile's charts at atlas density `s` for its smallest instance,
/// so that its largest instance fits within `page` texels if possible.
pub(crate) fn layout_tile(
    p: &Packer<'_>,
    t: &TileSpec<'_>,
    s: f64,
    page: Option<u32>,
) -> Result<TileLayout, Error> {
    let k_min = t.scales.iter().copied().fold(f64::INFINITY, f64::min);
    let k_max = t.scales.iter().copied().fold(0.0, f64::max);
    let density = s * k_min;
    let pad = 2 * p.padding;
    // Instance `k` is the layout scaled by `k / k_min`, padding included:
    // limit the layout (and its charts) so the largest instance fits.
    let shrink = |texels: u32| ((texels as f64 * k_min / k_max).floor() as u32).max(1);
    let chart_limit = if p.max_chart_size == 0 {
        0
    } else {
        shrink(p.max_chart_size)
    };
    let cap = page.map_or(u32::MAX, |size| {
        shrink(size.saturating_sub(pad)).saturating_add(pad)
    });
    // Nested packing is neither reported nor cancelled; the trials around it
    // are.
    let sub = Packer {
        charts: t.charts,
        tiles: &[],
        max_chart_size: chart_limit,
        block: 1,
        ctx: &QUIET,
        ..*p
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
