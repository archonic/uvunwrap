//! The unwrap pipeline behind [`generate`](crate::generate): validate, weld
//! and segment each mesh, parameterize its charts, pack all charts (and
//! instance rectangles) into the atlas, then assemble each output mesh with
//! its seam vertices split.

use crate::geom::V2;
use crate::progress::{self, Ctx};
use crate::topology::{Topology, NONE};
use crate::{
    pack, par, param, raster, segment, tile, validate, Atlas, Density, Error, InstanceRect,
    MeshInput, Options, OutputMesh, OutputVertex,
};
use std::collections::HashMap;

/// Validates the input, then unwraps, packs and assembles the output.
pub(crate) fn run(meshes: &[MeshInput<'_>], options: &Options, ctx: &Ctx) -> Result<Atlas, Error> {
    validate::options(options)?;
    for (mi, m) in meshes.iter().enumerate() {
        validate::mesh(mi, m)?;
    }
    let unwrapped = unwrap_meshes(meshes, options, ctx)?;
    let (slots, packed) = pack_meshes(meshes, &unwrapped, options, ctx)?;

    // Global chart ids are numbered mesh by mesh.
    let chart_bases: Vec<usize> = unwrapped
        .iter()
        .scan(0, |next, u| {
            let base = *next;
            *next += u.charts.len();
            Some(base)
        })
        .collect();
    let assembled = par::map((0..meshes.len()).collect(), |i| {
        let mesh = assemble_mesh(
            &meshes[i],
            &unwrapped[i],
            slots[i],
            chart_bases[i],
            &packed,
            options,
        );
        ctx.advance(progress::Stage::Output, 1, Some(meshes.len()))?;
        Ok(mesh)
    });
    let mut out_meshes = Vec::with_capacity(meshes.len());
    let mut covered = 0.0;
    for result in assembled {
        let (mesh, chart_texels) = result?;
        // Summed chart by chart in mesh order, so the result never depends
        // on the thread count.
        covered = chart_texels.iter().fold(covered, |sum, t| sum + t);
        out_meshes.push(mesh);
    }

    let total = packed.width as f64 * packed.height as f64 * packed.pages.max(1) as f64;
    Ok(Atlas {
        width: packed.width,
        height: packed.height,
        page_count: packed.pages,
        texels_per_unit: packed.texels_per_unit as f32,
        chart_count: unwrapped.iter().map(|u| u.charts.len()).sum::<usize>() as u32,
        utilization: (covered / total).min(1.0) as f32,
        meshes: out_meshes,
    })
}

/// A mesh after segmentation and parameterization.
struct Unwrapped {
    topo: Topology,
    charts: Vec<param::Chart>,
}

/// An optional limit as a bound: `0` is none.
fn limit(x: f32) -> f64 {
    if x > 0.0 {
        x as f64
    } else {
        f64::INFINITY
    }
}

/// Welds, segments and parameterizes every mesh, meshes and charts in
/// parallel.
fn unwrap_meshes(
    meshes: &[MeshInput<'_>],
    options: &Options,
    ctx: &Ctx,
) -> Result<Vec<Unwrapped>, Error> {
    let limits = segment::Limits {
        max_angle: (options.max_chart_angle as f64).to_radians(),
        max_area: limit(options.max_chart_area),
        max_boundary: limit(options.max_boundary_length),
    };
    let max_distortion = options.max_area_distortion as f64;
    let segmented = par::map_ref(meshes, |m| {
        if ctx.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let topo = Topology::build(m);
        let groups = if m.existing_uvs.is_some() {
            segment::components(&topo)
        } else {
            segment::segment(&topo, &limits)
        };
        ctx.advance(progress::Stage::Segmenting, 1, Some(meshes.len()))?;
        Ok((topo, groups))
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;

    // Progress counts faces across all meshes.
    let faces: usize = segmented.iter().flat_map(|(_, g)| g).map(Vec::len).sum();
    let unwrapped = par::map(
        segmented.into_iter().zip(meshes).collect(),
        |((topo, groups), m)| {
            // The input UVs of each face's corners, for repacking.
            let face_uvs: Option<Vec<[V2; 3]>> = m.existing_uvs.map(|uvs| {
                m.indices
                    .chunks_exact(3)
                    .map(|t| [0, 1, 2].map(|c| uvs[t[c] as usize].map(f64::from)))
                    .collect()
            });
            let charts = param::parameterize(
                &topo,
                groups,
                face_uvs.as_deref(),
                max_distortion,
                ctx,
                faces,
            );
            Unwrapped { topo, charts }
        },
    );
    if ctx.is_cancelled() {
        return Err(Error::Cancelled);
    }
    Ok(unwrapped)
}

/// Where a mesh's charts were packed.
#[derive(Clone, Copy)]
enum Slot {
    /// Directly into the atlas; its charts' placements start at index `first`.
    Direct { first: usize },
    /// As a tile (an instanced mesh), at this index of the packer's tiles.
    Tiled { tile: usize },
}

/// Packs the charts of meshes without instances directly and those of
/// instanced meshes as tiles.
fn pack_meshes(
    meshes: &[MeshInput<'_>],
    unwrapped: &[Unwrapped],
    options: &Options,
    ctx: &Ctx,
) -> Result<(Vec<Slot>, pack::PackResult), Error> {
    let geoms: Vec<Vec<pack::ChartGeom>> = unwrapped
        .iter()
        .map(|u| {
            u.charts
                .iter()
                .map(|c| pack::ChartGeom {
                    uvs: &c.uvs,
                    tris: &c.face_local,
                    size: c.size,
                    area: c.area,
                })
                .collect()
        })
        .collect();
    let scales: Vec<Vec<f64>> = meshes
        .iter()
        .map(|m| m.instance_scales.iter().map(|&k| k as f64).collect())
        .collect();
    let (mut slots, mut direct, mut tiles) = (Vec::new(), Vec::new(), Vec::new());
    for (i, m) in meshes.iter().enumerate() {
        if m.instance_scales.is_empty() {
            slots.push(Slot::Direct {
                first: direct.len(),
            });
            direct.extend(geoms[i].iter().copied());
        } else {
            slots.push(Slot::Tiled { tile: tiles.len() });
            tiles.push(tile::TileSpec {
                charts: &geoms[i],
                scales: &scales[i],
            });
        }
    }
    let packer = pack::Packer {
        charts: &direct,
        tiles: &tiles,
        padding: options.padding,
        allow_rotation: options.allow_rotation,
        method: options.packing,
        max_chart_size: options.max_chart_size,
        block: if options.block_align { 4 } else { 1 },
        ctx,
    };
    // The packer's form: a fixed density or none, and a page size or 0 for
    // one unbounded page.
    let (density, page_size) = match options.density {
        Density::Auto { page_size } => (None, page_size),
        Density::Fixed {
            texels_per_unit,
            page_size,
        } => (Some(texels_per_unit as f64), page_size.unwrap_or(0)),
    };
    Ok((slots, pack::pack(&packer, density, page_size)?))
}

/// A mesh's output: chart UVs placed in the atlas (or in its tile), seam
/// vertices duplicated, and its instance rectangles. Also returns the texel
/// area each chart covers, for utilization.
fn assemble_mesh(
    m: &MeshInput<'_>,
    u: &Unwrapped,
    slot: Slot,
    chart_base: usize,
    packed: &pack::PackResult,
    options: &Options,
) -> (OutputMesh, Vec<f64>) {
    let pad = options.padding as f64;
    let atlas_density = packed.texels_per_unit;
    let page = [packed.width as f64, packed.height as f64];
    let mut chart_texels = Vec::with_capacity(u.charts.len());
    let mut pages = Vec::with_capacity(u.charts.len());
    let uvs: Vec<Vec<[f32; 2]>> = u
        .charts
        .iter()
        .enumerate()
        .map(|(ci, ch)| match slot {
            Slot::Direct { first } => {
                let placement = packed.placements[first + ci];
                let density = pack::chart_density(ch.size, atlas_density, options.max_chart_size);
                chart_texels.push(ch.area * density * density);
                pages.push(Some(placement.page));
                place_chart(ch, placement, density, pad, [0.0, 0.0], page)
            }
            Slot::Tiled { tile } => {
                // UVs are relative to the tile's content box, inside its outer
                // padding; instance `k` shows it scaled by `k * atlas / layout`.
                let layout = &packed.tiles[tile];
                let density = pack::chart_density(ch.size, layout.density, layout.chart_limit);
                let per_instance = |k: f32| density * k as f64 * atlas_density / layout.density;
                chart_texels.push(
                    ch.area
                        * m.instance_scales
                            .iter()
                            .map(|&k| per_instance(k) * per_instance(k))
                            .sum::<f64>(),
                );
                pages.push(None);
                let content = layout
                    .inner
                    .map(|e| (e * layout.density).max(f64::MIN_POSITIVE));
                place_chart(ch, layout.placements[ci], density, pad, [pad, pad], content)
            }
        })
        .collect();
    let instances = match slot {
        Slot::Direct { .. } => Vec::new(),
        Slot::Tiled { tile } => instance_rects(
            m,
            &packed.tiles[tile],
            &packed.instances[tile],
            atlas_density,
            pad,
            page,
        ),
    };
    let (vertices, indices) = split_seams(m, u, &uvs, &pages, chart_base);
    (
        OutputMesh {
            vertices,
            indices,
            instances,
        },
        chart_texels,
    )
}

/// A chart's UVs in `[0, 1]` of the box at `origin` with size `extent` (in
/// texels of the page or tile it was placed on), given its placement and
/// texel density.
fn place_chart(
    chart: &param::Chart,
    placement: pack::Placement,
    density: f64,
    pad: f64,
    origin: V2,
    extent: V2,
) -> Vec<[f32; 2]> {
    let (cw, ch) = raster::content_size(chart.size, density);
    chart
        .uvs
        .iter()
        .map(|&[x, y]| {
            // Quarter-turn rotations keep orientation (no mirroring).
            let [x, y] =
                raster::rotate_texel([x * density, y * density], placement.rotation, cw, ch);
            let tx = placement.x as f64 + pad + x - origin[0];
            let ty = placement.y as f64 + pad + y - origin[1];
            [
                (tx / extent[0]).clamp(0.0, 1.0) as f32,
                (ty / extent[1]).clamp(0.0, 1.0) as f32,
            ]
        })
        .collect()
}

/// The atlas rectangle of each instance of a tiled mesh.
fn instance_rects(
    m: &MeshInput<'_>,
    layout: &tile::TileLayout,
    placements: &[pack::Placement],
    atlas_density: f64,
    pad: f64,
    page: V2,
) -> Vec<InstanceRect> {
    let uv = |x: f64, y: f64| {
        [
            (x / page[0]).clamp(0.0, 1.0) as f32,
            (y / page[1]).clamp(0.0, 1.0) as f32,
        ]
    };
    m.instance_scales
        .iter()
        .zip(placements)
        .map(|(&k, p)| {
            let (x, y) = (p.x as f64 + pad, p.y as f64 + pad);
            let [w, h] = layout.inner.map(|e| e * k as f64 * atlas_density);
            InstanceRect {
                page: p.page,
                min: uv(x, y),
                max: uv(x + w, y + h),
            }
        })
        .collect()
}

/// Output vertices and indices: one vertex per (input vertex, chart, chart
/// vertex), so vertices on chart seams are duplicated.
fn split_seams(
    m: &MeshInput<'_>,
    u: &Unwrapped,
    uvs: &[Vec<[f32; 2]>],
    pages: &[Option<u32>],
    chart_base: usize,
) -> (Vec<OutputVertex>, Vec<u32>) {
    let nf = u.topo.faces.len();
    let mut face_chart = vec![NONE; nf];
    let mut face_slot = vec![0u32; nf];
    // A (chart, chart vertex) for each welded vertex, for collapsing faces.
    let mut first = vec![(NONE, 0u32); u.topo.positions.len()];
    for (ci, ch) in u.charts.iter().enumerate() {
        for (k, &f) in ch.faces.iter().enumerate() {
            face_chart[f as usize] = ci as u32;
            face_slot[f as usize] = k as u32;
        }
        for (l, &w) in ch.verts.iter().enumerate() {
            if first[w as usize].0 == NONE {
                first[w as usize] = (ci as u32, l as u32);
            }
        }
    }

    // The (chart, chart vertex) that another triangle of face `f`'s polygon
    // gives welded vertex `w`, if any.
    let polygon_corner = |f: u32, w: u32| {
        u.topo.siblings(f).find_map(|g| {
            let ci = face_chart[g as usize];
            let c = u.topo.faces[g as usize].iter().position(|&x| x == w)?;
            (ci != NONE).then(|| {
                (
                    ci,
                    u.charts[ci as usize].face_local[face_slot[g as usize] as usize][c],
                )
            })
        })
    };

    let mut vertices = Vec::new();
    let mut indices = Vec::with_capacity(m.indices.len());
    let mut lookup: HashMap<(u32, u32, u32), u32> = HashMap::new();
    for (f, tri) in m.indices.chunks_exact(3).enumerate() {
        let chart = face_chart[f];
        // Degenerate and ignored faces get no UV area of their own. A corner
        // takes its polygon's UV for that vertex where another triangle of
        // the polygon has one, so each polygon corner keeps one UV; the
        // others collapse onto one existing UV of one of the corners, so the
        // face never stretches across the atlas.
        let welded = |c: usize| u.topo.welded[tri[c] as usize];
        let corners = if chart == NONE {
            [0, 1, 2].map(|c| polygon_corner(f as u32, welded(c)))
        } else {
            [None; 3]
        };
        let anchor = corners.iter().flatten().next().copied().or_else(|| {
            (0..3)
                .map(|c| first[welded(c) as usize])
                .find(|a| a.0 != NONE)
        });
        for (c, &orig) in tri.iter().enumerate() {
            let (ci, l) = if chart != NONE {
                (
                    chart,
                    u.charts[chart as usize].face_local[face_slot[f] as usize][c],
                )
            } else {
                corners[c].or(anchor).unwrap_or((NONE, NONE))
            };
            let index = *lookup.entry((orig, ci, l)).or_insert_with(|| {
                vertices.push(if ci == NONE {
                    OutputVertex {
                        xref: orig,
                        uv: [0.0, 0.0],
                        chart: None,
                        page: None,
                    }
                } else {
                    OutputVertex {
                        xref: orig,
                        uv: uvs[ci as usize][l as usize],
                        chart: Some((chart_base + ci as usize) as u32),
                        page: pages[ci as usize],
                    }
                });
                (vertices.len() - 1) as u32
            });
            indices.push(index);
        }
    }
    (vertices, indices)
}
