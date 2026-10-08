//! # uvunwrap
//!
//! Pure-Rust automatic UV unwrapping, in the spirit of
//! [xatlas](https://github.com/jpcy/xatlas): generates unique, non-overlapping
//! texture coordinates suitable for lightmaps, texture baking and painting.
//!
//! The pipeline has three stages:
//!
//! 1. **Segmentation** — faces are grouped into charts by region growing within
//!    a normal cone ([`Options::max_chart_angle`]).
//! 2. **Parameterization** — each chart is flattened with Least Squares
//!    Conformal Maps (or planar projection when flat). Results with flipped
//!    triangles, self-overlap or too much area distortion
//!    ([`Options::max_area_distortion`]) are split and retried, so the output is
//!    always valid.
//! 3. **Packing** — charts are rasterized and packed by shape into one or more
//!    atlas pages, nesting into each other's concavities ([`Packing`]).
//!
//! ```
//! // A unit quad made of two triangles.
//! let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
//! let indices = [0, 1, 2, 0, 2, 3];
//! let mesh = uvunwrap::MeshInput { positions: &positions, indices: &indices, ..Default::default() };
//!
//! let atlas = uvunwrap::generate(&[mesh], &uvunwrap::Options::default()).unwrap();
//! let out = &atlas.meshes[0];
//! assert_eq!(out.indices.len(), indices.len());
//! for v in &out.vertices {
//!     // `xref` points back to the input vertex; copy its attributes.
//!     let _original_position = positions[v.xref as usize];
//!     assert!((0.0..=1.0).contains(&v.uv[0]) && (0.0..=1.0).contains(&v.uv[1]));
//! }
//! ```
//!
//! Output meshes have the same triangles in the same order, but vertices on
//! chart seams are duplicated, so each output vertex carries an
//! [`OutputVertex::xref`] to the input vertex it came from.
//!
//! # Determinism
//!
//! The same input, options and crate features give bit-identical output
//! regardless of thread count. The platform math library is never used, so
//! without the `direct-solver` feature the result depends only on IEEE-754
//! arithmetic and is expected to match on every 64-bit platform and wasm32.
//! With `direct-solver`, `faer` picks SIMD kernels at runtime, so output may
//! differ between CPU architectures (e.g. x86-64 vs ARM); disable it if you
//! cache unwraps or bake lightmaps on one machine and use them on another.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// `!(x > 0.0)` is deliberate throughout: unlike `x <= 0.0`, it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod geom;
mod mask;
mod pack;
mod param;
mod progress;
mod segment;
mod solver;
mod topology;

use progress::Ctx;
pub use progress::{Progress, Stage};
use std::collections::HashMap;
use std::fmt;
use topology::{Topology, NONE};

/// An indexed triangle mesh to unwrap.
///
/// Vertices with bit-identical positions are welded, so meshes whose vertices
/// are split for normals or texture coordinates unwrap as one surface. Edges
/// where the input *is* split (hard edges, UV seams) are preferred as chart
/// boundaries.
///
/// ```
/// # let (positions, indices) = (Vec::<[f32; 3]>::new(), Vec::<u32>::new());
/// let mesh = uvunwrap::MeshInput { positions: &positions, indices: &indices, ..Default::default() };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MeshInput<'a> {
    /// Vertex positions.
    pub positions: &'a [[f32; 3]],
    /// Triangle list indices into `positions`, counter-clockwise front faces.
    pub indices: &'a [u32],
    /// Optional per-face material ids (one per triangle). Charts never span
    /// faces with different ids.
    pub face_materials: Option<&'a [u32]>,
    /// Optional per-face flags (one per triangle). Flagged faces get no chart:
    /// their corners collapse onto an existing UV, as for degenerate faces.
    pub face_ignore: Option<&'a [bool]>,
    /// Factor from mesh units to world units, so instances drawn at different
    /// scales share one texel density. Also usable as a per-mesh importance
    /// factor. Must be finite and `> 0`. Default `1`.
    pub scale: f32,
    /// Scales of the instances of this mesh, each multiplying `scale`. Empty
    /// (the default): the mesh's charts are packed into the atlas directly and
    /// its UVs are atlas coordinates. Otherwise the mesh is unwrapped once,
    /// its UVs cover `[0, 1]` of its own layout, and each instance gets its own
    /// unrotated rectangle on one page holding that layout at the instance's
    /// scale ([`OutputMesh::instances`]). This is how engines lightmap meshes
    /// drawn several times (e.g. Bevy's `Lightmap::uv_rect`), and with a single
    /// instance it keeps a mesh on one page. Each must be finite and `> 0`.
    ///
    /// All instances share one layout, padded for the smallest instance, so
    /// an instance `n` times larger gets `n` times the padding: keep scales
    /// within a small range (split very different ones into separate meshes).
    /// [`Options::max_chart_size`] applies to every instance.
    pub instances: &'a [f32],
    /// Optional existing texture coordinates (one per vertex) to repack
    /// instead of unwrapping from scratch: each UV island (faces connected
    /// through identical UVs) becomes one chart, scaled to its surface area so
    /// texel density stays uniform, and rotated freely. Mirrored islands are
    /// flipped; islands with flipped or degenerate triangles or self-overlap
    /// are unwrapped normally. Default `None`.
    pub uvs: Option<&'a [[f32; 2]]>,
}

impl Default for MeshInput<'_> {
    fn default() -> Self {
        MeshInput {
            positions: &[],
            indices: &[],
            face_materials: None,
            face_ignore: None,
            scale: 1.0,
            instances: &[],
            uvs: None,
        }
    }
}

/// Unwrapping options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// Maximum angle in degrees between a face normal and its chart's average
    /// normal. Lower values give flatter charts but more seams. Must be in
    /// `(0, 90)`. Default `70`.
    pub max_chart_angle: f32,
    /// Maximum allowed ratio between a triangle's relative UV area and its
    /// relative surface area (in either direction). Charts exceeding it are
    /// split. Must be `>= 1`. Default `4`.
    pub max_area_distortion: f32,
    /// Atlas page size in texels. With `texels_per_unit: None`, the density is
    /// chosen automatically so everything fits one page of this size (`0`
    /// means `1024`). With a fixed density, `0` means a single page of whatever
    /// size is needed, otherwise as many pages as necessary are created.
    /// Default `1024`.
    pub resolution: u32,
    /// Texels per world unit. `None` picks it automatically. Default `None`.
    pub texels_per_unit: Option<f32>,
    /// Empty texels around each chart, to avoid bleeding when filtering or
    /// generating mipmaps. Default `2`.
    pub padding: u32,
    /// Allow rotating charts by multiples of 90° during packing. Default `true`.
    ///
    /// Rotations are proper (never mirrored): every output triangle is
    /// counter-clockwise in UV space in its input index order, even where the
    /// input winding is inconsistent (what xatlas calls `fixWinding`).
    pub allow_rotation: bool,
    /// Packing algorithm. Default [`Packing::Shape`].
    pub packing: Packing,
    /// Largest chart width or height in texels, excluding padding (`0`: no
    /// limit). Bigger charts are scaled down to fit instead of growing the page
    /// or failing, so they get a lower texel density than the rest. Default `0`.
    pub max_chart_size: u32,
    /// Align charts to 4×4 texel blocks: every chart, with its padding,
    /// occupies whole blocks at block-aligned positions and no two charts share
    /// a block, so block-compressed (BC/ASTC 4×4) atlases don't bleed between
    /// charts. Page sizes become multiples of 4; `resolution` must be one.
    /// Default `false`.
    pub block_align: bool,
}

/// How charts are arranged in the atlas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Packing {
    /// Pack rasterized chart shapes, so charts nest into each other's
    /// concavities and holes. Denser atlases; slower to pack.
    #[default]
    Shape,
    /// Pack chart bounding rectangles with a skyline packer. Much faster,
    /// but wastes the space around non-rectangular charts.
    Rect,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_chart_angle: 70.0,
            max_area_distortion: 4.0,
            resolution: 1024,
            texels_per_unit: None,
            padding: 2,
            allow_rotation: true,
            packing: Packing::Shape,
            max_chart_size: 0,
            block_align: false,
        }
    }
}

/// The generated atlas.
#[derive(Clone, Debug, PartialEq)]
pub struct Atlas {
    /// Page width in texels.
    pub width: u32,
    /// Page height in texels.
    pub height: u32,
    /// Number of atlas pages (see [`OutputVertex::page`]).
    pub page_count: u32,
    /// Texel density that was used (charts scaled down by
    /// [`Options::max_chart_size`] have less).
    pub texels_per_unit: f32,
    /// Total number of charts across all meshes.
    pub chart_count: u32,
    /// Fraction of atlas texels covered by chart surface (0..1).
    pub utilization: f32,
    /// One output mesh per input mesh, in input order.
    pub meshes: Vec<OutputMesh>,
}

/// An unwrapped mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OutputMesh {
    /// New vertices; seam vertices are duplicated.
    pub vertices: Vec<OutputVertex>,
    /// Same triangles as the input, indexing `vertices`.
    pub indices: Vec<u32>,
    /// One rectangle per entry of [`MeshInput::instances`] (empty if none).
    pub instances: Vec<InstanceRect>,
}

/// Where one instance of a mesh lives in the atlas. The instance's texture
/// coordinate for a vertex is `min + uv * (max - min)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceRect {
    /// Atlas page.
    pub page: u32,
    /// Corner of the rectangle with the smallest coordinates, in `[0, 1]` page
    /// coordinates.
    pub min: [f32; 2],
    /// Opposite corner.
    pub max: [f32; 2],
}

/// A vertex of an [`OutputMesh`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputVertex {
    /// Index of the input vertex this vertex was created from.
    pub xref: u32,
    /// Texture coordinate in `[0, 1]`, relative to its page (or, for meshes
    /// with [`MeshInput::instances`], to each instance's rectangle). `v` grows
    /// with the texel row (row 0 at `v = 0`); flip it if your convention
    /// differs.
    pub uv: [f32; 2],
    /// Global chart id, or `None` for vertices only used by degenerate or
    /// ignored triangles that have no charted corner.
    pub chart: Option<u32>,
    /// Atlas page this vertex belongs to (`0` for meshes with instances; see
    /// [`InstanceRect::page`]).
    pub page: u32,
}

/// Errors returned by [`generate`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The index count of a mesh is not a multiple of 3.
    InvalidIndexCount {
        /// Mesh position in the input slice.
        mesh: usize,
        /// The offending index count.
        count: usize,
    },
    /// An index refers to a vertex that doesn't exist.
    IndexOutOfRange {
        /// Mesh position in the input slice.
        mesh: usize,
        /// The offending index value.
        index: u32,
        /// Number of vertices in the mesh.
        vertex_count: usize,
    },
    /// A vertex position contains NaN or infinity.
    NonFinitePosition {
        /// Mesh position in the input slice.
        mesh: usize,
        /// Vertex index.
        vertex: usize,
    },
    /// A vertex of [`MeshInput::uvs`] contains NaN or infinity.
    NonFiniteUv {
        /// Mesh position in the input slice.
        mesh: usize,
        /// Vertex index.
        vertex: usize,
    },
    /// A per-face array doesn't have one entry per triangle.
    FaceDataLength {
        /// Mesh position in the input slice.
        mesh: usize,
        /// Length of the per-face array.
        len: usize,
        /// Number of triangles in the mesh.
        face_count: usize,
    },
    /// [`MeshInput::uvs`] doesn't have one entry per vertex.
    UvCount {
        /// Mesh position in the input slice.
        mesh: usize,
        /// Number of UVs.
        len: usize,
        /// Number of vertices in the mesh.
        vertex_count: usize,
    },
    /// [`MeshInput::scale`] or an instance scale is not finite and positive.
    InvalidScale {
        /// Mesh position in the input slice.
        mesh: usize,
    },
    /// An option is out of range.
    InvalidOption(&'static str),
    /// The texel density is far too high for the meshes: the atlas would
    /// exceed 2^30 texels.
    AtlasTooLarge {
        /// Padded texel area of everything to pack.
        texels: u64,
    },
    /// The progress callback of [`generate_with_progress`] returned `false`.
    Cancelled,
    /// A chart (or instance) doesn't fit in a page: with a fixed texel
    /// density, or in automatic mode when even very low densities don't help
    /// (padding alone is too large, e.g. instances whose scales differ by far
    /// more than the page size over the padding).
    ChartTooLarge {
        /// Padded chart width in texels.
        width: u32,
        /// Padded chart height in texels.
        height: u32,
        /// Page resolution.
        resolution: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidIndexCount { mesh, count } => {
                write!(f, "mesh {mesh}: index count {count} is not a multiple of 3")
            }
            Error::IndexOutOfRange { mesh, index, vertex_count } => {
                write!(f, "mesh {mesh}: index {index} out of range ({vertex_count} vertices)")
            }
            Error::NonFinitePosition { mesh, vertex } => {
                write!(f, "mesh {mesh}: vertex {vertex} has a non-finite position")
            }
            Error::NonFiniteUv { mesh, vertex } => {
                write!(f, "mesh {mesh}: vertex {vertex} has a non-finite uv")
            }
            Error::FaceDataLength { mesh, len, face_count } => {
                write!(f, "mesh {mesh}: per-face data has {len} entries for {face_count} triangles")
            }
            Error::UvCount { mesh, len, vertex_count } => {
                write!(f, "mesh {mesh}: {len} uvs for {vertex_count} vertices")
            }
            Error::InvalidScale { mesh } => write!(f, "mesh {mesh}: scale must be finite and > 0"),
            Error::InvalidOption(msg) => write!(f, "invalid option: {msg}"),
            Error::AtlasTooLarge { texels } => write!(f, "atlas would need {texels} texels; lower texels_per_unit"),
            Error::Cancelled => write!(f, "cancelled"),
            Error::ChartTooLarge { width, height, resolution } => write!(
                f,
                "chart of {width}x{height} texels doesn't fit a {resolution}x{resolution} page; lower texels_per_unit"
            ),
        }
    }
}

impl std::error::Error for Error {}

fn validate_options(o: &Options) -> Result<(), Error> {
    if !(o.max_chart_angle > 0.0 && o.max_chart_angle < 90.0) {
        return Err(Error::InvalidOption("max_chart_angle must be in (0, 90)"));
    }
    if !(o.max_area_distortion >= 1.0 && o.max_area_distortion.is_finite()) {
        return Err(Error::InvalidOption(
            "max_area_distortion must be finite and >= 1",
        ));
    }
    if let Some(t) = o.texels_per_unit {
        if !(t > 0.0 && t.is_finite()) {
            return Err(Error::InvalidOption(
                "texels_per_unit must be finite and > 0",
            ));
        }
    }
    if o.block_align && o.resolution % 4 != 0 {
        return Err(Error::InvalidOption(
            "resolution must be a multiple of 4 with block_align",
        ));
    }
    if o.resolution > 0 && o.padding.saturating_mul(2) >= o.resolution {
        return Err(Error::InvalidOption("padding too large for resolution"));
    }
    Ok(())
}

fn validate_mesh(mi: usize, m: &MeshInput<'_>) -> Result<(), Error> {
    if m.indices.len() % 3 != 0 {
        return Err(Error::InvalidIndexCount {
            mesh: mi,
            count: m.indices.len(),
        });
    }
    if let Some(&index) = m.indices.iter().find(|&&i| i as usize >= m.positions.len()) {
        return Err(Error::IndexOutOfRange {
            mesh: mi,
            index,
            vertex_count: m.positions.len(),
        });
    }
    if let Some(vertex) = m
        .positions
        .iter()
        .position(|p| p.iter().any(|c| !c.is_finite()))
    {
        return Err(Error::NonFinitePosition { mesh: mi, vertex });
    }
    let face_count = m.indices.len() / 3;
    let lens = [
        m.face_materials.map(<[_]>::len),
        m.face_ignore.map(<[_]>::len),
    ];
    if let Some(len) = lens.into_iter().flatten().find(|&l| l != face_count) {
        return Err(Error::FaceDataLength {
            mesh: mi,
            len,
            face_count,
        });
    }
    if let Some(uvs) = m.uvs {
        if uvs.len() != m.positions.len() {
            return Err(Error::UvCount {
                mesh: mi,
                len: uvs.len(),
                vertex_count: m.positions.len(),
            });
        }
        if let Some(vertex) = uvs.iter().position(|p| p.iter().any(|c| !c.is_finite())) {
            return Err(Error::NonFiniteUv { mesh: mi, vertex });
        }
    }
    if !(m.scale > 0.0 && m.scale.is_finite())
        || m.instances.iter().any(|k| !(*k > 0.0 && k.is_finite()))
    {
        return Err(Error::InvalidScale { mesh: mi });
    }
    Ok(())
}

/// Unwraps the meshes into a shared atlas.
///
/// Charts never span meshes, but all charts are packed together, so the
/// meshes can share one lightmap.
pub fn generate(meshes: &[MeshInput<'_>], options: &Options) -> Result<Atlas, Error> {
    run(meshes, options, &Ctx::new(None))
}

/// [`generate`], reporting progress to `callback` as it goes. Returning
/// `false` from the callback cancels the unwrap: work in flight stops promptly
/// and [`Error::Cancelled`] is returned.
///
/// The callback may run on worker threads, but never concurrently with
/// itself, and `done` never decreases within a stage.
///
/// ```
/// # let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
/// # let indices = [0, 1, 2];
/// let mesh = uvunwrap::MeshInput { positions: &positions, indices: &indices, ..Default::default() };
/// let atlas = uvunwrap::generate_with_progress(&[mesh], &Default::default(), |p| {
///     println!("{:?}: {}/{}", p.stage, p.done, p.total);
///     true // keep going
/// })?;
/// # Ok::<(), uvunwrap::Error>(())
/// ```
pub fn generate_with_progress(
    meshes: &[MeshInput<'_>],
    options: &Options,
    callback: impl Fn(Progress) -> bool + Sync,
) -> Result<Atlas, Error> {
    run(meshes, options, &Ctx::new(Some(&callback)))
}

fn run(meshes: &[MeshInput<'_>], options: &Options, ctx: &Ctx) -> Result<Atlas, Error> {
    validate_options(options)?;
    for (mi, m) in meshes.iter().enumerate() {
        validate_mesh(mi, m)?;
    }
    let max_angle = (options.max_chart_angle as f64).to_radians();
    let max_distortion = options.max_area_distortion as f64;

    let mut segmented = Vec::with_capacity(meshes.len());
    for m in meshes {
        let topo = Topology::build(m);
        let groups = if m.uvs.is_some() {
            segment::components(&topo)
        } else {
            segment::segment(&topo, max_angle)
        };
        segmented.push((topo, groups));
        ctx.advance(progress::Stage::Segmenting, 1, meshes.len())?;
    }
    let faces: usize = segmented.iter().flat_map(|(_, g)| g).map(Vec::len).sum();
    let per_mesh: Vec<(Topology, Vec<param::Chart>)> = segmented
        .into_iter()
        .zip(meshes)
        .map(|((topo, groups), m)| {
            let charts = match m.uvs {
                None => param::parameterize_all(&topo, groups, max_distortion, ctx, faces),
                Some(uvs) => {
                    let face_uvs: Vec<[[f64; 2]; 3]> = m
                        .indices
                        .chunks_exact(3)
                        .map(|t| [0, 1, 2].map(|c| uvs[t[c] as usize].map(f64::from)))
                        .collect();
                    param::from_islands(&topo, groups, &face_uvs, max_distortion, ctx, faces)
                }
            };
            (topo, charts)
        })
        .collect();
    if ctx.is_cancelled() {
        return Err(Error::Cancelled);
    }

    // Charts of meshes without instances are packed directly; meshes with
    // instances become tiles.
    let geoms: Vec<Vec<pack::ChartGeom>> = per_mesh
        .iter()
        .map(|(_, charts)| {
            charts
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
    let direct: Vec<pack::ChartGeom> = meshes
        .iter()
        .zip(&geoms)
        .filter(|(m, _)| m.instances.is_empty())
        .flat_map(|(_, g)| g.iter().copied())
        .collect();
    let scales: Vec<Vec<f64>> = meshes
        .iter()
        .map(|m| m.instances.iter().map(|&k| k as f64).collect())
        .collect();
    let tiles: Vec<pack::TileSpec> = meshes
        .iter()
        .zip(&geoms)
        .zip(&scales)
        .filter(|((m, _), _)| !m.instances.is_empty())
        .map(|((_, g), k)| pack::TileSpec {
            charts: g,
            scales: k,
        })
        .collect();
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
    let packed = pack::pack(
        &packer,
        options.texels_per_unit.map(f64::from),
        options.resolution,
    )?;
    let s = packed.texels_per_unit;
    let (pw, ph) = (packed.width as f64, packed.height as f64);
    let pad = options.padding as f64;

    let mut out_meshes = Vec::with_capacity(meshes.len());
    let mut chart_base = 0usize;
    let (mut direct_base, mut tile_index) = (0usize, 0usize);
    let mut covered = 0.0;
    for (m, (topo, charts)) in meshes.iter().zip(&per_mesh) {
        let tile = (!m.instances.is_empty()).then(|| {
            tile_index += 1;
            (
                &packed.tiles[tile_index - 1],
                &packed.instances[tile_index - 1],
            )
        });
        let nf = topo.faces.len();
        let mut face_chart = vec![NONE; nf];
        let mut face_slot = vec![0u32; nf];
        let mut first = vec![(NONE, 0u32); topo.positions.len()];
        let mut uvs: Vec<Vec<[f32; 2]>> = Vec::with_capacity(charts.len());
        for (ci, ch) in charts.iter().enumerate() {
            // Placement, density and the box UVs are relative to: the page, or
            // the tile's content box (inside its outer padding).
            let (pl, s, origin, extent) = match tile {
                None => {
                    let s = pack::chart_density(ch.size, s, options.max_chart_size);
                    covered += ch.area * s * s;
                    (packed.placements[direct_base + ci], s, [0.0, 0.0], [pw, ph])
                }
                Some((layout, _)) => {
                    let d = layout.density;
                    let cd = pack::chart_density(ch.size, d, layout.chart_limit);
                    // Instance `k` shows the layout scaled by `k * s / d`.
                    covered += ch.area
                        * m.instances
                            .iter()
                            .map(|&k| (cd * k as f64 * s / d) * (cd * k as f64 * s / d))
                            .sum::<f64>();
                    let extent = [layout.inner[0] * d, layout.inner[1] * d]
                        .map(|e| e.max(f64::MIN_POSITIVE));
                    (layout.placements[ci], cd, [pad, pad], extent)
                }
            };
            for (k, &f) in ch.faces.iter().enumerate() {
                face_chart[f as usize] = ci as u32;
                face_slot[f as usize] = k as u32;
            }
            for (l, &w) in ch.verts.iter().enumerate() {
                if first[w as usize].0 == NONE {
                    first[w as usize] = (ci as u32, l as u32);
                }
            }
            let (cw, chh) = mask::content_size(ch.size, s);
            uvs.push(
                ch.uvs
                    .iter()
                    .map(|&[x, y]| {
                        // Quarter-turn rotations keep orientation (no mirroring).
                        let [x, y] = mask::rotate_texel([x * s, y * s], pl.rotation, cw, chh);
                        let tx = pl.x as f64 + pad + x - origin[0];
                        let ty = pl.y as f64 + pad + y - origin[1];
                        [
                            (tx / extent[0]).clamp(0.0, 1.0) as f32,
                            (ty / extent[1]).clamp(0.0, 1.0) as f32,
                        ]
                    })
                    .collect(),
            );
        }

        let instances = match tile {
            None => Vec::new(),
            Some((layout, placements)) => m
                .instances
                .iter()
                .zip(placements)
                .map(|(&k, ip)| {
                    let (x, y) = (ip.x as f64 + pad, ip.y as f64 + pad);
                    let (w, h) = (
                        layout.inner[0] * k as f64 * s,
                        layout.inner[1] * k as f64 * s,
                    );
                    let uv = |x: f64, y: f64| {
                        [
                            (x / pw).clamp(0.0, 1.0) as f32,
                            (y / ph).clamp(0.0, 1.0) as f32,
                        ]
                    };
                    InstanceRect {
                        page: ip.page,
                        min: uv(x, y),
                        max: uv(x + w, y + h),
                    }
                })
                .collect(),
        };
        let mut out = OutputMesh {
            vertices: Vec::new(),
            indices: Vec::with_capacity(m.indices.len()),
            instances,
        };
        // (input vertex, chart, local vertex or NONE) -> output vertex.
        let mut lookup: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for (f, tri) in m.indices.chunks_exact(3).enumerate() {
            // Degenerate and ignored faces collapse onto one existing UV of one
            // of their corners, so they never stretch across the atlas.
            let anchor = if face_chart[f] == NONE {
                tri.iter()
                    .map(|&o| first[topo.welded[o as usize] as usize])
                    .find(|a| a.0 != NONE)
            } else {
                None
            };
            for (c, &orig) in tri.iter().enumerate() {
                let key = match anchor {
                    _ if face_chart[f] != NONE => {
                        let ci = face_chart[f];
                        (
                            orig,
                            ci,
                            charts[ci as usize].face_local[face_slot[f] as usize][c],
                        )
                    }
                    Some((ci, l)) => (orig, ci, l),
                    None => (orig, NONE, NONE),
                };
                let idx = *lookup.entry(key).or_insert_with(|| {
                    let (_, ci, l) = key;
                    let (uv, chart, page) = if ci == NONE {
                        ([0.0, 0.0], None, 0)
                    } else {
                        let page = if tile.is_some() {
                            0
                        } else {
                            packed.placements[direct_base + ci as usize].page
                        };
                        (
                            uvs[ci as usize][l as usize],
                            Some((chart_base + ci as usize) as u32),
                            page,
                        )
                    };
                    out.vertices.push(OutputVertex {
                        xref: orig,
                        uv,
                        chart,
                        page,
                    });
                    (out.vertices.len() - 1) as u32
                });
                out.indices.push(idx);
            }
        }
        chart_base += charts.len();
        if tile.is_none() {
            direct_base += charts.len();
        }
        out_meshes.push(out);
        ctx.advance(progress::Stage::Output, 1, meshes.len())?;
    }

    let total = pw * ph * packed.pages.max(1) as f64;
    Ok(Atlas {
        width: packed.width,
        height: packed.height,
        page_count: packed.pages,
        texels_per_unit: s as f32,
        chart_count: chart_base as u32,
        utilization: (covered / total).min(1.0) as f32,
        meshes: out_meshes,
    })
}
