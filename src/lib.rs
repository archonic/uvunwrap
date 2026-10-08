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
//! let mesh = uvunwrap::MeshInput::new(&positions, &indices);
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
mod par;
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
/// Build one with [`MeshInput::new`] and set optional fields with struct
/// update syntax. Fields added in future versions will always have defaults,
/// so this pattern keeps compiling:
///
/// ```
/// # let (positions, indices) = (Vec::<[f32; 3]>::new(), Vec::<u32>::new());
/// let mesh = uvunwrap::MeshInput { scale: 2.0, ..uvunwrap::MeshInput::new(&positions, &indices) };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MeshInput<'a> {
    /// Vertex positions.
    pub positions: &'a [[f32; 3]],
    /// Triangle list: three indices into `positions` per triangle. Winding
    /// needn't be consistent; it only decides which way each output triangle
    /// faces in UV space (see [`OutputVertex::uv`]).
    pub indices: &'a [u32],
    /// Optional per-face material ids (one per triangle). Charts never span
    /// faces with different ids.
    pub face_materials: Option<&'a [u32]>,
    /// Optional per-face flags (one per triangle). Flagged faces get no chart:
    /// their corners collapse onto an existing UV, as for degenerate faces.
    pub ignored_faces: Option<&'a [bool]>,
    /// Factor from mesh units to world units, so meshes drawn at different
    /// scales share one texel density. Also usable as a per-mesh importance
    /// factor. Must be finite and `> 0`. Default `1`.
    pub scale: f32,
    /// One scale per instance of a mesh drawn several times, each multiplying
    /// `scale`. Empty (the default): the mesh's charts are packed into the
    /// atlas directly and its UVs are atlas coordinates. Otherwise the mesh is
    /// unwrapped once, its UVs cover `[0, 1]` of its own layout, and each
    /// instance gets its own unrotated rectangle on one page holding that
    /// layout at the instance's scale ([`OutputMesh::instances`]). This is how
    /// engines lightmap meshes drawn several times (e.g. Bevy's
    /// `Lightmap::uv_rect`), and with a single instance it keeps a mesh on one
    /// page. Each must be finite and `> 0`.
    ///
    /// All instances share one layout, padded for the smallest instance, so
    /// an instance `n` times larger gets `n` times the padding: keep scales
    /// within a small range (split very different ones into separate meshes).
    /// [`Options::max_chart_size`] applies to every instance.
    pub instance_scales: &'a [f32],
    /// Optional existing texture coordinates (one per vertex) to repack
    /// instead of unwrapping from scratch: each UV island (faces connected
    /// through identical UVs) becomes one chart, scaled to its surface area so
    /// texel density stays uniform, and rotated freely. Mirrored islands are
    /// flipped; islands with flipped or degenerate triangles or self-overlap
    /// are unwrapped normally. Default `None`.
    pub existing_uvs: Option<&'a [[f32; 2]]>,
}

impl<'a> MeshInput<'a> {
    /// A mesh with the given positions and triangle indices and default
    /// optional fields.
    pub fn new(positions: &'a [[f32; 3]], indices: &'a [u32]) -> Self {
        MeshInput {
            positions,
            indices,
            face_materials: None,
            ignored_faces: None,
            scale: 1.0,
            instance_scales: &[],
            existing_uvs: None,
        }
    }
}

impl Default for MeshInput<'_> {
    /// An empty mesh with default optional fields.
    fn default() -> Self {
        MeshInput::new(&[], &[])
    }
}

/// How texel density and page size are chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Density {
    /// The highest density at which everything fits one `page_size ×
    /// page_size` page. The page is then cropped to the rows actually used,
    /// so [`Atlas::height`] may be less than `page_size`. If the charts can't
    /// share one page at any density (very many charts, or padding too large
    /// for the page), several pages are used at a lower density.
    Auto {
        /// Page width (and maximum height) in texels; `> 2 × padding`.
        page_size: u32,
    },
    /// A fixed density, for consistent texel size across separately
    /// unwrapped atlases (how engines size lightmaps).
    Fixed {
        /// Texels per world unit. Must be finite and `> 0`.
        texels_per_unit: f32,
        /// `None`: one page that grows to fit (cropped to its content).
        /// `Some(size)`: as many `size × size` pages as needed; fails with
        /// [`Error::ChartTooLarge`] if a chart can't fit one.
        page_size: Option<u32>,
    },
}

impl Default for Density {
    /// `Auto { page_size: 1024 }`.
    fn default() -> Self {
        Density::Auto { page_size: 1024 }
    }
}

/// Unwrapping options. Fields added in future versions will always have
/// defaults, so `Options { padding: 4, ..Default::default() }` keeps
/// compiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Maximum angle in degrees between a face normal and its chart's average
    /// normal. Lower values give flatter charts but more seams. Must be in
    /// `(0, 90)`. Default `70`.
    pub max_chart_angle: f32,
    /// Maximum allowed ratio between a triangle's relative UV area and its
    /// relative surface area (in either direction). Charts exceeding it are
    /// split. Must be `>= 1`. Default `4`.
    pub max_area_distortion: f32,
    /// Texel density and page size. Default [`Density::Auto`] with 1024-texel
    /// pages.
    pub density: Density,
    /// Empty texels around each chart, to avoid bleeding when filtering or
    /// generating mipmaps: charts are at least `2 × padding` texels apart and
    /// `padding` texels from the page edges. Default `2`.
    pub padding: u32,
    /// Allow rotating charts by multiples of 90° during packing. Rotations
    /// never mirror. Default `true`.
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
    /// charts. Page sizes become multiples of 4; a `page_size` must be one.
    /// Default `false`.
    pub block_align: bool,
}

/// How charts are arranged in the atlas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
            density: Density::default(),
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
    /// Page height in texels. With a single cropped page (automatic density,
    /// or fixed density without a page size) this is the height actually
    /// used, which may differ from the width.
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
    /// One rectangle per entry of [`MeshInput::instance_scales`] (empty if
    /// none).
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
    /// Index of the input vertex this vertex was created from (xatlas'
    /// name): copy that vertex's other attributes (normals, colors, ...).
    pub xref: u32,
    /// Texture coordinate in `[0, 1]` page coordinates, or, for meshes with
    /// [`MeshInput::instance_scales`], in `[0, 1]` of each instance's
    /// rectangle.
    ///
    /// `(0, 0)` is the corner of texel row 0, column 0, and `v` grows with the
    /// row: the top-left of the image in wgpu, Vulkan, Direct3D and Metal (so
    /// Bevy), the bottom-left in OpenGL, where you may want `1 - v`. Every
    /// output triangle has positive signed area in `(u, v)` in its input index
    /// order: counter-clockwise when `v` points up, clockwise as seen in an
    /// image with row 0 at the top.
    pub uv: [f32; 2],
    /// Global chart id, or `None` for vertices only used by degenerate or
    /// ignored triangles that have no charted corner.
    pub chart: Option<u32>,
    /// Atlas page of this vertex, or `None` if it has no chart or its mesh
    /// has instances (see [`InstanceRect::page`]).
    pub page: Option<u32>,
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
    /// A vertex of [`MeshInput::existing_uvs`] contains NaN or infinity.
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
        /// Which array: `"face_materials"` or `"ignored_faces"`.
        field: &'static str,
        /// Length of the array.
        len: usize,
        /// Number of triangles in the mesh.
        face_count: usize,
    },
    /// [`MeshInput::existing_uvs`] doesn't have one entry per vertex.
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
        /// The offending entry of [`MeshInput::instance_scales`], or `None`
        /// for [`MeshInput::scale`].
        instance: Option<usize>,
    },
    /// An option is out of range.
    InvalidOption {
        /// The option, e.g. `"padding"` or `"density.page_size"`.
        option: &'static str,
        /// What it must satisfy.
        requirement: &'static str,
    },
    /// The texel density is far too high for the meshes: the atlas would
    /// exceed 2^30 texels.
    AtlasTooLarge {
        /// Padded texel area of everything to pack.
        texels: u64,
    },
    /// The progress callback of [`generate_with_progress`] returned `false`.
    Cancelled,
    /// A chart (or instance) doesn't fit in a page: with a fixed density and
    /// page size, or with automatic density when even very low densities don't
    /// help (padding alone is too large, e.g. instances whose scales differ by
    /// far more than the page size over the padding).
    ChartTooLarge {
        /// Padded chart width in texels.
        width: u32,
        /// Padded chart height in texels.
        height: u32,
        /// Page size in texels.
        page_size: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidIndexCount { mesh, count } => {
                write!(f, "mesh {mesh}: index count {count} is not a multiple of 3")
            }
            Error::IndexOutOfRange {
                mesh,
                index,
                vertex_count,
            } => write!(
                f,
                "mesh {mesh}: index {index} out of range ({vertex_count} vertices)"
            ),
            Error::NonFinitePosition { mesh, vertex } => {
                write!(f, "mesh {mesh}: vertex {vertex} has a non-finite position")
            }
            Error::NonFiniteUv { mesh, vertex } => {
                write!(
                    f,
                    "mesh {mesh}: vertex {vertex} has a non-finite existing uv"
                )
            }
            Error::FaceDataLength {
                mesh,
                field,
                len,
                face_count,
            } => write!(
                f,
                "mesh {mesh}: {field} has {len} entries for {face_count} triangles"
            ),
            Error::UvCount {
                mesh,
                len,
                vertex_count,
            } => write!(
                f,
                "mesh {mesh}: existing_uvs has {len} entries for {vertex_count} vertices"
            ),
            Error::InvalidScale {
                mesh,
                instance: None,
            } => write!(f, "mesh {mesh}: scale must be finite and > 0"),
            Error::InvalidScale {
                mesh,
                instance: Some(i),
            } => write!(
                f,
                "mesh {mesh}: instance_scales[{i}] must be finite and > 0"
            ),
            Error::InvalidOption {
                option,
                requirement,
            } => write!(f, "invalid option {option}: must be {requirement}"),
            Error::AtlasTooLarge { texels } => {
                write!(f, "atlas would need {texels} texels; lower texels_per_unit")
            }
            Error::Cancelled => write!(f, "cancelled"),
            Error::ChartTooLarge {
                width,
                height,
                page_size,
            } => write!(
                f,
                "chart of {width}x{height} texels doesn't fit a {page_size}x{page_size} page; \
                 lower texels_per_unit, set max_chart_size, or use larger pages"
            ),
        }
    }
}

impl std::error::Error for Error {}

fn validate_options(o: &Options) -> Result<(), Error> {
    let invalid = |option, requirement| {
        Err(Error::InvalidOption {
            option,
            requirement,
        })
    };
    if !(o.max_chart_angle > 0.0 && o.max_chart_angle < 90.0) {
        return invalid("max_chart_angle", "in (0, 90)");
    }
    if !(o.max_area_distortion >= 1.0 && o.max_area_distortion.is_finite()) {
        return invalid("max_area_distortion", "finite and >= 1");
    }
    let page_size = match o.density {
        Density::Auto { page_size } => Some(page_size),
        Density::Fixed {
            texels_per_unit,
            page_size,
        } => {
            if !(texels_per_unit > 0.0 && texels_per_unit.is_finite()) {
                return invalid("density.texels_per_unit", "finite and > 0");
            }
            page_size
        }
    };
    if let Some(size) = page_size {
        if size <= o.padding.saturating_mul(2) {
            return invalid("density.page_size", "greater than 2 × padding");
        }
        if o.block_align && size % 4 != 0 {
            return invalid("density.page_size", "a multiple of 4 with block_align");
        }
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
    let per_face = [
        ("face_materials", m.face_materials.map(<[_]>::len)),
        ("ignored_faces", m.ignored_faces.map(<[_]>::len)),
    ];
    for (field, len) in per_face {
        if let Some(len) = len.filter(|&l| l != face_count) {
            return Err(Error::FaceDataLength {
                mesh: mi,
                field,
                len,
                face_count,
            });
        }
    }
    if let Some(uvs) = m.existing_uvs {
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
    let valid = |k: f32| k > 0.0 && k.is_finite();
    if !valid(m.scale) {
        return Err(Error::InvalidScale {
            mesh: mi,
            instance: None,
        });
    }
    if let Some(i) = m.instance_scales.iter().position(|&k| !valid(k)) {
        return Err(Error::InvalidScale {
            mesh: mi,
            instance: Some(i),
        });
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
/// let mesh = uvunwrap::MeshInput::new(&positions, &indices);
/// let atlas = uvunwrap::generate_with_progress(&[mesh], &Default::default(), |p| {
///     println!("{:?}: {} of {:?}", p.stage, p.done, p.total);
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
        let groups = if m.existing_uvs.is_some() {
            segment::components(&topo)
        } else {
            segment::segment(&topo, max_angle)
        };
        segmented.push((topo, groups));
        ctx.advance(progress::Stage::Segmenting, 1, Some(meshes.len()))?;
    }
    let faces: usize = segmented.iter().flat_map(|(_, g)| g).map(Vec::len).sum();
    let per_mesh: Vec<(Topology, Vec<param::Chart>)> = segmented
        .into_iter()
        .zip(meshes)
        .map(|((topo, groups), m)| {
            let face_uvs: Option<Vec<[[f64; 2]; 3]>> = m.existing_uvs.map(|uvs| {
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
        .filter(|(m, _)| m.instance_scales.is_empty())
        .flat_map(|(_, g)| g.iter().copied())
        .collect();
    let scales: Vec<Vec<f64>> = meshes
        .iter()
        .map(|m| m.instance_scales.iter().map(|&k| k as f64).collect())
        .collect();
    let tiles: Vec<pack::TileSpec> = meshes
        .iter()
        .zip(&geoms)
        .zip(&scales)
        .filter(|((m, _), _)| !m.instance_scales.is_empty())
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
    // The packer's form: a fixed density or none, and a page size or 0 for
    // one unbounded page.
    let (density, page_size) = match options.density {
        Density::Auto { page_size } => (None, page_size),
        Density::Fixed {
            texels_per_unit,
            page_size,
        } => (Some(texels_per_unit as f64), page_size.unwrap_or(0)),
    };
    let packed = pack::pack(&packer, density, page_size)?;
    let s = packed.texels_per_unit;
    let (pw, ph) = (packed.width as f64, packed.height as f64);
    let pad = options.padding as f64;

    let mut out_meshes = Vec::with_capacity(meshes.len());
    let mut chart_base = 0usize;
    let (mut direct_base, mut tile_index) = (0usize, 0usize);
    let mut covered = 0.0;
    for (m, (topo, charts)) in meshes.iter().zip(&per_mesh) {
        let tile = (!m.instance_scales.is_empty()).then(|| {
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
                        * m.instance_scales
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
                .instance_scales
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
                        ([0.0, 0.0], None, None)
                    } else {
                        let page = match tile {
                            Some(_) => None,
                            None => Some(packed.placements[direct_base + ci as usize].page),
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
        ctx.advance(progress::Stage::Output, 1, Some(meshes.len()))?;
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
