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
//!    atlas pages, nesting into each other's concavities, or folded into rows
//!    for speed ([`Packing`]).
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
//! arithmetic and matches across platforms; CI compares output hashes on
//! Linux (x86-64, ARM64), macOS (ARM64), Windows (x86-64), wasm32-wasip1 and
//! wasm32-unknown-unknown.
//! With `direct-solver`, `faer` picks SIMD kernels at runtime, so output may
//! differ between CPU architectures (e.g. x86-64 vs ARM); disable it if you
//! cache unwraps or bake lightmaps on one machine and use them on another.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// `!(x > 0.0)` is deliberate throughout: unlike `x <= 0.0`, it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod error;
mod fold;
mod geom;
mod mask;
mod pack;
mod par;
mod param;
mod pipeline;
mod progress;
mod raster;
mod segment;
mod solver;
mod tile;
mod topology;
mod validate;

pub use error::Error;
use progress::Ctx;
pub use progress::{Progress, Stage};

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
    /// Optional polygon ids (one per triangle), for meshes triangulated from
    /// quads and n-gons, such as Blender's `loop_triangles` with their
    /// `polygon_index`. Triangles with the same id always share a chart, so
    /// each polygon corner gets exactly one UV, as UV maps that store one UV
    /// per polygon corner require. Ids are arbitrary; a polygon's triangles
    /// must have the same [`face_materials`](Self::face_materials) id and
    /// [`ignored_faces`](Self::ignored_faces) flag. The one exception is a
    /// polygon so bent that it can't be flattened on its own without folding
    /// over itself: its triangles are split between charts rather than
    /// overlapping. Default `None`: each triangle is its own polygon.
    pub polygons: Option<&'a [u32]>,
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
            polygons: None,
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
#[non_exhaustive]
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
    /// Fold charts into rows sorted by height and push each row up against
    /// the previous ones, compacting by each chart's per-row and per-column
    /// extent (TABI-style). Milliseconds even for thousands of charts, but
    /// can't nest charts into holes or deep concavities.
    Fast,
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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

/// Unwraps the meshes into a shared atlas.
///
/// Charts never span meshes, but all charts are packed together, so the
/// meshes can share one lightmap.
pub fn generate(meshes: &[MeshInput<'_>], options: &Options) -> Result<Atlas, Error> {
    pipeline::run(meshes, options, &Ctx::new(None))
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
    pipeline::run(meshes, options, &Ctx::new(Some(&callback)))
}
