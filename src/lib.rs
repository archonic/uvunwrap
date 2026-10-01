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
//! let mesh = uvunwrap::MeshInput { positions: &positions, indices: &indices };
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

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod geom;
mod mask;
mod pack;
mod param;
mod segment;
mod solver;
mod topology;

use std::collections::HashMap;
use std::fmt;
use topology::{Topology, NONE};

/// An indexed triangle mesh to unwrap.
#[derive(Clone, Copy, Debug)]
pub struct MeshInput<'a> {
    /// Vertex positions.
    pub positions: &'a [[f32; 3]],
    /// Triangle list indices into `positions`, counter-clockwise front faces.
    pub indices: &'a [u32],
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
    pub allow_rotation: bool,
    /// Packing algorithm. Default [`Packing::Shape`].
    pub packing: Packing,
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
        }
    }
}

/// The generated atlas.
#[derive(Clone, Debug)]
pub struct Atlas {
    /// Page width in texels.
    pub width: u32,
    /// Page height in texels.
    pub height: u32,
    /// Number of atlas pages (see [`OutputVertex::page`]).
    pub page_count: u32,
    /// Texel density that was used.
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
}

/// A vertex of an [`OutputMesh`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputVertex {
    /// Index of the input vertex this vertex was created from.
    pub xref: u32,
    /// Texture coordinate in `[0, 1]`, relative to its page. `v` grows with the
    /// texel row (row 0 at `v = 0`); flip it if your convention differs.
    pub uv: [f32; 2],
    /// Global chart id, or `None` for vertices only used by degenerate triangles.
    pub chart: Option<u32>,
    /// Atlas page this vertex belongs to.
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
    /// An option is out of range.
    InvalidOption(&'static str),
    /// With a fixed texel density, a chart doesn't fit in a page.
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
            Error::InvalidOption(msg) => write!(f, "invalid option: {msg}"),
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
        return Err(Error::InvalidOption("max_area_distortion must be finite and >= 1"));
    }
    if let Some(t) = o.texels_per_unit {
        if !(t > 0.0 && t.is_finite()) {
            return Err(Error::InvalidOption("texels_per_unit must be finite and > 0"));
        }
    }
    if o.resolution > 0 && o.padding.saturating_mul(2) >= o.resolution {
        return Err(Error::InvalidOption("padding too large for resolution"));
    }
    Ok(())
}

fn validate_mesh(mi: usize, m: &MeshInput<'_>) -> Result<(), Error> {
    if m.indices.len() % 3 != 0 {
        return Err(Error::InvalidIndexCount { mesh: mi, count: m.indices.len() });
    }
    if let Some(&index) = m.indices.iter().find(|&&i| i as usize >= m.positions.len()) {
        return Err(Error::IndexOutOfRange { mesh: mi, index, vertex_count: m.positions.len() });
    }
    if let Some(vertex) = m.positions.iter().position(|p| p.iter().any(|c| !c.is_finite())) {
        return Err(Error::NonFinitePosition { mesh: mi, vertex });
    }
    Ok(())
}

/// Unwraps the meshes into a shared atlas.
///
/// Charts never span meshes, but all charts are packed together, so the
/// meshes can share one lightmap.
pub fn generate(meshes: &[MeshInput<'_>], options: &Options) -> Result<Atlas, Error> {
    validate_options(options)?;
    for (mi, m) in meshes.iter().enumerate() {
        validate_mesh(mi, m)?;
    }
    let max_angle = (options.max_chart_angle as f64).to_radians();
    let max_distortion = options.max_area_distortion as f64;

    let per_mesh: Vec<(Topology, Vec<param::Chart>)> = meshes
        .iter()
        .map(|m| {
            let topo = Topology::build(m.positions, m.indices);
            let groups = segment::segment(&topo, max_angle);
            let charts = param::parameterize_all(&topo, groups, max_distortion);
            (topo, charts)
        })
        .collect();

    let geoms: Vec<pack::ChartGeom> = per_mesh
        .iter()
        .flat_map(|(_, c)| c.iter())
        .map(|c| pack::ChartGeom { uvs: &c.uvs, tris: &c.face_local, size: c.size, area: c.area })
        .collect();
    let packer = pack::Packer {
        charts: &geoms,
        padding: options.padding,
        allow_rotation: options.allow_rotation,
        method: options.packing,
    };
    let packed = pack::pack(&packer, options.texels_per_unit.map(f64::from), options.resolution)?;
    let s = packed.texels_per_unit;
    let (pw, ph) = (packed.width as f64, packed.height as f64);
    let pad = options.padding as f64;

    let mut out_meshes = Vec::with_capacity(meshes.len());
    let mut chart_base = 0usize;
    let mut covered = 0.0;
    for (m, (topo, charts)) in meshes.iter().zip(&per_mesh) {
        let nf = topo.faces.len();
        let mut face_chart = vec![NONE; nf];
        let mut face_slot = vec![0u32; nf];
        let mut first = vec![(NONE, 0u32); topo.positions.len()];
        let mut uvs: Vec<Vec<[f32; 2]>> = Vec::with_capacity(charts.len());
        for (ci, ch) in charts.iter().enumerate() {
            let pl = packed.placements[chart_base + ci];
            covered += ch.area * s * s;
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
                        let tx = pl.x as f64 + pad + x;
                        let ty = pl.y as f64 + pad + y;
                        [(tx / pw).clamp(0.0, 1.0) as f32, (ty / ph).clamp(0.0, 1.0) as f32]
                    })
                    .collect(),
            );
        }

        let mut out = OutputMesh { vertices: Vec::new(), indices: Vec::with_capacity(m.indices.len()) };
        // (input vertex, chart, local vertex or NONE) -> output vertex.
        let mut lookup: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for (f, tri) in m.indices.chunks_exact(3).enumerate() {
            // Degenerate faces collapse onto one existing UV of one of their
            // corners, so they never stretch across the atlas.
            let anchor = if face_chart[f] == NONE {
                tri.iter().map(|&o| first[topo.welded[o as usize] as usize]).find(|a| a.0 != NONE)
            } else {
                None
            };
            for (c, &orig) in tri.iter().enumerate() {
                let key = match anchor {
                    _ if face_chart[f] != NONE => {
                        let ci = face_chart[f];
                        (orig, ci, charts[ci as usize].face_local[face_slot[f] as usize][c])
                    }
                    Some((ci, l)) => (orig, ci, l),
                    None => (orig, NONE, NONE),
                };
                let idx = *lookup.entry(key).or_insert_with(|| {
                    let (_, ci, l) = key;
                    let (uv, chart, page) = if ci == NONE {
                        ([0.0, 0.0], None, 0)
                    } else {
                        let g = chart_base + ci as usize;
                        (uvs[ci as usize][l as usize], Some(g as u32), packed.placements[g].page)
                    };
                    out.vertices.push(OutputVertex { xref: orig, uv, chart, page });
                    (out.vertices.len() - 1) as u32
                });
                out.indices.push(idx);
            }
        }
        chart_base += charts.len();
        out_meshes.push(out);
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
