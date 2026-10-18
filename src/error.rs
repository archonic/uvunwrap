//! The crate's error type.

use std::fmt;

/// Errors returned by [`generate`](crate::generate).
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
    /// A vertex of [`MeshInput::existing_uvs`](crate::MeshInput::existing_uvs) contains NaN or infinity.
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
    /// [`MeshInput::existing_uvs`](crate::MeshInput::existing_uvs) doesn't have one entry per vertex.
    UvCount {
        /// Mesh position in the input slice.
        mesh: usize,
        /// Number of UVs.
        len: usize,
        /// Number of vertices in the mesh.
        vertex_count: usize,
    },
    /// [`MeshInput::scale`](crate::MeshInput::scale) or an instance scale is not finite and positive.
    InvalidScale {
        /// Mesh position in the input slice.
        mesh: usize,
        /// The offending entry of [`MeshInput::instance_scales`](crate::MeshInput::instance_scales), or `None`
        /// for [`MeshInput::scale`](crate::MeshInput::scale).
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
    /// The progress callback of [`generate_with_progress`](crate::generate_with_progress) returned `false`.
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
