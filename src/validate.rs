//! Input validation: every documented requirement on [`Options`] and
//! [`MeshInput`], reported as an [`Error`] naming the offending option,
//! array or instance.

use crate::{Density, Error, MeshInput, Options};

/// Checks the options against their documented ranges.
pub(crate) fn options(o: &Options) -> Result<(), Error> {
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
    if !(o.max_chart_area >= 0.0 && o.max_chart_area.is_finite()) {
        return invalid("max_chart_area", "finite and >= 0");
    }
    if !(o.max_boundary_length >= 0.0 && o.max_boundary_length.is_finite()) {
        return invalid("max_boundary_length", "finite and >= 0");
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

/// Checks mesh `mi` against its documented requirements.
pub(crate) fn mesh(mi: usize, m: &MeshInput<'_>) -> Result<(), Error> {
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
        ("seams", m.seams.map(<[_]>::len)),
        ("polygons", m.polygons.map(<[_]>::len)),
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
    if let Some(polygons) = m.polygons {
        // Triangles grouped by polygon: neighbours in this order with the
        // same id belong to one polygon and must agree.
        let mut order: Vec<usize> = (0..face_count).collect();
        order.sort_unstable_by_key(|&f| (polygons[f], f));
        for pair in order.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if polygons[a] != polygons[b] {
                continue;
            }
            let mixed = |field| Error::MixedPolygon {
                mesh: mi,
                polygon: polygons[a],
                field,
            };
            if m.face_materials.is_some_and(|fm| fm[a] != fm[b]) {
                return Err(mixed("face_materials"));
            }
            if m.ignored_faces.is_some_and(|ig| ig[a] != ig[b]) {
                return Err(mixed("ignored_faces"));
            }
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
