//! Polygons (`MeshInput::polygons`): a polygon's triangles always share a
//! chart, so each polygon corner gets exactly one UV. `check` verifies this
//! whenever polygons are given.

mod common;

use common::*;

fn with_polygons<'a>(m: &'a Mesh, polygons: &'a [u32]) -> MeshInput<'a> {
    MeshInput {
        polygons: Some(polygons),
        ..m.input()
    }
}

#[test]
fn quad_meshes_keep_their_quads_whole() {
    // The sphere's pole quads each contain a degenerate triangle.
    for m in [
        torus(32, 16),
        sphere(24, 12),
        cylinder(24),
        cube(),
        plane(8),
    ] {
        let polygons = quads(&m);
        for opts in [
            Options::default(),
            Options {
                packing: Packing::Fast,
                ..Options::default()
            },
            Options {
                block_align: true,
                density: Density::Fixed {
                    texels_per_unit: 40.0,
                    page_size: None,
                },
                ..Options::default()
            },
        ] {
            run_inputs(&[with_polygons(&m, &polygons)], &opts);
        }
    }
}

#[test]
fn bent_quads_stay_whole_when_charts_are_split() {
    // Non-planar quads on a bumpy grid, with a distortion limit tight enough
    // that charts fail validation and are split over and over.
    let m = surface(24, 24, false, false, |u, v| {
        let wobble = ((u * 97.0).sin() * (v * 89.0).cos()) * 0.02;
        [u, v, 0.4 * (u * 9.0).sin() * (v * 7.0).cos() + wobble]
    });
    let polygons = quads(&m);
    let opts = Options {
        max_area_distortion: 1.1,
        ..Options::default()
    };
    let atlas = run_inputs(&[with_polygons(&m, &polygons)], &opts);
    assert!(atlas.chart_count > 10, "{} charts", atlas.chart_count);
}

/// A capped cylinder: quads around the side and one n-gon per cap, each cap
/// fan-triangulated from its first rim vertex (as Blender triangulates).
fn capped_cylinder(n: u32) -> (Mesh, Vec<u32>) {
    let mut m = cylinder(n);
    let mut polygons = quads(&m);
    let side = polygons.len() as u32 / 2;
    let rows = 5; // cylinder(n) has 4 rings of quads: 5 rings of vertices
    for (cap, ring) in [(0, 0), (1, rows - 1)] {
        let rim = |i: u32| ring * n + i;
        for i in 1..n - 1 {
            // The bottom cap faces down, the top one up.
            let t = if cap == 0 {
                [rim(0), rim(i + 1), rim(i)]
            } else {
                [rim(0), rim(i), rim(i + 1)]
            };
            m.indices.extend(t);
            polygons.push(side + cap);
        }
    }
    (m, polygons)
}

#[test]
fn n_gons_stay_whole() {
    let (m, polygons) = capped_cylinder(20);
    let atlas = run_inputs(&[with_polygons(&m, &polygons)], &Options::default());
    // Each 20-gon cap is one chart: its 18 triangles share it.
    let out = &atlas.meshes[0];
    let side_faces = 20 * 4 * 2;
    for cap in 0..2 {
        let faces = side_faces + cap * 18..side_faces + (cap + 1) * 18;
        let charts: Vec<Option<u32>> = faces
            .map(|f| out.vertices[out.indices[3 * f] as usize].chart)
            .collect();
        assert!(charts.iter().all(|c| c.is_some() && *c == charts[0]));
    }
}

#[test]
fn degenerate_triangles_keep_their_polygons_corner_uvs() {
    // A pentagon with three collinear corners: its fan triangulation from
    // corner 0 starts with a zero-area sliver, which shares corners 0 and 2
    // with the rest of the polygon.
    let m = Mesh {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        indices: vec![0, 1, 2, 0, 2, 3, 0, 3, 4],
    };
    run_inputs(&[with_polygons(&m, &[7, 7, 7])], &Options::default());
}

#[test]
fn a_polygon_that_folds_over_itself_is_split() {
    // A fan polygon whose corners wind 540° around its apex, rising so it
    // doesn't intersect itself in 3D: no flattening avoids overlap, so the
    // documented exception applies and its triangles are split between
    // charts instead of overlapping.
    let n = 18;
    let mut positions = vec![[0.0, 0.0, 0.0]];
    positions.extend((0..=n).map(|i| {
        let a = i as f32 / n as f32 * 3.0 * std::f32::consts::PI;
        [a.cos(), a.sin(), i as f32 * 0.1]
    }));
    let m = Mesh {
        positions,
        indices: (1..=n).flat_map(|i| [0, i, i + 1]).collect(),
    };
    let polygons = vec![0; n as usize];
    let opts = Options::default();
    let atlas = generate(&[with_polygons(&m, &polygons)], &opts).unwrap();
    // Valid UVs, checked without the one-chart-per-polygon guarantee.
    check(&atlas, &[m.input()], &opts);
    assert!(atlas.chart_count >= 2, "{} charts", atlas.chart_count);
}

#[test]
fn single_triangle_polygons_change_nothing() {
    let m = torus(24, 12);
    let own: Vec<u32> = (0..m.faces() as u32).collect();
    let opts = Options::default();
    assert_eq!(
        generate(&[with_polygons(&m, &own)], &opts),
        generate(&[m.input()], &opts)
    );
}

#[test]
fn polygons_must_agree_on_per_face_data() {
    let m = plane(2);
    let polygons = quads(&m);
    let opts = Options::default();
    let mut materials = vec![0; m.faces()];
    materials[1] = 1; // the second triangle of quad 0
    let mut ignored = vec![false; m.faces()];
    ignored[2] = true; // the first triangle of quad 1
    let mixed = |input: MeshInput| generate(&[input], &opts).unwrap_err();
    assert_eq!(
        mixed(MeshInput {
            face_materials: Some(&materials),
            ..with_polygons(&m, &polygons)
        }),
        Error::MixedPolygon {
            mesh: 0,
            polygon: 0,
            field: "face_materials"
        }
    );
    assert_eq!(
        mixed(MeshInput {
            ignored_faces: Some(&ignored),
            ..with_polygons(&m, &polygons)
        }),
        Error::MixedPolygon {
            mesh: 0,
            polygon: 1,
            field: "ignored_faces"
        }
    );
    assert_eq!(
        mixed(with_polygons(&m, &polygons[1..])),
        Error::FaceDataLength {
            mesh: 0,
            field: "polygons",
            len: m.faces() - 1,
            face_count: m.faces()
        }
    );
}
