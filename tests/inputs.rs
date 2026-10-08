//! Per-mesh inputs: authored seams, materials, ignored faces, scale and
//! existing UVs.

mod common;
use common::*;

/// Asserts that faces are on the same chart as face 0 exactly when `side`
/// matches face 0's.
fn assert_split_by(out: &OutputMesh, side: &[bool]) {
    for f in 0..side.len() {
        assert_eq!(face_chart(out, f) == face_chart(out, 0), side[f] == side[0], "face {f}");
    }
}

#[test]
fn authored_seams_become_chart_boundaries() {
    // A flat plane would be one chart, but its vertices are split down the
    // middle, as an exporter does for a UV seam.
    let m = split_plane();
    let atlas = run(&[&m], &Options::default());
    assert_eq!(atlas.chart_count, 2);
    assert_split_by(&atlas.meshes[0], &right_of(&m, 1.0));
}

#[test]
fn flat_shading_is_not_treated_as_seams() {
    // Every edge is split; seams carry no information, so the mesh unwraps as
    // if welded.
    let welded = torus(16, 8);
    let a = run(&[&welded], &Options::default());
    let b = run(&[&soup(&welded)], &Options::default());
    assert_eq!(a.chart_count, b.chart_count);
}

#[test]
fn charts_never_span_materials() {
    let m = plane(12);
    let materials: Vec<u32> = right_of(&m, 1.0).into_iter().map(u32::from).collect();
    let atlas = run_inputs(&[MeshInput { face_materials: Some(&materials), ..m.input() }], &Options::default());
    assert_eq!(atlas.chart_count, 2);
    assert_split_by(&atlas.meshes[0], &right_of(&m, 1.0));
}

#[test]
fn ignored_faces_get_no_uv_space() {
    // `check` asserts ignored faces have zero UV area and the rest are valid.
    let m = plane(12);
    let ignore = right_of(&m, 1.0);
    let atlas = run_inputs(&[MeshInput { face_ignore: Some(&ignore), ..m.input() }], &Options::default());
    assert_eq!(atlas.chart_count, 1);
    assert!(atlas.utilization > 0.8, "utilization {}", atlas.utilization);
}

#[test]
fn scale_sets_relative_texel_density() {
    let m = plane(4);
    let atlas = run_inputs(&[m.input(), MeshInput { scale: 2.0, ..m.input() }], &Options::default());
    let ratio = total_uv_area(&atlas.meshes[1]) / total_uv_area(&atlas.meshes[0]);
    assert!((ratio - 4.0).abs() < 1e-3, "area ratio {ratio}");
}

fn run_uvs(m: &Mesh, uvs: &[[f32; 2]]) -> Atlas {
    run_inputs(&[MeshInput { uvs: Some(uvs), ..m.input() }], &Options::default())
}

#[test]
fn input_uv_islands_become_charts() {
    // Continuous UVs across a vertex split (e.g. a hard edge): one island.
    let m = split_plane();
    let uvs: Vec<[f32; 2]> = m.positions.iter().map(|p| [p[0], p[1]]).collect();
    assert_eq!(run_uvs(&m, &uvs).chart_count, 1);
    // A UV seam on welded geometry: two islands.
    let m = soup(&plane(12));
    let right = right_of(&m, 1.0);
    let uvs: Vec<[f32; 2]> = m.positions.iter().enumerate().map(|(i, p)| [p[0] + if right[i / 3] { 5.0 } else { 0.0 }, p[1]]).collect();
    let atlas = run_uvs(&m, &uvs);
    assert_eq!(atlas.chart_count, 2);
    assert_split_by(&atlas.meshes[0], &right);
}

#[test]
fn input_uvs_are_normalized_mirrored_or_replaced() {
    let m = plane(8); // 2 x 1 units
    // Scale doesn't matter: islands are sized by surface area.
    let small: Vec<[f32; 2]> = m.positions.iter().map(|p| [p[0] * 0.1, p[1] * 0.1]).collect();
    let big: Vec<[f32; 2]> = m.positions.iter().map(|p| [p[0] * 10.0, p[1] * 10.0]).collect();
    let both = run_inputs(&[MeshInput { uvs: Some(&small), ..m.input() }, MeshInput { uvs: Some(&big), ..m.input() }], &Options::default());
    let ratio = total_uv_area(&both.meshes[0]) / total_uv_area(&both.meshes[1]);
    assert!((ratio - 1.0).abs() < 1e-3, "area ratio {ratio}");
    // Mirrored, degenerate and folded (self-overlapping) islands all come out
    // valid (`check`: positive area, no overlaps).
    let mirrored: Vec<[f32; 2]> = m.positions.iter().map(|p| [-p[0], p[1]]).collect();
    assert_eq!(run_uvs(&m, &mirrored).chart_count, 1);
    run_uvs(&m, &vec![[0.0, 0.0]; m.positions.len()]);
    let folded: Vec<[f32; 2]> = m.positions.iter().map(|p| [(p[0] - 1.0).abs(), p[1]]).collect();
    // The flat plane re-unwraps as one (unfolded) chart.
    assert_eq!(run_uvs(&m, &folded).chart_count, 1);
    // -0.0 and 0.0 are the same UV: the split plane's seam copies carry -0.0
    // where the originals carry 0.0, and it is still one island.
    let split = split_plane();
    let originals = plane(12).positions.len();
    let signed: Vec<[f32; 2]> =
        split.positions.iter().enumerate().map(|(i, p)| [if i >= originals { -(p[0] - 1.0) } else { p[0] - 1.0 }, p[1]]).collect();
    assert_eq!(run_uvs(&split, &signed).chart_count, 1);
}

#[test]
fn input_errors() {
    let m = plane(1);
    let o = Options::default();
    let short = [false];
    let bad_len = MeshInput { face_ignore: Some(&short), ..m.input() };
    assert!(matches!(generate(&[bad_len], &o), Err(Error::FaceDataLength { len: 1, face_count: 2, .. })));
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let bad_scale = MeshInput { scale, ..m.input() };
        assert!(matches!(generate(&[bad_scale], &o), Err(Error::InvalidScale { mesh: 0 })));
    }
    let short_uvs = [[0.0f32; 2]; 3];
    let bad = MeshInput { uvs: Some(&short_uvs), ..m.input() };
    assert!(matches!(generate(&[bad], &o), Err(Error::UvCount { len: 3, .. })));
    let mut nan = vec![[0.0f32; 2]; m.positions.len()];
    nan[2] = [f32::NAN, 0.0];
    let bad = MeshInput { uvs: Some(&nan), ..m.input() };
    assert!(matches!(generate(&[bad], &o), Err(Error::NonFiniteUv { vertex: 2, .. })));
}
