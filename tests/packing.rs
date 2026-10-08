//! Packing options: fixed density, pages, chart size limits, block alignment.

mod common;
use common::*;

#[test]
fn fixed_texel_density() {
    let m = plane(4); // 2 x 1 units
    let opts = Options {
        texels_per_unit: Some(64.0),
        resolution: 0,
        padding: 1,
        ..Options::default()
    };
    let atlas = run(&[&m], &opts);
    assert_eq!(atlas.texels_per_unit, 64.0);
    // One 128 x 64 texel chart plus 1 texel of padding on each side.
    assert_eq!(
        (atlas.width.max(atlas.height), atlas.width.min(atlas.height)),
        (130, 66)
    );
    assert_eq!(atlas.page_count, 1);
}

#[test]
fn fixed_density_spills_onto_pages() {
    let m = cube();
    let opts = Options {
        texels_per_unit: Some(30.0),
        resolution: 64,
        padding: 1,
        ..Options::default()
    };
    let atlas = run(&[&m], &opts);
    assert!(atlas.page_count > 1);
}

#[test]
fn chart_too_large() {
    let m = plane(2);
    let opts = Options {
        texels_per_unit: Some(1000.0),
        resolution: 64,
        ..Options::default()
    };
    assert!(matches!(
        generate(&[m.input()], &opts),
        Err(Error::ChartTooLarge { .. })
    ));
}

#[test]
fn max_chart_size_scales_down_big_charts_only() {
    let (big, small) = (plane(4), plane(1)); // both 2 x 1 units
    let small_input = MeshInput {
        scale: 0.1,
        ..small.input()
    };
    let opts = Options {
        texels_per_unit: Some(64.0),
        resolution: 0,
        padding: 1,
        max_chart_size: 32,
        ..Options::default()
    };
    let atlas = run_inputs(&[big.input(), small_input], &opts);
    let (lo, hi) = texel_bounds(&atlas, &atlas.meshes[0]);
    let side = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    assert!(side <= 32.0 && side > 31.0, "big chart side {side}");
    // The small chart (0.2 x 0.1 units, 12.8 texels) keeps the full density.
    let (lo, hi) = texel_bounds(&atlas, &atlas.meshes[1]);
    let side = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    assert!((side - 12.8).abs() < 1e-3, "small chart side {side}");
}

/// Whether a UV triangle (in texels) overlaps the open 4x4 block at `(bx, by)`.
fn touches_block(t: &[[f64; 2]; 3], bx: u32, by: u32) -> bool {
    let (x0, y0) = (bx as f64 * 4.0, by as f64 * 4.0);
    let (x1, y1) = (x0 + 4.0, y0 + 4.0);
    // Inset by a hair: triangles may touch a block edge without entering it.
    let inside =
        |p: [f64; 2]| p[0] > x0 + 1e-6 && p[0] < x1 - 1e-6 && p[1] > y0 + 1e-6 && p[1] < y1 - 1e-6;
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    t.iter().any(|&p| inside(p))
        || corners.iter().any(|&c| {
            tri_area([t[0], t[1], c]) > 0.0
                && tri_area([t[1], t[2], c]) > 0.0
                && tri_area([t[2], t[0], c]) > 0.0
        })
        || (0..3).any(|i| {
            (0..4).any(|j| segments_cross(t[i], t[(i + 1) % 3], corners[j], corners[(j + 1) % 4]))
        })
}

/// Whether two segments properly cross.
fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let side = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| tri_area([p, q, r]);
    side(a, b, c) * side(a, b, d) < 0.0 && side(c, d, a) * side(c, d, b) < 0.0
}

#[test]
fn block_align_gives_each_chart_its_own_blocks() {
    let (a, b) = (torus(32, 16), cube());
    for packing in [Packing::Shape, Packing::Rect] {
        for opts in [
            Options {
                block_align: true,
                padding: 0,
                resolution: 256,
                packing,
                ..Options::default()
            },
            Options {
                block_align: true,
                padding: 1,
                texels_per_unit: Some(40.0),
                resolution: 0,
                packing,
                ..Options::default()
            },
        ] {
            let atlas = run(&[&a, &b], &opts);
            assert_eq!((atlas.width % 4, atlas.height % 4), (0, 0));
            let mut owner = std::collections::HashMap::new();
            for out in &atlas.meshes {
                for t in out.indices.chunks_exact(3) {
                    let vs = [0, 1, 2].map(|c| out.vertices[t[c] as usize]);
                    let p = vs.map(|v| {
                        [
                            v.uv[0] as f64 * atlas.width as f64,
                            v.uv[1] as f64 * atlas.height as f64,
                        ]
                    });
                    let Some(chart) = vs[0].chart else { continue };
                    let lo = [0, 1].map(|k| {
                        (p.iter().map(|q| q[k]).fold(f64::MAX, f64::min) / 4.0).floor() as u32
                    });
                    let hi = [0, 1].map(|k| {
                        (p.iter().map(|q| q[k]).fold(f64::MIN, f64::max) / 4.0).floor() as u32
                    });
                    for by in lo[1]..=hi[1] {
                        for bx in lo[0]..=hi[0] {
                            if touches_block(&p, bx, by) {
                                let o = *owner.entry((vs[0].page, bx, by)).or_insert(chart);
                                assert_eq!(o, chart, "block ({bx}, {by}) shared by charts {o} and {chart} ({packing:?})");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn absurd_density_or_padding_errors_instead_of_allocating() {
    let m = plane(1);
    let dense = Options {
        texels_per_unit: Some(1e6),
        resolution: 0,
        ..Options::default()
    };
    assert!(matches!(
        generate(&[m.input()], &dense),
        Err(Error::AtlasTooLarge { .. })
    ));
    let paged = Options {
        texels_per_unit: Some(1e6),
        resolution: 1024,
        ..Options::default()
    };
    assert!(matches!(
        generate(&[m.input()], &paged),
        Err(Error::ChartTooLarge { .. })
    ));
    let padded = Options {
        texels_per_unit: Some(1.0),
        resolution: 0,
        padding: u32::MAX / 2,
        ..Options::default()
    };
    assert!(matches!(
        generate(&[m.input()], &padded),
        Err(Error::AtlasTooLarge { .. })
    ));
}

#[test]
fn every_packer_configuration_is_valid() {
    // `check` covers overlaps, padding (including 0) and page edges.
    let (a, b, c) = (torus(32, 16), cube(), sphere(24, 12));
    for packing in [Packing::Shape, Packing::Rect] {
        for allow_rotation in [true, false] {
            for padding in [0, 1, 3] {
                for (texels_per_unit, resolution) in
                    [(None, 256), (Some(30.0), 0), (Some(30.0), 128)]
                {
                    let opts = Options {
                        packing,
                        allow_rotation,
                        padding,
                        texels_per_unit,
                        resolution,
                        ..Options::default()
                    };
                    run(&[&a, &b, &c], &opts);
                }
            }
        }
    }
}

#[test]
fn invalid_options() {
    let o = Options::default();
    for bad in [
        Options {
            max_chart_angle: 0.0,
            ..o.clone()
        },
        Options {
            max_chart_angle: 95.0,
            ..o.clone()
        },
        Options {
            max_area_distortion: 0.5,
            ..o.clone()
        },
        Options {
            max_area_distortion: f32::NAN,
            ..o.clone()
        },
        Options {
            texels_per_unit: Some(0.0),
            ..o.clone()
        },
        Options {
            texels_per_unit: Some(f32::INFINITY),
            ..o.clone()
        },
        Options {
            padding: 512,
            resolution: 1024,
            ..o.clone()
        },
        Options {
            block_align: true,
            resolution: 1022,
            ..o.clone()
        },
    ] {
        assert!(
            matches!(generate(&[], &bad), Err(Error::InvalidOption(_))),
            "{bad:?}"
        );
    }
}
