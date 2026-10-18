//! Hostile inputs: tiny and non-manifold meshes, extreme coordinates, slivers,
//! and seeded random meshes with random options. Every result goes through
//! `check`.

mod common;
use common::*;

fn mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    Mesh { positions, indices }
}

#[test]
fn tiny_and_non_manifold_meshes() {
    let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let cases = [
        // A single triangle; two disjoint ones.
        mesh(tri.to_vec(), vec![0, 1, 2]),
        mesh(
            vec![
                tri[0],
                tri[1],
                tri[2],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 1.0],
            ],
            vec![0, 1, 2, 3, 4, 5],
        ),
        // Three faces on one edge (non-manifold fan).
        mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            vec![0, 1, 2, 1, 0, 3, 0, 1, 4],
        ),
        // The same triangle twice, and once reversed.
        mesh(tri.to_vec(), vec![0, 1, 2, 0, 1, 2, 0, 2, 1]),
        // A T-junction: the right edge of the left quad is split by a vertex
        // the left quad doesn't use.
        mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            vec![0, 1, 2, 0, 2, 3, 1, 5, 4, 4, 5, 6, 4, 6, 2],
        ),
    ];
    for m in &cases {
        run(&[m], &Options::default());
    }
}

#[test]
fn extreme_coordinates() {
    let base = torus(24, 12);
    for (scale, offset) in [(1e-6f32, 0.0f32), (1e6, 0.0), (1.0, 1e6), (1e-3, 1e3)] {
        let m = mesh(
            base.positions
                .iter()
                .map(|p| p.map(|c| c * scale + offset))
                .collect(),
            base.indices.clone(),
        );
        let atlas = run(&[&m], &Options::default());
        assert!(atlas.chart_count >= 2, "scale {scale}, offset {offset}");
    }
}

#[test]
fn slivers_and_degenerate_meshes() {
    // A fan of 300 thin triangles around the origin.
    let mut positions = vec![[0.0f32; 3]];
    let mut indices = Vec::new();
    for i in 0..=300u32 {
        let a = i as f32 * 1e-3;
        positions.push([a.cos(), a.sin(), 0.0]);
        if i > 0 {
            indices.extend([0, i, i + 1]);
        }
    }
    run(&[&mesh(positions, indices)], &Options::default());
    // Collinear and point-like meshes: nothing to chart, but valid output.
    let line = mesh(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![0, 1, 2],
    );
    let point = mesh(vec![[1.0, 1.0, 1.0]; 3], vec![0, 1, 2]);
    let atlas = run(&[&line, &point], &Options::default());
    assert_eq!(atlas.chart_count, 0);
    // Everything ignored.
    let m = plane(4);
    let all = vec![true; m.faces()];
    let atlas = run_inputs(
        &[MeshInput {
            ignored_faces: Some(&all),
            ..m.input()
        }],
        &Options::default(),
    );
    assert_eq!(atlas.chart_count, 0);
}

/// A random mesh: a jittered height field, a closed bumpy surface, or a soup
/// of random triangles.
fn random_mesh(rng: &mut Rng) -> Mesh {
    match rng.below(3) {
        0 => {
            let (n, amp) = (2 + rng.below(10) as u32, rng.range(0.0, 2.0));
            let jitter: Vec<f32> = (0..(n + 1) * (n + 1))
                .map(|_| rng.range(-0.3, 0.3))
                .collect();
            surface(n, n, false, false, |u, v| {
                let i = ((v * n as f32).round() * (n + 1) as f32 + (u * n as f32).round()) as usize;
                [
                    u + jitter[i] * 0.1 / n as f32,
                    v,
                    amp * (u * 7.0).sin() * (v * 5.0).cos() + jitter[i] * 0.2,
                ]
            })
        }
        1 => {
            let (bumps, depth) = (1.0 + rng.below(6) as f32, rng.range(0.0, 0.4));
            surface(
                8 + rng.below(16) as u32,
                4 + rng.below(8) as u32,
                true,
                false,
                move |u, v| {
                    let (a, b) = (u * std::f32::consts::TAU, (v - 0.5) * std::f32::consts::PI);
                    let r = 1.0 + depth * (a * bumps).sin() * (b * bumps).cos();
                    [r * b.cos() * a.cos(), r * b.cos() * a.sin(), r * b.sin()]
                },
            )
        }
        _ => {
            let n = 1 + rng.below(40) as usize;
            let positions: Vec<[f32; 3]> = (0..n * 3)
                .map(|_| {
                    [
                        rng.range(0.0, 1.0),
                        rng.range(0.0, 1.0),
                        rng.range(0.0, 1.0),
                    ]
                })
                .collect();
            Mesh {
                positions,
                indices: (0..n as u32 * 3).collect(),
            }
        }
    }
}

fn random_options(rng: &mut Rng) -> Options {
    let fixed = rng.coin();
    let block_align = rng.below(4) == 0;
    let resolution = [0, 64, 256, 1024][rng.below(4) as usize];
    Options {
        max_chart_angle: rng.range(20.0, 85.0),
        max_area_distortion: rng.range(1.2, 6.0),
        padding: rng.below(4) as u32,
        allow_rotation: rng.coin(),
        packing: if rng.coin() {
            Packing::Shape
        } else {
            Packing::Fast
        },
        density: match fixed.then(|| rng.range(4.0, 40.0)) {
            Some(texels_per_unit) => Density::Fixed {
                texels_per_unit,
                page_size: (resolution > 0).then_some(resolution),
            },
            None => Density::Auto {
                page_size: if resolution == 0 { 256 } else { resolution },
            },
        },
        max_chart_size: if rng.below(4) == 0 {
            8 + rng.below(32) as u32
        } else {
            0
        },
        block_align,
    }
}

#[test]
fn random_meshes_and_options() {
    for seed in 0..60 {
        let mut rng = Rng::new(seed);
        let meshes: Vec<Mesh> = (0..1 + rng.below(3))
            .map(|_| random_mesh(&mut rng))
            .collect();
        let scales: Vec<Vec<f32>> = meshes
            .iter()
            .map(|_| {
                if rng.below(3) == 0 {
                    vec![1.0, rng.range(0.5, 2.0)]
                } else {
                    vec![]
                }
            })
            .collect();
        let inputs: Vec<MeshInput> = meshes
            .iter()
            .zip(&scales)
            .map(|(m, k)| MeshInput {
                instance_scales: k,
                ..m.input()
            })
            .collect();
        let opts = random_options(&mut rng);
        match generate(&inputs, &opts) {
            Ok(atlas) => check(&atlas, &inputs, &opts),
            // Fixed density can legitimately be too high for the page.
            Err(Error::ChartTooLarge { .. })
                if matches!(
                    opts.density,
                    Density::Fixed {
                        page_size: Some(_),
                        ..
                    }
                ) => {}
            Err(e) => panic!("seed {seed}: {e:?} with {opts:?}"),
        }
    }
}
