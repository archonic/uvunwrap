//! `cargo bench` benchmarks of [`uvunwrap::generate`].
//!
//! Procedural meshes always run. Models from
//! <https://github.com/alecjacobson/common-3d-test-models> are included when
//! present in `models/` (`scripts/fetch-models.sh` downloads them; never
//! committed); missing ones are listed at startup. Each mesh is benchmarked with shape and fast
//! packing (auto density, 1024 page), fixed density (about one 1024 page),
//! and four instances; models with UVs also repack them. The iterative solver
//! is benchmarked with `cargo bench --no-default-features --features parallel`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::time::Duration;
use uvunwrap::{generate, Density, MeshInput, Options, Packing};

struct Mesh {
    name: String,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    uvs: Option<Vec<[f32; 2]>>,
}

impl Mesh {
    fn input(&self) -> MeshInput<'_> {
        MeshInput {
            positions: &self.positions,
            indices: &self.indices,
            ..Default::default()
        }
    }

    /// Fixed density that fills about one 1024 × 1024 page at 60% coverage.
    fn density(&self) -> f32 {
        let area: f32 = self
            .indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| self.positions[t[k] as usize]);
                let (u, v) = (
                    [0, 1, 2].map(|k| b[k] - a[k]),
                    [0, 1, 2].map(|k| c[k] - a[k]),
                );
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
            })
            .sum();
        (0.6 * 1024.0 * 1024.0 / area).sqrt()
    }
}

fn torus(nu: u32, nv: u32) -> Mesh {
    let mut positions = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let (a, b) = (i as f32 / nu as f32 * TAU, j as f32 / nv as f32 * TAU);
            let r = 1.0 + 0.4 * b.cos();
            positions.push([r * a.cos(), r * a.sin(), 0.4 * b.sin()]);
        }
    }
    let id = |i: u32, j: u32| (j % nv) * nu + (i % nu);
    let mut indices = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            indices.extend([
                id(i, j),
                id(i + 1, j),
                id(i + 1, j + 1),
                id(i, j),
                id(i + 1, j + 1),
                id(i, j + 1),
            ]);
        }
    }
    Mesh {
        name: format!("torus{}", indices.len() / 3),
        positions,
        indices,
        uvs: None,
    }
}

/// `n` random, unconnected triangles: one chart each.
fn soup(n: usize) -> Mesh {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let positions: Vec<[f32; 3]> = (0..n * 3).map(|_| [next(), next(), next()]).collect();
    Mesh {
        name: format!("soup{n}"),
        positions,
        indices: (0..n as u32 * 3).collect(),
        uvs: None,
    }
}

/// Loads an OBJ with one vertex per distinct `v/vt` corner, keeping the
/// `vt` coordinates if every corner has one.
fn load_obj(name: &str) -> Option<Mesh> {
    let text = std::fs::read_to_string(format!("{}/models/{name}.obj", env!("CARGO_MANIFEST_DIR")))
        .ok()?;
    let (mut v, mut vt) = (Vec::new(), Vec::new());
    let mut corners: HashMap<(i64, i64), u32> = HashMap::new();
    let (mut positions, mut uvs, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).map(|s| s.parse().unwrap_or(0.0)).collect();
                v.push([c[0], c[1], c[2]]);
            }
            Some("vt") => {
                let c: Vec<f32> = it.take(2).map(|s| s.parse().unwrap_or(0.0)).collect();
                vt.push([c[0], c.get(1).copied().unwrap_or(0.0)]);
            }
            Some("f") => {
                let resolve = |s: Option<&str>, n: usize| {
                    s.and_then(|s| s.parse::<i64>().ok())
                        .map(|i| if i < 0 { n as i64 + i } else { i - 1 })
                        .unwrap_or(-1)
                };
                let poly: Vec<u32> = it
                    .map(|t| {
                        let mut parts = t.split('/');
                        let key = (
                            resolve(parts.next(), v.len()),
                            resolve(parts.next(), vt.len()),
                        );
                        *corners.entry(key).or_insert_with(|| {
                            positions.push(v[key.0 as usize]);
                            uvs.push(usize::try_from(key.1).ok().and_then(|i| vt.get(i).copied()));
                            (positions.len() - 1) as u32
                        })
                    })
                    .collect();
                for k in 1..poly.len().saturating_sub(1) {
                    indices.extend([poly[0], poly[k], poly[k + 1]]);
                }
            }
            _ => {}
        }
    }
    let uvs = uvs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .filter(|u| !u.is_empty());
    Some(Mesh {
        name: name.to_string(),
        positions,
        indices,
        uvs,
    })
}

fn bench(c: &mut Criterion) {
    let mut meshes = vec![torus(64, 32), torus(256, 128), soup(2000)];
    for name in ["spot", "beetle", "bunny", "armadillo", "nefertiti", "igea"] {
        match load_obj(name) {
            Some(m) => meshes.push(m),
            None => eprintln!("skipping models/{name}.obj (not found)"),
        }
    }

    let mut group = c.benchmark_group("generate");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    for m in &meshes {
        let fixed = Options {
            density: Density::Fixed {
                texels_per_unit: m.density(),
                page_size: None,
            },
            ..Options::default()
        };
        let cases = [
            ("shape", [m.input()], Options::default()),
            (
                "fast",
                [m.input()],
                Options {
                    packing: Packing::Fast,
                    ..Options::default()
                },
            ),
            ("fixed", [m.input()], fixed),
            (
                "instances",
                [MeshInput {
                    instance_scales: &[1.0, 1.0, 0.5, 2.0],
                    ..m.input()
                }],
                Options::default(),
            ),
        ];
        for (label, input, opts) in &cases {
            group.bench_with_input(BenchmarkId::new(*label, &m.name), input, |b, input| {
                b.iter(|| generate(input, opts).unwrap())
            });
        }
        if let Some(uvs) = &m.uvs {
            let input = [MeshInput {
                existing_uvs: Some(uvs),
                ..m.input()
            }];
            group.bench_with_input(BenchmarkId::new("repack", &m.name), &input, |b, input| {
                b.iter(|| generate(input, &Options::default()).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
