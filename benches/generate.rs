//! `cargo bench` benchmarks of [`uvunwrap::generate`].
//!
//! Procedural meshes always run. Models from
//! <https://github.com/alecjacobson/common-3d-test-models> are included when
//! present in `models/` (e.g. `models/bunny.obj`; never committed).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f32::consts::TAU;
use std::time::Duration;
use uvunwrap::{generate, MeshInput, Options, Packing};

struct Mesh {
    name: String,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
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
            indices.extend([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    Mesh { name: format!("torus{}", indices.len() / 3), positions, indices }
}

fn load_obj(name: &str) -> Option<Mesh> {
    let text = std::fs::read_to_string(format!("{}/models/{name}.obj", env!("CARGO_MANIFEST_DIR"))).ok()?;
    let (mut positions, mut indices) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).map(|s| s.parse().unwrap_or(0.0)).collect();
                positions.push([c[0], c[1], c[2]]);
            }
            Some("f") => {
                let n = positions.len() as i64;
                let poly: Vec<u32> = it
                    .filter_map(|t| t.split('/').next()?.parse::<i64>().ok())
                    .map(|i| if i < 0 { n + i } else { i - 1 } as u32)
                    .collect();
                for k in 1..poly.len().saturating_sub(1) {
                    indices.extend([poly[0], poly[k], poly[k + 1]]);
                }
            }
            _ => {}
        }
    }
    Some(Mesh { name: name.to_string(), positions, indices })
}

fn bench(c: &mut Criterion) {
    let mut meshes = vec![torus(64, 32), torus(256, 128)];
    meshes.extend(["spot", "beetle", "bunny", "armadillo", "nefertiti"].iter().filter_map(|n| load_obj(n)));

    let mut group = c.benchmark_group("generate");
    group.sample_size(10).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    for m in &meshes {
        let input = [MeshInput { positions: &m.positions, indices: &m.indices, ..Default::default() }];
        for (label, packing) in [("shape", Packing::Shape), ("rect", Packing::Rect)] {
            let opts = Options { packing, ..Options::default() };
            group.bench_with_input(BenchmarkId::new(label, &m.name), &input, |b, input| {
                b.iter(|| generate(input, &opts).unwrap())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
