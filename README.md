# uvunwrap

Pure-Rust automatic UV unwrapping: generates unique, non-overlapping texture
coordinates for lightmaps, texture baking and texture painting. Inspired by
[xatlas](https://github.com/jpcy/xatlas), with no C++ toolchain required, so it
builds for wasm and cross-compiles trivially.

![Atlases for a torus, Spot, Suzanne and the Stanford bunny](docs/atlases.png)

## Usage

```rust
let mesh = uvunwrap::MeshInput { positions: &positions, indices: &indices };
let atlas = uvunwrap::generate(&[mesh], &uvunwrap::Options::default())?;

for v in &atlas.meshes[0].vertices {
    // v.xref: original vertex index, v.uv: [0, 1] coordinate, v.page: atlas page
}
```

Output meshes keep the input triangles in order; vertices on chart seams are
duplicated, and `xref` maps each one back to its input vertex.

Try it on an OBJ file (writes an OBJ with `vt` coordinates plus an SVG per page):

```sh
cargo run --release --example unwrap -- model.obj out/model
```

## How it works

1. **Topology**: welds identical positions, drops triangles too thin for f32
   UVs, and connects faces only across manifold, consistently oriented edges.
2. **Segmentation**: grows charts from the largest faces, limited by
   `max_chart_angle` from the chart's average normal.
3. **Parameterization**: planar projection for flat charts, otherwise Least
   Squares Conformal Maps solved by sparse Cholesky. Results with flipped
   triangles, overlaps or area distortion above `max_area_distortion` are split
   and retried, so output is always valid.
4. **Packing**: charts are rotated to their minimum-area bounding rectangle and
   skyline-packed. Texel density is automatic (fit one page) or fixed (one
   growing page, or multiple fixed-size pages), with padding and 90° rotation.

## Features

| Feature | Default | Effect |
|---|---|---|
| `direct-solver` | yes | Sparse Cholesky via pure-Rust `faer`; about 10× faster on large charts. Without it, an iterative CGLS solver is used. |
| `parallel` | yes | Parameterizes charts in parallel with `rayon`. Disable for wasm32 without threads. |

With no default features the crate has zero dependencies.

## Performance (single core)

| Model | Triangles | Charts | Time | Utilization |
|---|---|---|---|---|
| Spot | 5.9K | 46 | 56 ms | 51% |
| Stanford bunny | 69K | 184 | 0.9 s | 44% |
| Armadillo | 100K | 866 | 1.9 s | 41% |

## Status and roadmap

Version 0.1: correct and tested, but behind xatlas on quality.

- **Shape-aware packing**: charts are packed by bounding rectangle, so holes and
  concave space are wasted. Rasterized chart-mask packing (as xatlas does) is
  the biggest planned improvement, from about 45% toward 70–80% utilization.
- **Smoother charts**: region growing produces jagged boundaries; costs that
  favor round, compact charts would reduce seams and splitting.
- **Speed**: most time goes to large charts that fail validation and are
  re-solved after splitting; better segmentation fixes both.
- **Integrations**: `glam` types and a small Bevy crate.

## License

MIT
