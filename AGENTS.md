# uvunwrap: agent instructions

Pure-Rust automatic UV unwrapper (xatlas-like): segmentation → LSCM
parameterization → atlas packing.

## Goal

Give Rust, and Bevy in particular, a native way to generate a fresh,
non-overlapping UV set for arbitrary triangle meshes, both at build time and at
runtime (including wasm). The only alternative today is `xatlas-rs`, which
binds stale C++. Bevy renders lightmaps from `ATTRIBUTE_UV_1`, with a
`Lightmap { image, uv_rect }` component so many meshes can share one atlas
texture. It has no unwrapper or baker of its own, so procedural, CSG, voxel and
imported meshes without a second UV set can't be lightmapped or baked. Lighting
bakers, texture painting, decals and AO/curvature bakes all need the same
input: a lightmap-quality UV set.

What that means for the crate (✅ done, ◐ partial, ☐ missing):

1. ✅ **Bake-quality UVs.** No overlaps or flips, every face inside [0,1],
   padding in texels at the target resolution, low area distortion, uniform
   texel density.
2. ✅ **Seam splitting a renderer can apply.** Output vertices carry `xref` to
   the *input* vertex, so callers can remap every other attribute (normals,
   tangents, UV_0, colors, joints/weights, custom attributes) and keep the
   existing hard edges and UV_0 seams.
3. ◐ **Multi-mesh atlases.** `generate` packs many meshes into shared pages
   (xatlas `AddMesh`). Missing: a per-mesh world scale or instance transform,
   so density stays consistent across a scene of transformed instances, and
   output that maps directly onto `Lightmap::uv_rect`.
4. ◐ **Respect authored structure.** Charts follow position-welded topology
   and creases. Missing: optional input normals and UV_0 as seam hints (xatlas
   `normalSeamWeight`/`textureSeamWeight`), per-face material groups that
   charts never cross, and per-face ignore.
5. ✅ **Runtime-friendly.** Fast, no global state, `Send`, zero dependencies
   and single-threaded under `--no-default-features` for wasm. ☐ Progress
   reporting and cancellation for large meshes (xatlas `ProgressCallback`), so
   jobs run cleanly on Bevy's `AsyncComputeTaskPool`.
6. ◐ **Robust on real assets.** Degenerate faces, triangle soup and
   non-manifold input must never panic and must still receive valid UVs. Keep
   adding real-world models to the tests.
7. ☐ **Repack existing UVs** (xatlas `AddUvMesh`): pack authored UV charts
   into an atlas without re-segmenting, for example to combine several meshes'
   UV_0 into one lightmap.
8. ☐ **Packing extras that bakers use:** maximum chart size, 4×4 block
   alignment for BC-compressed lightmaps, and an optional chart-ID / coverage
   raster for bake dilation (xatlas `createImage`).
9. ☐ **Bevy integration crate** (kept separate so this crate stays
   engine-agnostic): `Mesh` → `Mesh` with UV_1 (or UV_0) and every attribute
   remapped, `u16`/`u32` indices, tangents regenerated when present, `Lightmap`
   `uv_rect` helpers for shared atlases, and use from an asset processor for
   build-time unwrapping.

Non-goals: lighting bakes themselves, matching xatlas output exactly, quad or
polygon output, and interactive UV editing.

Maintainer constraint: keep the crate focused and publishable. Engine-specific
code belongs in the integration crate, and new features need a concrete
Bevy or baking use case.

## Layout

- `src/lib.rs`: public API (`generate`, `MeshInput`, `Options`, `Atlas`,
  `OutputMesh`, `OutputVertex`, `Error`) and output assembly (seam vertex
  duplication, `xref` back to input vertices, degenerate faces collapsed onto
  one existing UV).
- `src/topology.rs`: welding by exact position, degeneracy (altitude <
  1e-6 × bbox diagonal), manifold + consistently oriented adjacency only.
- `src/segment.rs`: xatlas-style growth over planar regions: cost = normal
  deviation (crease-preserving smoothed normals, hard `max_chart_angle` cone)
  + straightness + roundness + crease penalty; then merge small enclosed
  charts and smooth single-face teeth off boundaries.
- `src/param.rs`: planar projection if flat, else LSCM; validate (no flips, no
  overlaps via grid + SAT, area ratio ≤ `max_area_distortion`); on failure
  split chart in two (graph-distance Voronoi, seeded at the overlapping pair
  when validation found an overlap) and retry, halves in parallel.
  Charts ≥ 1000 faces with |total curvature| > 3 rad are split without
  solving. Finalize: scale to 3D area, rotate to min-area rect.
- `src/solver.rs`: sparse rows, `least_squares` = faer supernodal sparse
  Cholesky on normal equations (AᵀA assembled directly in CSC;
  `direct-solver` feature) with Jacobi-preconditioned CGLS fallback.
- `src/mask.rs`: conservative chart rasterization + padding dilation into
  per-row span masks (4 rotations derived exactly), and the page bitset with
  per-row free runs used for exact bottom-left placement search.
- `src/pack.rs`: `Packing::Shape` (masks, default) or `Packing::Rect`
  (skyline); auto density (bisection to fit one page, seeded by the rect
  result, cropped to used height), fixed density (single growing page or
  multiple fixed pages).
- `tests/meshes.rs`: plane, cube, sphere, torus, cylinder, soup, degenerates,
  errors; checks topology preservation, UV range, no flips, no overlaps.
- `examples/unwrap.rs`: OBJ in → OBJ with `vt` + SVG per page.
- `benches/generate.rs`: criterion (`cargo bench`), procedural meshes plus
  `models/{spot,beetle,bunny,armadillo,nefertiti}.obj` when present.

## Commands

- `cargo test` and `cargo test --no-default-features` (both must pass)
- `cargo run --release --example unwrap -- [--rect] model.obj out/model`
- Test models: https://github.com/alecjacobson/common-3d-test-models (`data/*.obj`),
  downloaded into the gitignored `models/` dir (bunny is `stanford-bunny.obj`).
- Render atlas SVGs: `rsvg-convert -w 800 out/x_page0.svg -o out/x.png`
  (`qlmanage` hangs).

## Conventions

- `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`.
- Zero dependencies with `--no-default-features`; must stay wasm-friendly.
- Never commit downloaded models or `out/`.

## Baseline (release, 1024 page, padding 2, Apple M5 10 cores)

| Model | Triangles | Charts | 1 core | Multi-core | Utilization |
|---|---|---|---|---|---|
| Spot | 5.9K | 38 | 34 ms | 19 ms | 68% |
| Stanford bunny | 69K | 58 | 0.19 s | 82 ms | 65% |
| Armadillo | 100K | 257 | 0.40 s | 0.16 s | 60% |
| Nefertiti | 100K | 456 | 0.74 s | 0.40 s | 64% |
| Igea | 269K | 83 | 0.96 s | 0.48 s | 66% |

Utilization is true surface area / atlas area (stricter than xatlas, which
counts rasterized texels). xatlas reference (default pack options, 10
threads): 55–71% utilization, 13–14 s on bunny/armadillo.

Benchmark with best-of-N single-core runs (`RAYON_NUM_THREADS=1`); the dev
machine is often loaded, so multi-core timings are noisy.

## Notes from tuning (don't retry blindly)

- Whether a density fits is not monotonic (greedy placement), so utilization
  jitters ±2–3 points with tiny density changes. Parallel multi-point density
  search wasn't faster and found worse densities; reverted.
- Re-segmenting failed charts with a tighter cone shatters them (armadillo
  1600 charts); kept Voronoi bisection.
- Splitting distortion failures between min/max area-ratio triangles: no
  clear gain. Splitting pinch vertices per fan: no change in failures.
- Curvature predictors: signed total K works for skipping (|K| > 3, ≥ 1000
  faces); absolute K is useless on scans (noise).
- faer parallel factorization (`Par::rayon`) for big systems: no gain.
- Remaining costs: LSCM solves (successful + failed) dominate; igea's
  critical path is one 98K-face chart that fails and is re-solved in halves.
  Packing ≈ 9 sequential density trials.

## Next steps (priority order)

1. Optional `glam` feature; a separate small Bevy integration crate.
2. CI (tests in both feature configs, wasm32 build), then publish 0.1 to crates.io.
3. Refresh `docs/atlases.png` (still shows the old rect-packed atlases).
