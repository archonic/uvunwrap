# uvunwrap: agent instructions

Pure-Rust automatic UV unwrapper (xatlas-like): segmentation → LSCM
parameterization → atlas packing. Bevy is the primary consumer: lightmaps
from `ATTRIBUTE_UV_1` (`Lightmap { image, uv_rect }`), second UV sets for
procedural meshes, texture painting and decals. xatlas's feature set as
engines use it (Godot's lightmap UV2 unwrap: fixed density, padding,
`maxChartSize`, `blockAlign`, `fixWinding`) is the parity bar; xatlas's
known problems (cross-platform nondeterminism, scheduler deadlocks, slow
large imports) are what to avoid.

Plans, status, benchmark results and experiment history go in `TODO.md`
(gitignored, local), never in this file or anything packaged.

## Invariants (every change must keep these)

- **Valid output for any input.** No overlaps or flipped triangles, every
  face inside [0, 1], padding in texels between charts and to page edges.
  Degenerate faces, triangle soup and non-manifold input must never panic
  and must still get valid UVs. `tests/common/mod.rs::check` verifies all
  of this; use it in new tests.
- **Callers can apply the output.** Same triangles in the same order;
  output vertices carry `xref` to the input vertex.
- **Consistent UV winding** (xatlas `fixWinding`): every output triangle has
  positive signed UV area in its input index order. So charts are never
  mirrored: rotations by multiples of 90° only.
- **Authored structure is respected.** Edges where the input splits
  vertices (hard edges, UV seams) are chart boundaries; `face_materials`
  and `ignored_faces` are honoured.
- **Determinism.** Bit-identical output for any thread count. No platform
  math in `src/`: never call `sin`/`cos`/`atan2`/`acos`/`powf`/`exp`/`ln`
  etc. (use `geom::cos`/`geom::angle`; `sqrt` is fine, it is exactly
  rounded). Without `direct-solver` output must match across platforms (CI
  compares the torus hash on Linux x86-64/ARM64, macOS ARM64, Windows,
  wasm32-wasip1 and wasm32-unknown-unknown); the example's torus must not
  use libm either. `direct-solver` stays the default for speed; it may
  differ across CPU architectures.
- **No global state or scheduler**: rayon only, safe to call from any
  thread any number of times. Zero dependencies and single-threaded with
  `--no-default-features` (wasm). `#![forbid(unsafe_code)]`,
  `#![warn(missing_docs)]`.
- **Scope.** Keep the crate focused and publishable: engine glue belongs in
  separate crates, and new features need a concrete engine or baking use
  case. Non-goals: lighting bakes, matching xatlas output exactly, quad or
  polygon output, interactive UV editing.

## Layout

- `src/lib.rs`: the public API only: crate docs, `generate`/
  `generate_with_progress`, `MeshInput::new` plus optional
  `face_materials`/`ignored_faces`/`polygons`/`scale`/`instance_scales`/
  `existing_uvs`,
  `Options` with `Density::{Auto, Fixed}` and `Packing::{Shape, Fast}`,
  `Atlas`, `OutputMesh`, `OutputVertex` (`page: Option`), `InstanceRect`.
- `src/error.rs`: `Error` (re-exported), naming the offending
  option/array/instance. `src/validate.rs`: every documented input
  requirement.
- `src/pipeline.rs`: `run` is validate → `unwrap_meshes` (segment +
  parameterize, meshes in parallel) → `pack_meshes` (each mesh's `Slot`:
  `Direct` or `Tiled`) → `assemble_mesh` per mesh in parallel
  (`place_chart`, `instance_rects`, `split_seams`: seam vertex duplication,
  `xref` back to input vertices, degenerate/ignored faces collapsed onto one
  existing UV, except that a degenerate face's corners take its polygon's UV
  for that vertex, so each polygon corner keeps one UV).
- `src/par.rs`: the only place that switches on the `parallel` feature
  (`map`, `map_ref`, `flat_map`, `join`; order-preserving).
- `src/geom.rs`: f64 vector helpers, convex hull, min-area rectangle, triangle
  overlap, and deterministic `cos`/`angle` (no platform math anywhere).
- `src/topology.rs`: welding by exact position (scaled by `MeshInput::scale`),
  exclusion (degenerate: altitude < 1e-6 × bbox diagonal, or ignored),
  manifold + consistently oriented adjacency within one material only. Edges
  where the input splits vertices (authored seams) are cut too, unless more
  than half of all adjacent edges are split (flat shading); with input UVs,
  edges where the UVs differ are cut instead (always). Polygon links
  (`polygon`: each face's lowest-numbered face of its polygon, `siblings`:
  the others) from `MeshInput::polygons`; every face is its own polygon
  without them.
- `src/segment.rs`: xatlas-style growth over planar regions (a polygon's
  faces always share a region, so charts never split a polygon): cost = normal
  deviation (crease-preserving smoothed normals, hard `max_chart_angle` cone)
  + straightness + roundness + crease penalty; then merge small enclosed
  charts and smooth single-face teeth off boundaries. `components` gives the
  UV islands of meshes with input UVs (no segmentation).
- `src/param.rs`: planar projection if flat, else LSCM; validate (no flips, no
  overlaps via grid + SAT, area ratio ≤ `max_area_distortion`); on failure
  split chart in two (graph-distance Voronoi, seeded at the overlapping pair
  when validation found an overlap; whole polygons go to one side, so only
  a single polygon that folds over itself is ever split) and retry, halves
  in parallel. A triangulated polygon has no interior vertices, so it
  always unfolds exactly unless it overlaps itself.
  Charts ≥ 1000 faces with |total curvature| > 3 rad are split without
  solving. Finalize: scale to 3D area, rotate to min-area rect. With input
  UVs, `parameterize` keeps each island (validated with unbounded
  distortion, mirrored ones flipped) and falls back to the normal path per
  island.
- `src/progress.rs`: `Stage`, `Progress`, and `Ctx` (callback behind a mutex
  so reports are serialized and monotonic, plus an `AtomicBool` that
  parameterization polls; `QUIET` for nested packing).
- `src/solver.rs`: sparse rows, `least_squares` = faer supernodal sparse
  Cholesky on normal equations (AᵀA assembled directly in CSC;
  `direct-solver` feature) with Jacobi-preconditioned CGLS fallback.
- `src/raster.rs`: shared by both packers: `rasterize_chart` (conservative
  rasterization + square padding dilation into a `Bits` bitmap) and texel
  geometry (`content_size`, `rotate_texel`).
- `src/mask.rs`: the shape packer: per-row span masks from the bitmap (4
  rotations derived exactly) and the page bitset with per-row free runs
  used for exact bottom-left placement search.
- `src/pack.rs`: `prepare` (sizes and tile layouts), then `rasterize` only
  once sizes pass the page and texel-budget checks (returns `Shapes`: masks
  or profiles), `place` (by `Shapes`), and one function per mode: `pack_auto` (bisection to fit one page;
  shape is seeded by the fast result with at most 8 back-off steps; cropped to used
  height, multi-page fallback), `pack_unbounded` (fixed density, 7 widths in
  parallel on one set of masks, smallest area), `pack_pages` (fixed density,
  fixed pages). Fast trials fold onto an unbounded page (same rows until
  one overflows), so each also gives the height needed, and the next trial
  jumps to `s · sqrt(page / used)` when that's inside the bracket. Per-chart density (`chart_density`) applies
  `max_chart_size`; `block_align` packs in 4×4 block units
  (`Mask::downsample`/`Profile::downsample` after rotating in texels,
  sizes rounded up) and scales placements back.
- `src/fold.rs`: `Packing::Fast`, TABI-style fold-and-push (`pack_folded`).
  `Profile` (`from_bits`, word-at-a-time): per-column top/bottom and
  per-row left/right extents of a padded bitmap, rotated exactly; `oriented` turns charts taller than wide,
  then 180° if that leaves more free space bottom/right (no mirroring).
  Items sorted by height fold into rows (`fold`: bounding boxes abutting, or
  compacted against a per-row frontier of all previous items in the row),
  are pushed up against the page frontline (`push`), and colliding pairs in
  a row are lowered to equal height until none collide (exact version of
  TABI Alg. 1). Per row: both directions × full width / knee concavity
  (`Knee`, Alg. 2) × compacted or not; more items wins, then lowest
  (max, sum) frontline. Beyond TABI: each placed row's `Pocket` (per column,
  old frontline to the row's item tops) is filled with the largest remaining
  items that fit (`Folder::fill`, `Pocket::fit`: exact, candidates are
  columns as deep as the item's tallest column). A row that overflows
  starts a new page.
- `src/tile.rs`: instanced meshes: `layout_tile` packs a tile's charts once
  (`pack_unbounded` at the smallest instance's density, capped so the
  largest instance fits the page); each instance becomes one unrotatable
  rectangle (`Mask::rect` or `Profile::rect`) among the packed items.
- `tests/`: `common/mod.rs` holds generators and `check`, which verifies
  every output guarantee (topology, positive UV area, no overlaps, 2×padding
  between charts and instances, padding to page edges, ignored faces
  collapsed, one UV per polygon corner when `polygons` is given;
  sweep-and-prune, so large meshes are fine). Topic files: `unwrap`,
  `inputs`, `polygons`, `packing`, `instances`, `progress`, `robustness`
  (hostile inputs and a seeded random mesh/options property test, half
  the seeds with bent quads as polygons).
- `python/`: Python bindings (workspace crate `uvunwrap-python`, not on
  crates.io; PyPI package and module `uvunwrap`, built with maturin, abi3
  wheels for Python ≥ 3.11). `src/lib.rs`: `Mesh` (inputs from any
  buffer-protocol object or flat/nested sequence, copied into an
  `Arc<MeshData>`), `generate` (GIL released) and `start` (a `Job` on a
  std thread: `progress`, `done`, `cancel()`, `result(timeout)`), outputs as
  `array.array`s including `corner_uvs` (UV per triangle corner, for
  per-loop UV maps such as Blender's), `UnwrapError`/`Cancelled`.
  `uvunwrap.pyi`: type stubs, shipped in the wheel. `tests/`: pytest.
- `examples/unwrap.rs`: OBJ in (one vertex per `v/vt/vn` corner, like an
  engine import; `--flat` splits every face) → OBJ with `vt` + SVG per page.
- `examples/torus_hash.rs` (+ `.mjs`): cdylib exporting the torus hash, for
  wasm32-unknown-unknown in CI (no stdout there); `examples/common/mod.rs`
  holds the torus and hash both examples share.
- `benches/generate.rs`: criterion (`cargo bench`; `-- --test` runs each case
  once): tori and a 2000-triangle soup plus
  `models/{spot,beetle,bunny,armadillo,nefertiti,igea}.obj` when present, each
  with shape, fast, fixed density and four instances, plus `repack` for
  models with UVs.

## Commands

- `cargo test` and `cargo test --no-default-features` (both must pass).
  CI (`.github/workflows/ci.yml`) also runs fmt, clippy `-D warnings` in both
  feature configs, `cargo doc` with `-D warnings` (public and private items), MSRV 1.85 (faer's
  `generativity` needs it), a wasm32 build and the cross-platform hash
  comparison; run the same locally before committing.
- `cargo run --release --example unwrap -- [--fast] model.obj out/model`
  (no model: the built-in torus, printing the output hash).
- `scripts/fetch-models.sh`: downloads the test models into the gitignored
  `models/` dir (pinned commit, SHA-256 checked; upstream
  `stanford-bunny.obj` is saved as `bunny.obj`).
- `cargo bench` (`-- --test` runs each case once).
- Python bindings: in a virtualenv with `maturin pytest numpy`, from
  `python/`: `maturin develop --release` then `pytest -q tests`; clippy with
  `cargo clippy -p uvunwrap-python --all-targets -- -D warnings`. CI builds
  the wheel and runs the tests on Linux/macOS/Windows with Python 3.11 and
  3.13 (the version Blender bundles). To try it in Blender, unzip a wheel
  from `maturin build --release` and put that directory on `sys.path`
  (`blender -b --python script.py`).
- Python releases: `.github/workflows/release-python.yml` builds abi3 wheels
  (Linux x86-64/ARM64, macOS ARM/Intel, Windows x64) and the sdist, tests
  the native ones, and uploads to PyPI by trusted publishing (environment
  `pypi`, no token) when a GitHub release is published; run by hand it only
  builds unless `publish` is checked. Bump `python/Cargo.toml`'s version
  with the crate's.
- Render atlas SVGs: `rsvg-convert -w 800 out/x_page0.svg -o out/x.png`
  (`qlmanage` hangs).

## Benchmarking

- Compare against the README's performance table (1024 page, padding 2).
  Utilization is true surface area / atlas area (stricter than xatlas,
  which counts rasterized texels).
- Use best-of-N single-core runs (`RAYON_NUM_THREADS=1`); the dev machine
  is often loaded, so multi-core timings are noisy. Interleave runs of the
  two binaries being compared (a `git worktree` of the baseline commit
  works well). Check the load average before trusting absolute numbers.
- A change that adds no hot code can still move timings ~4% by changing how
  code is split into codegen units; to tell whether a slowdown is real,
  compare both builds with `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1`.
- Whether a density fits is not monotonic (greedy placement), so
  utilization jitters ±2–3 points with tiny density changes; treat
  differences under ~3 points as noise.

## Conventions

- Run `cargo fmt --all` (default settings, no `rustfmt.toml`) before every
  commit; `cargo fmt --all --check` must pass.
- Commits are authored by Joshua Flark <joshuajmark@gmail.com>, with no
  `Co-Authored-By` or other AI attribution trailers. Prefer small commits,
  each of which passes all checks on its own.
- Never commit downloaded models, `out/` or `TODO.md`.
- The Python bindings track the crate: a crate API change updates them in
  the same commit, and `python/Cargo.toml`'s version matches the crate's.
- Public API: output types (`Atlas`, `OutputMesh`, `OutputVertex`,
  `InstanceRect`, `Progress`) and `Density`, `Packing`, `Stage`, `Error` are
  `#[non_exhaustive]`. `Options` and `MeshInput` are not (that would forbid
  `..Default::default()`), so any new field there must have a default.
- The packaged crate (`include` in `Cargo.toml`) must not contain plans or
  roadmaps: keep those in `TODO.md`. Publishing to crates.io needs the
  owner's explicit go-ahead (versions are permanent).
