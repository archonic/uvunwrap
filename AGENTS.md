# uvunwrap: agent instructions

Pure-Rust automatic UV unwrapper (xatlas-like): segmentation → LSCM
parameterization → atlas packing.

## Goal

A pure-Rust, xatlas-quality UV unwrapper, with Bevy as the primary consumer.
Bevy renders lightmaps from `ATTRIBUTE_UV_1` (`Lightmap { image, uv_rect }`,
where `uv_rect` lets meshes share one atlas texture) but has no unwrapper or
baker. The open baker issue
([#12233](https://github.com/bevyengine/bevy/issues/12233)) names xatlas for
the UV step; this crate should be the obvious choice instead. Near-term Bevy
uses: second UV sets for procedural/CSG/voxel meshes, texture painting and
decals, and the UV step of any future baker.

Finish this crate before any integration crate. Use xatlas's feature set as
used by engines (Godot's lightmap UV2 unwrap is a good reference: fixed
density, padding 1, `maxChartSize` 4094, `blockAlign`, `fixWinding`, page
size as a size hint) as the parity bar. Avoid the problems Godot users hit
with xatlas: cross-platform nondeterminism
([godot#117752](https://github.com/godotengine/godot/issues/117752)), a
scheduler deadlock ([godot#92119](https://github.com/godotengine/godot/issues/92119)),
and slow unwraps on large imports (xatlas: 13–14 s on bunny/armadillo).

Crate goals (✅ done, ◐ partial, ☐ missing):

1. ✅ **Bake-quality UVs.** No overlaps or flips, every face inside [0,1],
   padding in texels, low area distortion, uniform texel density.
2. ✅ **Seam splitting a caller can apply.** Output vertices carry `xref` to
   the input vertex, so callers remap every other attribute.
3. ✅ **Respect authored structure.** Edges where the input splits vertices
   (hard edges, UV seams) are chart boundaries, which covers xatlas's
   normal/texture seam weights without passing normals or UVs. Also
   `face_materials` and `ignored_faces`.
4. ✅ **Fixed or automatic density**, one growing page or fixed-size pages,
   page size reported; per-mesh `scale` for scaled instances.
5. ✅ **Multi-mesh atlases and instances.** `generate` packs many meshes
   into shared pages. `MeshInput::instance_scales` unwraps a mesh once into its own
   layout (UVs in [0,1]) and gives each instance an unrotated rectangle on one
   page (`OutputMesh::instances`), which is Bevy's `Lightmap::uv_rect`; one
   instance keeps a mesh on one page.
6. ◐ **Determinism.** Same input gives bit-identical output for any thread
   count (tested). No platform math (`geom::cos`/`geom::angle` replace libm
   trig; never call `sin`/`cos`/`atan2`/`acos`/`powf`/`exp` etc. in `src/`),
   so without `direct-solver` output should match across platforms;
   unverified until CI compares example hashes on x86-64, ARM and wasm32.
   Missing: `direct-solver` (faer) picks SIMD kernels at runtime, so it can
   differ across architectures; fixing that needs a deterministic sparse
   Cholesky of our own.
7. ✅ **Maximum chart size** (xatlas `maxChartSize`): `Options::max_chart_size`
   scales bigger charts down (per-chart density) instead of growing the page
   or failing.
8. ✅ **Block alignment** (xatlas `blockAlign`): `Options::block_align`; each
   chart owns whole 4×4 blocks (masks downsampled to blocks, packed in block
   units), costing 1–4 points of utilization.
9. ✅ **Consistent UV winding** (xatlas `fixWinding`, which is about output
   UV orientation): every output triangle is CCW in UV in its input index
   order, including inconsistently wound input (those faces become separate
   charts).
10. ✅ **Progress and cancellation** (xatlas `ProgressCallback`):
    `generate_with_progress` reports `Progress { stage, done, total }`
    (meshes, faces, packing trials, meshes) and cancels when the callback
    returns `false`. Latency is one in-flight chart solve (igea: ~0.13 s).
11. ✅ **Repack existing UVs** (xatlas `AddUvMesh`, but better for
    lightmaps): `MeshInput::existing_uvs` makes each UV island one chart, scaled to
    its 3D area (xatlas keeps UV scale) and rotated freely; mirrored islands
    are flipped, invalid ones (flips, degenerate, self-overlap) unwrapped
    normally. Spot: 14 charts, 70% utilization, half the time.
12. ✅ **No global state or scheduler**: rayon only, safe to call any number
    of times from any thread; zero dependencies and single-threaded under
    `--no-default-features` for wasm. Keep it that way.
13. ◐ **Robust on real assets.** Degenerate faces, triangle soup and
    non-manifold input must never panic and must still receive valid UVs.

After these: CI and publish 0.1, then a separate Bevy integration crate
(`Mesh` → `Mesh` with UV_1 and every attribute remapped, `Lightmap`
`uv_rect` helpers). Fyrox's lightmapper uses `uvgen` (box projection + rect
packing); replacing it is an optional later win.

Non-goals: lighting bakes themselves, matching xatlas output exactly, quad or
polygon output, and interactive UV editing.

Maintainer constraint: keep the crate focused and publishable. Engine glue
belongs in separate crates, and new features need a concrete engine or baking
use case.

## Layout

- `src/lib.rs`: public API (`generate`/`generate_with_progress`,
  `MeshInput::new` plus optional `face_materials`/`ignored_faces`/`scale`/
  `instance_scales`/`existing_uvs`, `Options` with `Density::{Auto, Fixed}`,
  `Atlas`, `OutputMesh`, `OutputVertex` (`page: Option`), `InstanceRect`,
  `Error` naming the offending option/array/instance). `run` is validate →
  `unwrap_meshes` (segment + parameterize, meshes in parallel) →
  `pack_meshes` (each mesh's `Slot`: `Direct` or `Tiled`) → `assemble_mesh`
  per mesh in parallel (`place_chart`, `instance_rects`, `split_seams`: seam
  vertex duplication, `xref` back to input vertices, degenerate/ignored faces
  collapsed onto one existing UV).
- `src/par.rs`: the only place that switches on the `parallel` feature
  (`map`, `map_ref`, `flat_map`, `join`; order-preserving).
- `src/geom.rs`: f64 vector helpers, convex hull, min-area rectangle, triangle
  overlap, and deterministic `cos`/`angle` (no platform math anywhere).
- `src/topology.rs`: welding by exact position (scaled by `MeshInput::scale`),
  exclusion (degenerate: altitude < 1e-6 × bbox diagonal, or ignored),
  manifold + consistently oriented adjacency within one material only. Edges
  where the input splits vertices (authored seams) are cut too, unless more
  than half of all adjacent edges are split (flat shading); with input UVs,
  edges where the UVs differ are cut instead (always).
- `src/segment.rs`: xatlas-style growth over planar regions: cost = normal
  deviation (crease-preserving smoothed normals, hard `max_chart_angle` cone)
  + straightness + roundness + crease penalty; then merge small enclosed
  charts and smooth single-face teeth off boundaries. `components` gives the
  UV islands of meshes with input UVs (no segmentation).
- `src/param.rs`: planar projection if flat, else LSCM; validate (no flips, no
  overlaps via grid + SAT, area ratio ≤ `max_area_distortion`); on failure
  split chart in two (graph-distance Voronoi, seeded at the overlapping pair
  when validation found an overlap) and retry, halves in parallel.
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
- `src/mask.rs`: conservative chart rasterization + padding dilation into
  per-row span masks (4 rotations derived exactly), and the page bitset with
  per-row free runs used for exact bottom-left placement search.
- `src/pack.rs`: `prepare` (sizes and tile layouts, then `rasterize` masks
  only once sizes pass the page and texel-budget checks), `place` (shape or
  rect), and one function per mode: `pack_auto` (bisection to fit one page,
  seeded by the rect result with at most 8 back-off steps, cropped to used
  height, multi-page fallback), `pack_unbounded` (fixed density, 7 widths in
  parallel on one set of masks, smallest area), `pack_pages` (fixed density,
  fixed pages). Per-chart density (`chart_density`) applies
  `max_chart_size`; `block_align` packs in 4×4 block units
  (`Mask::downsample`, rects rounded up) and scales placements back.
- `src/skyline.rs`: rectangle packer (`pack_rects`) for `Packing::Rect` and
  the auto-mode lower bound.
- `src/tile.rs`: instanced meshes: `layout_tile` packs a tile's charts once
  (`pack_unbounded` at the smallest instance's density, capped so the
  largest instance fits the page); each instance becomes one unrotatable
  rectangle (`Mask::rect`) among the packed items.
- `tests/`: `common/mod.rs` holds generators and `check`, which verifies
  every output guarantee (topology, positive UV area, no overlaps, 2×padding
  between charts and instances, padding to page edges, ignored faces
  collapsed; sweep-and-prune, so large meshes are fine). Topic files:
  `unwrap`, `inputs`, `packing`, `instances`, `progress`, `robustness`
  (hostile inputs and a seeded random mesh/options property test).
- `examples/unwrap.rs`: OBJ in (one vertex per `v/vt/vn` corner, like an
  engine import; `--flat` splits every face) → OBJ with `vt` + SVG per page.
- `benches/generate.rs`: criterion (`cargo bench`; `-- --test` runs each case
  once): tori and a 2000-triangle soup plus
  `models/{spot,beetle,bunny,armadillo,nefertiti}.obj` when present, each
  with shape, rect, fixed density and four instances, plus `repack` for
  models with UVs.

## Commands

- `cargo test` and `cargo test --no-default-features` (both must pass)
- `cargo run --release --example unwrap -- [--rect] model.obj out/model`
- Test models: https://github.com/alecjacobson/common-3d-test-models (`data/*.obj`),
  downloaded into the gitignored `models/` dir (bunny is `stanford-bunny.obj`).
- Render atlas SVGs: `rsvg-convert -w 800 out/x_page0.svg -o out/x.png`
  (`qlmanage` hangs).

## Conventions

- Run `cargo fmt` (default settings, no `rustfmt.toml`) before every commit;
  `cargo fmt --check` must pass.
- Commits are authored by Joshua Flark <joshuajmark@gmail.com>, with no
  `Co-Authored-By` or other AI attribution trailers.
- `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`.
- Zero dependencies with `--no-default-features`; must stay wasm-friendly.
- Never commit downloaded models or `out/`.

## Baseline (release, 1024 page, padding 2, Apple M5 10 cores)

| Model | Triangles | Charts | 1 core | Multi-core | Utilization |
|---|---|---|---|---|---|
| Spot | 5.9K | 47 | 27 ms | 19 ms | 71% |
| Stanford bunny | 69K | 58 | 0.19 s | 82 ms | 65% |
| Armadillo | 100K | 257 | 0.40 s | 0.16 s | 60% |
| Nefertiti | 100K | 456 | 0.74 s | 0.40 s | 64% |
| Igea | 269K | 83 | 0.96 s | 0.48 s | 66% |

Utilization is true surface area / atlas area (stricter than xatlas, which
counts rasterized texels). xatlas reference (default pack options, 10
threads): 55–71% utilization, 13–14 s on bunny/armadillo.

Benchmark with best-of-N single-core runs (`RAYON_NUM_THREADS=1`); the dev
machine is often loaded, so multi-core timings are noisy. Interleave runs of
the two binaries being compared. A change that adds no hot code can still move
timings ~4% by changing how code is split into codegen units; to tell whether
a slowdown is real, compare both builds with
`CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1` (also 2–3% faster overall).

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
- Instances share one layout padded for the smallest instance, so a mesh
  whose instance scales differ by 4× gives the largest instance 4× the
  padding (visible on spot with scales 2, 1, 0.5). Inherent to sharing UVs;
  callers with very different scales can split them into separate meshes.
- Large scenes pack slowly with shape packing: 300 small tori (2485 charts)
  take 1.6 s on a 1024 page, 8.2 s at 2048, 20.8 s at 4096 (rect: 31 ms).
  Placement is ~99% of it, single-threaded; most of that is the rotated
  searches of small late charts (a 90°-rotated strip is narrow, so the
  row-width rejection in `Page::find` rarely fires). Tried (2026-10-08):
  - jumping past rows too narrow for any mask row: exact, no gain (rows are
    never that full);
  - rotations searched in parallel on pages ≥ 2048: exact, 4096 20.8 → 12.3 s
    (unmerged branch `packing-perf`);
  - Blender-style coarse density search (grid capped near 1024, then one
    full-resolution pack): 4096 → 5.2 s, −0.9 points;
  - rotating only the largest N charts (Blender uses 50): ~10× faster
    packing; N = 150 leaves the single models' utilization unchanged but
    costs the tori scene 1.2 points; N = 50 costs nefertiti 2.2;
  - xatlas flow (0°/90° only, per-size-bucket start positions): 1–5 points
    worse on the single models (bunny −4.9) for 10–15% speed: our shape
    packer relies on 180°/270° and full-page search to nest charts.
  Decision: keep the exhaustive four-rotation search; revisit only for
  large-scene users, preferring one of the above over new techniques.
- Low utilization on small atlases is padding, not the packer: padding 2
  around 58 charts on a ~150 px page costs ~20 points (xatlas pads the same
  way: 8-neighbour dilation on both charts, plus a bilinear border by
  default). At equal density fixed mode matches auto within ±2.5 points;
  trying 7 widths gains another 1–3.
- Authored seams as a soft segmentation cost do nothing below weight 2:
  growth only stops at `MAX_COST`, so costs just reorder growth (spot followed
  10% of UV seams at weight 0–1, 90% at 2). Hence hard cuts in topology.
- Remaining costs: LSCM solves (successful + failed) dominate; igea's
  critical path is one 98K-face chart that fails and is re-solved in halves.
  Packing ≈ 9 sequential density trials.

## Next steps (priority order)

### 0.1

1. Push to github.com/archonic/uvunwrap and get CI (`.github/workflows/ci.yml`)
   green: fmt, clippy `-D warnings` and docs; tests on Linux/macOS/Windows in
   both feature configs; MSRV 1.85 (faer's `generativity` needs it); a
   wasm32 build without default features; and the example's torus hash
   compared across Linux x86-64/ARM64, macOS ARM64 and Windows without
   `direct-solver`. Once that passes, state the cross-platform guarantee as
   tested in the crate docs and README. Then publish 0.1.

### 0.2

2. Replace `Packing::Rect` (rectangle skyline) with a TABI-style
   fold-and-push packer, so there are still exactly two packers: `Shape`
   (exhaustive raster search, best quality, default) and the new fast one.
   Reference: Gu, Vining & Sheffer, "TABI: Tight and Balanced Interactive
   Atlas Packing", arXiv 2602.07782 (2026), Sec. 3–4 and Algorithms 1–4.
   Their GPU version averages ~5 ms vs ~2.3 s for xatlas (random) at mean
   L2 stretch 2.16 vs 1.89–2.00; on UV-unwrapped models often equal to
   xatlas. Plan, CPU-only:
   - Sort charts by height (bounding boxes rotated taller than wide), fold
     into rows of page width, push each row up against a frontline (per
     column, the lowest filled texel).
   - Use exact per-column top/bottom profiles and per-row left/right
     profiles read off our existing padded masks (TABI approximates them
     with sliced AABBs + OBBs because it never rasterizes).
   - Compact neighbours horizontally within a row using left/right
     profiles; push charts using top profiles and write bottom profiles
     into the frontline; correct offsets where a pushed chart could
     intersect its compacted neighbour (Alg. 1).
   - Balance: per row, try left/right direction × full width / fold into
     a knee concavity × with/without horizontal compacting; keep the
     configuration with the lowest maximum frontline (knee: height jump
     ≥ 10% of page height and ≥ 20% of the taller chart).
   - Keep our density search (bisection; TABI tests 64 discrete scales),
     padding, block alignment, instances and multi-page support.
   - No reflection (TABI mirrors charts; that would flip UV winding, which
     we guarantee). Rotations by multiples of 90° only.
   - Known limitation: a frontline can't nest charts into enclosed holes
     or deep side concavities (torus rings), which `Shape` can.
   Evaluate against `Shape` on the benchmark models and the 300-torus scene
   (1024/2048/4096 pages): utilization (treat differences under ~3 points
   as noise) and time. Target: milliseconds for thousands of charts.
3. Deterministic direct solver (own sparse Cholesky), if cross-platform
   determinism must hold with the fast path.
4. Bevy integration crate.
