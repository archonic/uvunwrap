# uvunwrap: agent instructions

Pure-Rust automatic UV unwrapper (xatlas-like): segmentation → LSCM
parameterization → atlas packing.

## Goal

Be a drop-in replacement for every use Godot makes of xatlas, and better than
xatlas where Godot's users run into problems with it. Godot calls xatlas in one
place, `xatlas_mesh_lightmap_unwrap_callback` in `modules/xatlas_unwrap`
(installed as `array_mesh_lightmap_unwrap_callback`). It is behind
`ArrayMesh.lightmap_unwrap`, the scene and OBJ importers' "Static Lightmaps"
option (with an `.unwrap_cache` keyed by an MD5 of the input), and GridMap
baking. Per call it unwraps one mesh (all surfaces merged) and passes:

- positions with the base transform's scale already applied, normals,
  `u32` indices;
- `fixWinding = true`, `padding = 1`, `maxChartSize = 4094`,
  `blockAlign = true`, `texelsPerUnit = 1 / texel_size` (fixed density, a
  single page that grows to fit);
- and reads back `xref`, UVs normalized to the page, and the page size as a
  size hint, which LightmapGI uses to allocate that mesh's lightmap.

Known Godot problems with xatlas, which this crate should not have:
[#117752](https://github.com/godotengine/godot/issues/117752) (output differs
across platforms, so a lightmap baked on one machine misaligns on another),
[#92119](https://github.com/godotengine/godot/issues/92119) (deadlock in
xatlas's task scheduler after many unwraps),
[proposal #12464](https://github.com/godotengine/godot-proposals/issues/12464)
(wants global, bilinear-aware lightmap packing across meshes). Unwrap time on
large imports matters too: xatlas takes 13–14 s on bunny or armadillo.

What that means for the crate (✅ done, ◐ partial, ☐ missing):

1. ✅ **Bake-quality UVs.** No overlaps or flips, every face inside [0,1],
   padding in texels, low area distortion, uniform texel density.
2. ✅ **Fixed density, one growing page, size reported.** `texels_per_unit:
   Some(1 / texel_size)` with `resolution: 0`; `Atlas::{width, height}` is
   the size hint.
3. ✅ **Seam splitting a caller can apply.** Output vertices carry `xref` to
   the input vertex, as Godot's `SurfaceTool` rebuild expects.
4. ✅ **Normals as seam hints.** Godot passes normals so hard edges become
   seams. Here, edges where the input splits vertices are chart boundaries,
   which covers that (and UV seams, and surface boundaries) without the
   normals. Also `face_materials` and `face_ignore`.
5. ☐ **Maximum chart size** (`maxChartSize`): charts bigger than a limit are
   scaled down rather than growing the page or failing.
6. ☐ **Block alignment** (`blockAlign`): place charts on a 4×4 texel grid for
   block-compressed lightmaps.
7. ◐ **Inconsistent winding** (`fixWinding`): flipped faces are currently cut
   off as separate charts rather than fixed.
8. ☐ **Determinism.** Same input gives the same output bit for bit, for any
   thread count and ideally across platforms (x86/ARM, with and without
   SIMD), so unwrap caches and baked lightmaps stay valid (#117752). At the
   least, guarantee it per platform, test it, and document the limits.
9. ✅ **No global state or scheduler** (#92119): rayon only, safe to call any
   number of times from any thread. Keep it that way.
10. ◐ **Robust on real assets.** Degenerate faces, triangle soup and
    non-manifold input must never panic and must still receive valid UVs.
11. ☐ **C API** for embedding in C++ engines: a separate crate producing a
    static library and header that maps directly onto Godot's callback
    signature, plus a Godot module that installs it. Godot core doesn't take
    Rust dependencies, so this targets custom engine builds first.
12. ☐ **Beyond xatlas, for Godot's open requests:** multi-mesh global packing
    (`generate` already packs many meshes into shared pages; LightmapGI would
    need per-mesh placement output), progress reporting and cancellation, and
    repacking existing UVs (xatlas `AddUvMesh`).

Non-goals: lighting bakes themselves, matching xatlas output exactly, quad or
polygon output, interactive UV editing, and engine-specific integrations other
than the Godot path above (Bevy included, for now).

Maintainer constraint: keep the crate focused and publishable. Engine glue
belongs in separate crates, and new features need a concrete Godot or baking
use case.

## Layout

- `src/lib.rs`: public API (`generate`, `MeshInput` with optional
  `face_materials`/`face_ignore`/`scale`, `Options`, `Atlas`, `OutputMesh`,
  `OutputVertex`, `Error`) and output assembly (seam vertex duplication,
  `xref` back to input vertices, degenerate/ignored faces collapsed onto one
  existing UV).
- `src/topology.rs`: welding by exact position (scaled by `MeshInput::scale`),
  exclusion (degenerate: altitude < 1e-6 × bbox diagonal, or ignored),
  manifold + consistently oriented adjacency within one material only. Edges
  where the input splits vertices (authored seams) are cut too, unless more
  than half of all adjacent edges are split (flat shading).
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
- `examples/unwrap.rs`: OBJ in (one vertex per `v/vt/vn` corner, like an
  engine import; `--flat` splits every face) → OBJ with `vt` + SVG per page.
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
| Spot | 5.9K | 47 | 27 ms | 19 ms | 71% |
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
- Authored seams as a soft segmentation cost do nothing below weight 2:
  growth only stops at `MAX_COST`, so costs just reorder growth (spot followed
  10% of UV seams at weight 0–1, 90% at 2). Hence hard cuts in topology.
- Remaining costs: LSCM solves (successful + failed) dominate; igea's
  critical path is one 98K-face chart that fails and is re-solved in halves.
  Packing ≈ 9 sequential density trials.

## Next steps (priority order)

1. Determinism: test identical output across thread counts and runs; find
   platform-dependent math (libm trig, faer SIMD paths) and decide what to
   guarantee.
2. Godot parity: maximum chart size, 4×4 block alignment, winding fixing.
3. C API crate + Godot module implementing `array_mesh_lightmap_unwrap_callback`;
   compare against xatlas on Godot's demo scenes.
4. CI (tests in both feature configs, wasm32 build), then publish 0.1 to crates.io.
5. Refresh `docs/atlases.png` (still shows the old rect-packed atlases).
