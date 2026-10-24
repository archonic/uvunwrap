# uvunwrap

Fast automatic UV unwrapping and atlas packing for lightmaps, texture baking
and texture painting: Python bindings for the
[uvunwrap](https://crates.io/crates/uvunwrap) Rust crate. Prebuilt wheels,
no compiler needed; Python 3.11+, no required dependencies.

```sh
pip install uvunwrap
```

## Usage

```python
import uvunwrap

# A unit quad as two triangles of one polygon.
positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
indices = [0, 1, 2, 0, 2, 3]
mesh = uvunwrap.Mesh(positions, indices, polygons=[0, 0])

atlas = uvunwrap.generate([mesh], page_size=1024, padding=2)
out = atlas.meshes[0]
out.uv          # array('f'): u, v per output vertex, in [0, 1]
out.xref        # array('I'): the input vertex each output vertex came from
out.indices     # array('I'): the same triangles, indexing output vertices
out.corner_uvs  # array('f'): u, v per triangle corner
```

Output meshes keep the input triangles in order; vertices on chart seams are
duplicated, and `xref` maps each one back to its input vertex so other
attributes can be copied. `corner_uvs` gives one UV per triangle corner
directly, for UV maps stored per polygon corner (e.g. Blender's).

Inputs accept anything with the buffer protocol (`array.array`, numpy arrays
of any common dtype, `memoryview`) or plain sequences, flat or nested.

### Background jobs, progress and cancellation

```python
job = uvunwrap.start([mesh])
while not job.done:
    print(job.progress)   # e.g. Progress(parameterizing, 1200/5856)
    ...                   # or job.cancel(); result() then raises uvunwrap.Cancelled
atlas = job.result()
```

The unwrap runs in a background thread, so a UI stays responsive.

### Options

Keyword arguments of `generate` and `start`:

| Option | Default | |
|---|---|---|
| `page_size` | `1024` | Page size in texels. |
| `texels_per_unit` | `None` | `None`: the highest density that fits one page. A number: that fixed density, on `page_size` pages (`page_size=None`: one page that grows to fit). |
| `padding` | `2` | Empty texels around each chart. |
| `packing` | `"shape"` | `"shape"` (densest) or `"fast"` (milliseconds for thousands of charts). |
| `max_chart_angle` | `70.0` | Maximum angle in degrees between a face and its chart's average normal. |
| `max_area_distortion` | `4.0` | Maximum per-triangle area distortion before a chart is split. |
| `max_chart_area` | `0.0` | Largest chart surface area in world units (`0`: no limit). |
| `max_boundary_length` | `0.0` | Largest chart boundary length in world units (`0`: no limit). |
| `allow_rotation` | `True` | Rotate charts by multiples of 90° while packing. |
| `max_chart_size` | `0` | Largest chart side in texels (`0`: no limit). |
| `block_align` | `False` | Give every chart whole 4×4 texel blocks (for BC/ASTC compression). |

`Mesh` also takes `polygons` (the polygon each triangle came from, so
polygons stay whole), `seams` (three flags per triangle, one per edge, for
edges that must become seams), `face_materials`, `ignored_faces`, `scale`,
`instance_scales` and `existing_uvs`; see the
[crate documentation](https://docs.rs/uvunwrap) for their details.

## License

MIT
