"""Tests of the uvunwrap Python bindings (the Rust crate tests the unwrapping
itself): input conversion, options, output arrays, errors and jobs."""

import array
import math

import pytest

import uvunwrap


def grid(nu, nv, wrap=False):
    """A torus (wrap) or a bumpy plane of quads, each as two triangles.
    Returns positions, indices and the quad each triangle came from."""
    cu, cv = (nu, nv) if wrap else (nu + 1, nv + 1)
    positions = []
    for j in range(cv):
        for i in range(cu):
            u, v = i / nu, j / nv
            if wrap:
                a, b = u * math.tau, v * math.tau
                r = 1 + 0.4 * math.cos(b)
                positions.append([r * math.cos(a), r * math.sin(a), 0.4 * math.sin(b)])
            else:
                positions.append([u, v, 0.2 * math.sin(u * 7) * math.cos(v * 5)])
    vid = lambda i, j: (j % cv) * cu + (i % cu)
    indices, polygons = [], []
    for j in range(nv):
        for i in range(nu):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            indices += [a, b, c, a, c, d]
            polygons += [len(polygons) // 2] * 2
    return positions, indices, polygons


def quad():
    return uvunwrap.Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 2, 0, 2, 3])


def test_output_arrays():
    positions, indices, _ = grid(8, 8)
    atlas = uvunwrap.generate([uvunwrap.Mesh(positions, indices)])
    assert atlas.width == 1024 and atlas.page_count == 1 and atlas.chart_count >= 1
    out = atlas.meshes[0]
    n = len(out.xref)
    assert (out.uv.typecode, out.xref.typecode, out.indices.typecode) == ("f", "I", "I")
    assert len(out.uv) == 2 * n and len(out.chart) == n and len(out.page) == n
    assert list(out.indices) != [] and len(out.indices) == len(indices)
    assert all(0.0 <= x <= 1.0 for x in out.uv)
    # Same triangles: each output vertex maps back to the input corner.
    assert [out.xref[i] for i in out.indices] == indices
    # corner_uvs is uv looked up per triangle corner.
    assert list(out.corner_uvs) == [out.uv[2 * i + k] for i in out.indices for k in (0, 1)]
    assert set(out.chart) <= set(range(atlas.chart_count)) and set(out.page) == {0}
    assert out.instances == []


@pytest.mark.parametrize(
    "positions, indices",
    [
        (array.array("f", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), array.array("I", [0, 1, 2, 0, 2, 3])),
        (array.array("d", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), array.array("i", [0, 1, 2, 0, 2, 3])),
        (memoryview(array.array("f", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0])), array.array("q", [0, 1, 2, 0, 2, 3])),
        ([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [[0, 1, 2], [0, 2, 3]]),
    ],
)
def test_buffer_and_sequence_inputs(positions, indices):
    out = uvunwrap.generate([uvunwrap.Mesh(positions, indices)]).meshes[0]
    assert len(out.indices) == 6


def test_numpy_inputs():
    np = pytest.importorskip("numpy")
    positions, indices, polygons = grid(6, 6)
    mesh = uvunwrap.Mesh(
        np.array(positions, dtype=np.float32),
        np.array(indices, dtype=np.int32).reshape(-1, 3),
        polygons=np.array(polygons, dtype=np.int64),
        ignored_faces=np.zeros(len(polygons), dtype=bool),
    )
    out = uvunwrap.generate([mesh]).meshes[0]
    assert len(out.indices) == len(indices)


def test_polygons_get_one_uv_per_corner():
    positions, indices, polygons = grid(24, 12, wrap=True)
    for packing in ("shape", "fast"):
        atlas = uvunwrap.generate(
            [uvunwrap.Mesh(positions, indices, polygons=polygons)],
            packing=packing,
            max_area_distortion=1.2,
        )
        out = atlas.meshes[0]
        seen = {}
        for corner, vertex in enumerate(indices):
            uv = (out.corner_uvs[2 * corner], out.corner_uvs[2 * corner + 1])
            assert seen.setdefault((polygons[corner // 3], vertex), uv) == uv


def test_options():
    positions, indices, _ = grid(8, 8)
    mesh = uvunwrap.Mesh(positions, indices)
    growing = uvunwrap.generate([mesh], texels_per_unit=100.0, page_size=None)
    assert growing.texels_per_unit == pytest.approx(100.0) and growing.width < 1024
    # Three 83 × 73-texel charts: only one fits a 128 × 128 page.
    pages = uvunwrap.generate([mesh] * 3, texels_per_unit=60.0, page_size=128)
    assert (pages.width, pages.height, pages.page_count) == (128, 128, 3)
    aligned = uvunwrap.generate([mesh], block_align=True, page_size=256, packing="fast")
    assert aligned.width % 4 == 0 and aligned.height % 4 == 0
    with pytest.raises(ValueError, match="packing"):
        uvunwrap.generate([mesh], packing="tight")
    with pytest.raises(ValueError, match="page_size is required"):
        uvunwrap.generate([mesh], page_size=None)


def test_instances():
    atlas = uvunwrap.generate([uvunwrap.Mesh(*grid(4, 4)[:2], instance_scales=[1.0, 2.0])])
    rects = atlas.meshes[0].instances
    assert len(rects) == 2 and all(r.page == 0 for r in rects)
    assert all(0 <= r.min[k] < r.max[k] <= 1 for r in rects for k in (0, 1))


def test_seams_cut_charts():
    # The quad's diagonal is edge 2 of triangle 0 (corner 2 -> 0).
    positions, indices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 2, 0, 2, 3]
    assert uvunwrap.generate([uvunwrap.Mesh(positions, indices)]).chart_count == 1
    for seams in ([[False, False, True], [False] * 3], [False, False, True] + [False] * 3,
                  array.array("B", [0, 0, 1, 0, 0, 0])):
        mesh = uvunwrap.Mesh(positions, indices, seams=seams)
        assert uvunwrap.generate([mesh]).chart_count == 2
    with pytest.raises(ValueError, match="multiple of 3"):
        uvunwrap.Mesh(positions, indices, seams=[True])


def test_errors():
    with pytest.raises(uvunwrap.UnwrapError, match="index 9 out of range"):
        uvunwrap.generate([uvunwrap.Mesh([[0, 0, 0]] * 3, [0, 1, 9])])
    with pytest.raises(uvunwrap.UnwrapError, match="polygon 0 has triangles with different face_materials"):
        uvunwrap.generate(
            [uvunwrap.Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 2, 0, 2, 3],
                           polygons=[0, 0], face_materials=[0, 1])]
        )
    with pytest.raises(ValueError, match="multiple of 3"):
        uvunwrap.Mesh([0, 0], [0, 1, 2])
    with pytest.raises(ValueError, match="out of range"):
        uvunwrap.Mesh([[0, 0, 0]] * 3, array.array("i", [0, -1, 2]))
    assert issubclass(uvunwrap.Cancelled, uvunwrap.UnwrapError)


def test_job_reports_progress_and_returns_the_atlas():
    positions, indices, _ = grid(32, 16, wrap=True)
    job = uvunwrap.start([uvunwrap.Mesh(positions, indices)])
    atlas = job.result(timeout=60)
    assert job.done
    assert job.progress.stage == "output" and job.progress.done == job.progress.total == 1
    assert job.result().chart_count == atlas.chart_count  # repeatable


def test_job_cancel():
    # Big enough that the cancel lands while it runs.
    positions, indices, _ = grid(512, 256, wrap=True)
    job = uvunwrap.start([uvunwrap.Mesh(positions, indices)])
    with pytest.raises(TimeoutError):
        job.result(timeout=0)
    job.cancel()
    with pytest.raises(uvunwrap.Cancelled):
        job.result(timeout=60)
    assert job.done


def test_start_validates_options_up_front():
    with pytest.raises(ValueError, match="packing"):
        uvunwrap.start([quad()], packing="nope")
