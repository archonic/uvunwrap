"""Fast automatic UV unwrapping and atlas packing (Rust crate `uvunwrap`)."""

from array import array
from typing import Any, Literal, Optional, Sequence

__version__: str

class UnwrapError(Exception):
    """Raised when uvunwrap rejects its input or can't produce an atlas."""

class Cancelled(UnwrapError):
    """Raised by `Job.result()` after `Job.cancel()`."""

class Mesh:
    """A triangle mesh to unwrap.

    `positions` holds x, y, z per vertex and `indices` three vertex indices
    per triangle, as buffers (array.array, numpy, memoryview) or flat or
    nested sequences. `polygons`, `face_materials` and `ignored_faces` have
    one entry per triangle, `seams` three per triangle (flag k marks the
    edge from corner k to corner k + 1 as a required seam),
    `instance_scales` one per instance and `existing_uvs` u, v per
    vertex."""

    def __init__(
        self,
        positions: Any,
        indices: Any,
        *,
        polygons: Any = None,
        face_materials: Any = None,
        ignored_faces: Any = None,
        seams: Any = None,
        scale: float = 1.0,
        instance_scales: Any = None,
        existing_uvs: Any = None,
    ) -> None: ...

class Progress:
    """Progress of a job: the current stage, units done in it, and units in
    it if known (not for packing, which counts trials)."""

    stage: Literal["segmenting", "parameterizing", "packing", "output", "unknown"]
    done: int
    total: Optional[int]

class InstanceRect:
    """Where one instance of a mesh lives: its page and its rectangle's
    corners in page coordinates."""

    page: int
    min: tuple[float, float]
    max: tuple[float, float]

class OutputMesh:
    """An unwrapped mesh. Per output vertex: `xref` (its input vertex), `uv`
    (u, v in [0, 1]), `chart` and `page` (-1 for none). `indices`: the input
    triangles, indexing output vertices. `corner_uvs`: u, v per triangle
    corner."""

    xref: array[int]
    uv: array[float]
    chart: array[int]
    page: array[int]
    indices: array[int]
    corner_uvs: array[float]
    instances: list[InstanceRect]

class Atlas:
    """The generated atlas and one OutputMesh per input mesh."""

    width: int
    height: int
    page_count: int
    texels_per_unit: float
    chart_count: int
    utilization: float
    meshes: list[OutputMesh]

class Job:
    """An unwrap running in a background thread (see `start`)."""

    @property
    def progress(self) -> Optional[Progress]: ...
    @property
    def done(self) -> bool: ...
    def cancel(self) -> None: ...
    def result(self, timeout: Optional[float] = None) -> Atlas: ...

def generate(
    meshes: Sequence[Mesh],
    *,
    page_size: Optional[int] = 1024,
    texels_per_unit: Optional[float] = None,
    padding: int = 2,
    packing: Literal["shape", "fast"] = "shape",
    max_chart_angle: float = 70.0,
    max_area_distortion: float = 4.0,
    max_chart_area: float = 0.0,
    max_boundary_length: float = 0.0,
    allow_rotation: bool = True,
    max_chart_size: int = 0,
    block_align: bool = False,
) -> Atlas:
    """Unwraps `meshes` into one shared atlas in the calling thread."""

def start(
    meshes: Sequence[Mesh],
    *,
    page_size: Optional[int] = 1024,
    texels_per_unit: Optional[float] = None,
    padding: int = 2,
    packing: Literal["shape", "fast"] = "shape",
    max_chart_angle: float = 70.0,
    max_area_distortion: float = 4.0,
    max_chart_area: float = 0.0,
    max_boundary_length: float = 0.0,
    allow_rotation: bool = True,
    max_chart_size: int = 0,
    block_align: bool = False,
) -> Job:
    """Starts unwrapping `meshes` in a background thread."""
