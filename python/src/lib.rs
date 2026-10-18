//! Python bindings for [uvunwrap](https://crates.io/crates/uvunwrap),
//! published to PyPI as `uvunwrap`. A thin layer over the crate's API:
//! `Mesh` holds an input mesh, `generate` unwraps in the calling thread
//! and `start` in a background thread (a `Job` that reports progress and
//! can be cancelled), and the result is an `Atlas` of output meshes whose
//! arrays are `array.array`s.
//!
//! Inputs accept anything with the buffer protocol (`array.array`, numpy
//! arrays, `memoryview`) or plain sequences, flat or nested.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

create_exception!(
    uvunwrap,
    UnwrapError,
    PyException,
    "Raised when uvunwrap rejects its input or can't produce an atlas."
);
create_exception!(
    uvunwrap,
    Cancelled,
    UnwrapError,
    "Raised by `Job.result()` after `Job.cancel()`."
);

/// The Python exception for a crate error.
fn to_py(e: uvunwrap::Error) -> PyErr {
    match e {
        uvunwrap::Error::Cancelled => Cancelled::new_err("cancelled"),
        e => UnwrapError::new_err(e.to_string()),
    }
}

/// A buffer's items as `T`, if its item type is `S`.
fn buffer_as<S: Element + Copy, T>(
    obj: &Bound<'_, PyAny>,
    convert: impl Fn(S) -> PyResult<T>,
) -> Option<PyResult<Vec<T>>> {
    let buffer = PyBuffer::<S>::get(obj).ok()?;
    Some(
        buffer
            .to_vec(obj.py())
            .and_then(|v| v.into_iter().map(&convert).collect()),
    )
}

/// Non-negative integers from a buffer of any common integer type, or a flat
/// or nested sequence.
fn numbers(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
    let int = |x: i64| {
        u32::try_from(x).map_err(|_| PyValueError::new_err(format!("{name}: {x} is out of range")))
    };
    if let Some(v) = buffer_as::<i32, u32>(obj, |x| int(x as i64))
        .or_else(|| buffer_as::<u32, u32>(obj, Ok))
        .or_else(|| buffer_as::<i64, u32>(obj, int))
        .or_else(|| buffer_as::<u64, u32>(obj, |x| int(x.min(i64::MAX as u64) as i64)))
    {
        return v;
    }
    if let Ok(flat) = obj.extract::<Vec<u32>>() {
        return Ok(flat);
    }
    let nested: Vec<Vec<u32>> = obj.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{name}: expected a buffer or a sequence of non-negative integers"
        ))
    })?;
    Ok(nested.into_iter().flatten().collect())
}

/// Floats from a float32/float64 buffer or a flat or nested sequence.
fn floats(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f32>> {
    if let Some(v) =
        buffer_as::<f32, f32>(obj, Ok).or_else(|| buffer_as::<f64, f32>(obj, |x| Ok(x as f32)))
    {
        return v;
    }
    if let Ok(flat) = obj.extract::<Vec<f32>>() {
        return Ok(flat);
    }
    let nested: Vec<Vec<f32>> = obj.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{name}: expected a buffer or a sequence of numbers"
        ))
    })?;
    Ok(nested.into_iter().flatten().collect())
}

/// Booleans from an integer buffer or a sequence (numpy bool arrays work as
/// sequences).
fn bools(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    if let Some(v) = buffer_as::<u8, bool>(obj, |x| Ok(x != 0))
        .or_else(|| buffer_as::<i32, bool>(obj, |x| Ok(x != 0)))
    {
        return v;
    }
    obj.extract().map_err(|_| {
        PyValueError::new_err(format!("{name}: expected a buffer or a sequence of bools"))
    })
}

/// `values` in groups of `N`, or an error naming the array.
fn groups<const N: usize>(values: Vec<f32>, name: &str) -> PyResult<Vec<[f32; N]>> {
    if values.len() % N != 0 {
        return Err(PyValueError::new_err(format!(
            "{name}: {} values is not a multiple of {N}",
            values.len()
        )));
    }
    Ok(values
        .chunks_exact(N)
        .map(|c| std::array::from_fn(|k| c[k]))
        .collect())
}

/// The data of a [`Mesh`], shared with background jobs.
struct MeshData {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    face_materials: Option<Vec<u32>>,
    ignored_faces: Option<Vec<bool>>,
    polygons: Option<Vec<u32>>,
    scale: f32,
    instance_scales: Vec<f32>,
    existing_uvs: Option<Vec<[f32; 2]>>,
}

impl MeshData {
    fn input(&self) -> uvunwrap::MeshInput<'_> {
        uvunwrap::MeshInput {
            face_materials: self.face_materials.as_deref(),
            ignored_faces: self.ignored_faces.as_deref(),
            polygons: self.polygons.as_deref(),
            scale: self.scale,
            instance_scales: &self.instance_scales,
            existing_uvs: self.existing_uvs.as_deref(),
            ..uvunwrap::MeshInput::new(&self.positions, &self.indices)
        }
    }
}

/// A triangle mesh to unwrap. `positions` holds x, y, z per vertex and
/// `indices` three vertex indices per triangle, flat or nested. The optional
/// arrays have one entry per triangle (`face_materials`, `ignored_faces`,
/// `polygons`), per instance (`instance_scales`) or per vertex
/// (`existing_uvs`, u and v); see the crate's `MeshInput` for their meaning.
#[pyclass(frozen, module = "uvunwrap")]
struct Mesh(Arc<MeshData>);

#[pymethods]
impl Mesh {
    #[new]
    #[pyo3(signature = (
        positions, indices, *, polygons=None, face_materials=None, ignored_faces=None,
        scale=1.0, instance_scales=None, existing_uvs=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        positions: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        polygons: Option<&Bound<'_, PyAny>>,
        face_materials: Option<&Bound<'_, PyAny>>,
        ignored_faces: Option<&Bound<'_, PyAny>>,
        scale: f32,
        instance_scales: Option<&Bound<'_, PyAny>>,
        existing_uvs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Mesh(Arc::new(MeshData {
            positions: groups(floats(positions, "positions")?, "positions")?,
            indices: numbers(indices, "indices")?,
            face_materials: face_materials
                .map(|m| numbers(m, "face_materials"))
                .transpose()?,
            ignored_faces: ignored_faces
                .map(|m| bools(m, "ignored_faces"))
                .transpose()?,
            polygons: polygons.map(|m| numbers(m, "polygons")).transpose()?,
            scale,
            instance_scales: instance_scales
                .map(|k| floats(k, "instance_scales"))
                .transpose()?
                .unwrap_or_default(),
            existing_uvs: existing_uvs
                .map(|uv| groups(floats(uv, "existing_uvs")?, "existing_uvs"))
                .transpose()?,
        })))
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh({} vertices, {} triangles)",
            self.0.positions.len(),
            self.0.indices.len() / 3
        )
    }
}

/// The crate's `Options` from keyword arguments. Without `texels_per_unit`
/// the density is automatic, fitting one `page_size` page; with it, fixed,
/// on `page_size` pages or (`page_size=None`) one page that grows to fit.
#[allow(clippy::too_many_arguments)]
fn options(
    page_size: Option<u32>,
    texels_per_unit: Option<f32>,
    padding: u32,
    packing: &str,
    max_chart_angle: f32,
    max_area_distortion: f32,
    allow_rotation: bool,
    max_chart_size: u32,
    block_align: bool,
) -> PyResult<uvunwrap::Options> {
    let density = match (texels_per_unit, page_size) {
        (Some(texels_per_unit), page_size) => uvunwrap::Density::Fixed {
            texels_per_unit,
            page_size,
        },
        (None, Some(page_size)) => uvunwrap::Density::Auto { page_size },
        (None, None) => {
            return Err(PyValueError::new_err(
                "page_size is required without texels_per_unit",
            ))
        }
    };
    let packing = match packing {
        "shape" => uvunwrap::Packing::Shape,
        "fast" => uvunwrap::Packing::Fast,
        other => {
            return Err(PyValueError::new_err(format!(
                "packing must be \"shape\" or \"fast\", not {other:?}"
            )))
        }
    };
    // Every field is set today; the update keeps this compiling, with
    // defaults, when the crate adds options.
    #[allow(clippy::needless_update)]
    let opts = uvunwrap::Options {
        density,
        padding,
        packing,
        max_chart_angle,
        max_area_distortion,
        allow_rotation,
        max_chart_size,
        block_align,
        ..uvunwrap::Options::default()
    };
    Ok(opts)
}

/// A unit of progress: the current stage (`"segmenting"`,
/// `"parameterizing"`, `"packing"` or `"output"`), units done in it, and
/// units in it if known (not for packing, which counts trials).
#[pyclass(frozen, get_all, module = "uvunwrap")]
struct Progress {
    stage: &'static str,
    done: usize,
    total: Option<usize>,
}

#[pymethods]
impl Progress {
    fn __repr__(&self) -> String {
        match self.total {
            Some(t) => format!("Progress({}, {}/{t})", self.stage, self.done),
            None => format!("Progress({}, {})", self.stage, self.done),
        }
    }
}

impl From<uvunwrap::Progress> for Progress {
    fn from(p: uvunwrap::Progress) -> Self {
        let stage = match p.stage {
            uvunwrap::Stage::Segmenting => "segmenting",
            uvunwrap::Stage::Parameterizing => "parameterizing",
            uvunwrap::Stage::Packing => "packing",
            uvunwrap::Stage::Output => "output",
            _ => "unknown",
        };
        Progress {
            stage,
            done: p.done,
            total: p.total,
        }
    }
}

/// Where one instance of a mesh lives: its page and the `min`/`max` corners
/// of its rectangle in page coordinates.
#[pyclass(frozen, get_all, module = "uvunwrap")]
struct InstanceRect {
    page: u32,
    min: (f32, f32),
    max: (f32, f32),
}

/// An unwrapped mesh, as `array.array`s: per output vertex, `xref` (its
/// input vertex), `uv` (u and v), `chart` and `page` (-1 for none); `indices`
/// (three per triangle, same triangles as the input); `corner_uvs`, u and v
/// per triangle corner (what per-corner UV maps such as Blender's store);
/// and `instances`.
#[pyclass(frozen, get_all, module = "uvunwrap")]
struct OutputMesh {
    xref: Py<PyAny>,
    uv: Py<PyAny>,
    chart: Py<PyAny>,
    page: Py<PyAny>,
    indices: Py<PyAny>,
    corner_uvs: Py<PyAny>,
    instances: Vec<Py<InstanceRect>>,
}

/// An `array.array` of `typecode` holding `values`.
fn array<T: Copy, const N: usize>(
    py: Python<'_>,
    typecode: &str,
    values: impl IntoIterator<Item = T>,
    bytes: impl Fn(T) -> [u8; N],
) -> PyResult<Py<PyAny>> {
    let data: Vec<u8> = values.into_iter().flat_map(bytes).collect();
    Ok(py
        .import("array")?
        .getattr("array")?
        .call1((typecode, PyBytes::new(py, &data)))?
        .unbind())
}

impl OutputMesh {
    fn new(py: Python<'_>, m: &uvunwrap::OutputMesh) -> PyResult<Self> {
        let v = &m.vertices;
        let id = |x: Option<u32>| x.map_or(-1, |x| x as i32);
        let corner_uvs = m.indices.iter().flat_map(|&i| v[i as usize].uv);
        Ok(OutputMesh {
            xref: array(py, "I", v.iter().map(|v| v.xref), u32::to_ne_bytes)?,
            uv: array(py, "f", v.iter().flat_map(|v| v.uv), f32::to_ne_bytes)?,
            chart: array(py, "i", v.iter().map(|v| id(v.chart)), i32::to_ne_bytes)?,
            page: array(py, "i", v.iter().map(|v| id(v.page)), i32::to_ne_bytes)?,
            indices: array(py, "I", m.indices.iter().copied(), u32::to_ne_bytes)?,
            corner_uvs: array(py, "f", corner_uvs, f32::to_ne_bytes)?,
            instances: m
                .instances
                .iter()
                .map(|r| {
                    Py::new(
                        py,
                        InstanceRect {
                            page: r.page,
                            min: (r.min[0], r.min[1]),
                            max: (r.max[0], r.max[1]),
                        },
                    )
                })
                .collect::<PyResult<_>>()?,
        })
    }
}

/// The generated atlas: page size in texels (`width`, `height`),
/// `page_count`, the `texels_per_unit` used, `chart_count`, `utilization`
/// (fraction of the atlas covered) and one `OutputMesh` per input mesh.
#[pyclass(frozen, get_all, module = "uvunwrap")]
struct Atlas {
    width: u32,
    height: u32,
    page_count: u32,
    texels_per_unit: f32,
    chart_count: u32,
    utilization: f32,
    meshes: Vec<Py<OutputMesh>>,
}

#[pymethods]
impl Atlas {
    fn __repr__(&self) -> String {
        format!(
            "Atlas({}x{} x{} page(s), {} charts, {:.1}% utilization)",
            self.width,
            self.height,
            self.page_count,
            self.chart_count,
            self.utilization * 100.0
        )
    }
}

impl Atlas {
    fn new(py: Python<'_>, a: &uvunwrap::Atlas) -> PyResult<Self> {
        Ok(Atlas {
            width: a.width,
            height: a.height,
            page_count: a.page_count,
            texels_per_unit: a.texels_per_unit,
            chart_count: a.chart_count,
            utilization: a.utilization,
            meshes: a
                .meshes
                .iter()
                .map(|m| Py::new(py, OutputMesh::new(py, m)?))
                .collect::<PyResult<_>>()?,
        })
    }
}

/// Unwraps `meshes` into one shared atlas, in the calling thread (other
/// Python threads keep running). See `start` for a background job.
#[pyfunction]
#[pyo3(signature = (
    meshes, *, page_size=Some(1024), texels_per_unit=None, padding=2, packing="shape",
    max_chart_angle=70.0, max_area_distortion=4.0, allow_rotation=true,
    max_chart_size=0, block_align=false,
))]
#[allow(clippy::too_many_arguments)]
fn generate(
    py: Python<'_>,
    meshes: Vec<PyRef<'_, Mesh>>,
    page_size: Option<u32>,
    texels_per_unit: Option<f32>,
    padding: u32,
    packing: &str,
    max_chart_angle: f32,
    max_area_distortion: f32,
    allow_rotation: bool,
    max_chart_size: u32,
    block_align: bool,
) -> PyResult<Atlas> {
    let opts = options(
        page_size,
        texels_per_unit,
        padding,
        packing,
        max_chart_angle,
        max_area_distortion,
        allow_rotation,
        max_chart_size,
        block_align,
    )?;
    let data: Vec<Arc<MeshData>> = meshes.iter().map(|m| m.0.clone()).collect();
    let atlas = py.detach(|| {
        let inputs: Vec<uvunwrap::MeshInput> = data.iter().map(|d| d.input()).collect();
        uvunwrap::generate(&inputs, &opts)
    });
    Atlas::new(py, &atlas.map_err(to_py)?)
}

/// What a background job and its thread share.
#[derive(Default)]
struct Shared {
    progress: Mutex<Option<uvunwrap::Progress>>,
    cancel: AtomicBool,
    finished: AtomicBool,
}

/// An unwrap running in a background thread (see `start`).
#[pyclass(frozen, module = "uvunwrap")]
struct Job {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<Result<uvunwrap::Atlas, uvunwrap::Error>>>>,
    outcome: Mutex<Option<Result<uvunwrap::Atlas, uvunwrap::Error>>>,
}

#[pymethods]
impl Job {
    /// The latest progress report, or `None` before the first.
    #[getter]
    fn progress(&self) -> Option<Progress> {
        let p = *self
            .shared
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        p.map(Progress::from)
    }

    /// Whether the job has finished (successfully or not).
    #[getter]
    fn done(&self) -> bool {
        self.shared.finished.load(Ordering::Acquire)
    }

    /// Asks the job to stop; `result()` then raises `Cancelled`.
    fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }

    /// Waits for the job (at most `timeout` seconds, if given; raises
    /// `TimeoutError` after that) and returns its atlas.
    #[pyo3(signature = (timeout=None))]
    fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Atlas> {
        let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
        let finished = py.detach(|| loop {
            if self.shared.finished.load(Ordering::Acquire) {
                return true;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        });
        if !finished {
            return Err(PyTimeoutError::new_err("the job is still running"));
        }
        let mut outcome = self.outcome.lock().unwrap_or_else(|e| e.into_inner());
        if outcome.is_none() {
            let thread = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
            let joined = py.detach(|| thread.map(JoinHandle::join));
            *outcome = Some(match joined {
                Some(Ok(result)) => result,
                _ => return Err(PyRuntimeError::new_err("the unwrap thread panicked")),
            });
        }
        match outcome.as_ref().expect("set above") {
            Ok(atlas) => Atlas::new(py, atlas),
            Err(e) => Err(to_py(e.clone())),
        }
    }
}

/// Starts unwrapping `meshes` in a background thread and returns its `Job`,
/// for UIs that report progress and allow cancelling. Takes the same
/// options as `generate`.
#[pyfunction]
#[pyo3(signature = (
    meshes, *, page_size=Some(1024), texels_per_unit=None, padding=2, packing="shape",
    max_chart_angle=70.0, max_area_distortion=4.0, allow_rotation=true,
    max_chart_size=0, block_align=false,
))]
#[allow(clippy::too_many_arguments)]
fn start(
    meshes: Vec<PyRef<'_, Mesh>>,
    page_size: Option<u32>,
    texels_per_unit: Option<f32>,
    padding: u32,
    packing: &str,
    max_chart_angle: f32,
    max_area_distortion: f32,
    allow_rotation: bool,
    max_chart_size: u32,
    block_align: bool,
) -> PyResult<Job> {
    let opts = options(
        page_size,
        texels_per_unit,
        padding,
        packing,
        max_chart_angle,
        max_area_distortion,
        allow_rotation,
        max_chart_size,
        block_align,
    )?;
    let data: Vec<Arc<MeshData>> = meshes.iter().map(|m| m.0.clone()).collect();
    let shared = Arc::new(Shared::default());
    let state = shared.clone();
    let thread = std::thread::spawn(move || {
        let inputs: Vec<uvunwrap::MeshInput> = data.iter().map(|d| d.input()).collect();
        let result = uvunwrap::generate_with_progress(&inputs, &opts, |p| {
            *state.progress.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
            !state.cancel.load(Ordering::Relaxed)
        });
        state.finished.store(true, Ordering::Release);
        result
    });
    Ok(Job {
        shared,
        thread: Mutex::new(Some(thread)),
        outcome: Mutex::new(None),
    })
}

#[pymodule]
#[pyo3(name = "uvunwrap")]
fn uvunwrap_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("UnwrapError", m.py().get_type::<UnwrapError>())?;
    m.add("Cancelled", m.py().get_type::<Cancelled>())?;
    m.add_class::<Mesh>()?;
    m.add_class::<Atlas>()?;
    m.add_class::<OutputMesh>()?;
    m.add_class::<InstanceRect>()?;
    m.add_class::<Progress>()?;
    m.add_class::<Job>()?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(start, m)?)?;
    Ok(())
}
