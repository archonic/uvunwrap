//! Progress reporting and cancellation shared by the pipeline stages.

use crate::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// A pipeline stage, in the order they run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Stage {
    /// Welding and chart segmentation; counts meshes.
    Segmenting,
    /// Flattening charts; counts faces.
    Parameterizing,
    /// Placing charts in the atlas; counts packing trials (total unknown).
    Packing,
    /// Building output meshes; counts meshes.
    Output,
}

/// A progress report passed to the callback of
/// [`generate_with_progress`](crate::generate_with_progress).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Progress {
    /// Current stage.
    pub stage: Stage,
    /// Units of work finished in this stage.
    pub done: usize,
    /// Units of work in this stage, if known (not for packing).
    pub total: Option<usize>,
}

/// A context that reports nothing and is never cancelled.
pub(crate) static QUIET: Ctx<'static> = Ctx::new(None);

/// The callback, the running count of the current stage, and the
/// cancellation flag that parallel work polls.
pub(crate) struct Ctx<'a> {
    callback: Option<&'a (dyn Fn(Progress) -> bool + Sync)>,
    count: Mutex<(Stage, usize)>,
    cancelled: AtomicBool,
}

impl<'a> Ctx<'a> {
    pub const fn new(callback: Option<&'a (dyn Fn(Progress) -> bool + Sync)>) -> Self {
        Ctx {
            callback,
            count: Mutex::new((Stage::Segmenting, 0)),
            cancelled: AtomicBool::new(false),
        }
    }

    /// Adds `by` units to `stage` (resetting the count when the stage changes)
    /// and reports. Calls are serialized, so counts never go backwards.
    pub fn advance(&self, stage: Stage, by: usize, total: Option<usize>) -> Result<(), Error> {
        let Some(callback) = self.callback else {
            return Ok(());
        };
        let mut count = self.count.lock().unwrap_or_else(|e| e.into_inner());
        // Checked under the lock, so nothing is reported after a cancellation.
        if self.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if count.0 != stage {
            *count = (stage, 0);
        }
        count.1 += by;
        if callback(Progress {
            stage,
            done: count.1,
            total,
        }) {
            Ok(())
        } else {
            self.cancelled.store(true, Ordering::Relaxed);
            Err(Error::Cancelled)
        }
    }

    /// Whether the callback asked to stop; long-running work should bail out.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}
