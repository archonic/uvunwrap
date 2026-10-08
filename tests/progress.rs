//! Progress reporting and cancellation.

mod common;
use common::*;

#[test]
fn progress_reports_every_stage_in_order() {
    use std::sync::Mutex;
    use uvunwrap::{generate_with_progress, Stage};
    let (a, b) = (torus(48, 24), cube());
    let inputs = [a.input(), b.input()];
    let log = Mutex::new(Vec::new());
    let atlas = generate_with_progress(&inputs, &Options::default(), |p| {
        log.lock().unwrap().push(p);
        true
    })
    .unwrap();
    assert!(atlas == generate(&inputs, &Options::default()).unwrap(), "progress changed the output");
    let log = log.into_inner().unwrap();
    let order = [Stage::Segmenting, Stage::Parameterizing, Stage::Packing, Stage::Output];
    let rank = |s: Stage| order.iter().position(|&o| o == s).unwrap();
    for w in log.windows(2) {
        assert!(rank(w[0].stage) <= rank(w[1].stage), "stages out of order: {w:?}");
        if w[0].stage == w[1].stage {
            assert!(w[0].done < w[1].done && w[0].total == w[1].total, "count went backwards: {w:?}");
        }
    }
    let last = |s: Stage| *log.iter().rev().find(|p| p.stage == s).expect("stage reported");
    let faces = (a.indices.len() + b.indices.len()) / 3;
    assert_eq!((last(Stage::Segmenting).done, last(Stage::Segmenting).total), (2, 2));
    assert_eq!((last(Stage::Parameterizing).done, last(Stage::Parameterizing).total), (faces, faces));
    assert!(last(Stage::Packing).done >= 1 && last(Stage::Packing).total == 0);
    assert_eq!((last(Stage::Output).done, last(Stage::Output).total), (2, 2));
}

#[test]
fn progress_callback_can_cancel_in_any_stage() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uvunwrap::{generate_with_progress, Stage};
    let m = torus(48, 24);
    for stage in [Stage::Segmenting, Stage::Parameterizing, Stage::Packing, Stage::Output] {
        let refusals = AtomicUsize::new(0);
        let result = generate_with_progress(&[m.input()], &Options::default(), |p| {
            assert_eq!(refusals.load(Ordering::Relaxed), 0, "reported after cancelling");
            let go_on = p.stage != stage;
            if !go_on {
                refusals.fetch_add(1, Ordering::Relaxed);
            }
            go_on
        });
        assert!(matches!(result, Err(Error::Cancelled)), "{stage:?}");
        assert_eq!(refusals.load(Ordering::Relaxed), 1);
    }
    // Cancelling partway through parameterization stops promptly.
    let calls = AtomicUsize::new(0);
    let result = generate_with_progress(&[m.input()], &Options::default(), |p| {
        let n = calls.fetch_add(1, Ordering::Relaxed);
        !(p.stage == Stage::Parameterizing && n >= 3)
    });
    assert!(matches!(result, Err(Error::Cancelled)));
}
