//! Chart segmentation by normal-cone region growing.

use crate::geom::{add3, dot3, normalize3, scale3};
use crate::topology::{Topology, NONE};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Candidate {
    cost: f64,
    face: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    // Reversed so that `BinaryHeap` pops the lowest cost first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost).then(other.face.cmp(&self.face))
    }
}

/// Splits the non-degenerate faces into connected charts. Each chart is grown
/// from the largest unassigned face, always absorbing the neighbor whose normal
/// best matches the chart's area-weighted average normal, and rejecting faces
/// that deviate from it by more than `max_angle` radians.
///
/// Keeping every face within a < 90° cone guarantees that each chart can at
/// least be planar-projected without flipping triangles.
pub(crate) fn segment(topo: &Topology, max_angle: f64) -> Vec<Vec<u32>> {
    let cos_max = max_angle.cos();
    let nf = topo.faces.len();
    let mut assigned = vec![false; nf];
    let mut order: Vec<u32> = (0..nf as u32).filter(|&f| !topo.degenerate[f as usize]).collect();
    order.sort_by(|&a, &b| topo.areas[b as usize].total_cmp(&topo.areas[a as usize]).then(a.cmp(&b)));

    let mut charts = Vec::new();
    let mut heap = BinaryHeap::new();
    for &seed in &order {
        if assigned[seed as usize] {
            continue;
        }
        let mut faces = Vec::new();
        let mut nsum = [0.0; 3];
        heap.clear();
        heap.push(Candidate { cost: 0.0, face: seed });
        while let Some(Candidate { face, .. }) = heap.pop() {
            let f = face as usize;
            if assigned[f] {
                continue;
            }
            let n = topo.normals[f];
            if let Some(avg) = normalize3(nsum) {
                if dot3(n, avg) < cos_max {
                    continue; // may still be claimed by a later chart
                }
            }
            assigned[f] = true;
            faces.push(face);
            nsum = add3(nsum, scale3(n, topo.areas[f]));
            let avg = normalize3(nsum).unwrap_or(n);
            for &nb in &topo.neighbors[f] {
                if nb != NONE && !assigned[nb as usize] {
                    let cost = 1.0 - dot3(topo.normals[nb as usize], avg);
                    heap.push(Candidate { cost, face: nb });
                }
            }
        }
        charts.push(faces);
    }
    charts
}
