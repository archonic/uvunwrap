//! Chart segmentation by cost-driven region growing, after xatlas.
//!
//! Faces are first grouped into planar regions (connected, exactly coplanar
//! faces), which are the units of growth, so flat areas are never cut. Charts
//! are grown one at a time from the largest unassigned region, always
//! absorbing the cheapest neighbor, where the cost combines:
//!
//! - **normal deviation** from the chart's average normal (and a hard limit of
//!   `max_angle`, so every chart can at least be planar-projected),
//! - **straightness**: a bonus for regions that fill a notch in the boundary,
//!   which keeps boundaries smooth and charts compact,
//! - **roundness**: a small penalty for making the chart less round,
//! - **creases**: a penalty for growing across sharp edges, so seams follow
//!   creases instead of cutting through smooth areas.
//!
//! Finally, small charts mostly enclosed by a neighbor are merged into it.

use crate::geom::{add3, dot3, len3, normalize3, scale3, sub3, V3};
use crate::topology::{Topology, NONE};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const NORMAL_DEVIATION_WEIGHT: f64 = 2.0;
const ROUNDNESS_WEIGHT: f64 = 0.01;
const STRAIGHTNESS_WEIGHT: f64 = 6.0;
const CREASE_WEIGHT: f64 = 4.0;
const MAX_COST: f64 = 2.0;
/// Rounds of normal smoothing before segmentation.
const SMOOTHING_ROUNDS: usize = 8;

struct Candidate {
    cost: f64,
    region: u32,
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
        other
            .cost
            .total_cmp(&self.cost)
            .then(other.region.cmp(&self.region))
    }
}

/// A shared boundary between two regions (or charts).
#[derive(Clone, Copy)]
struct Link {
    other: u32,
    /// Total length of the shared edges.
    len: f64,
    /// Sum of `len * (1 - cos dihedral)` over the shared edges.
    crease: f64,
}

/// Connected groups of coplanar faces.
struct Regions {
    of_face: Vec<u32>,
    faces: Vec<Vec<u32>>,
    normal: Vec<V3>,
    area: Vec<f64>,
    perimeter: Vec<f64>,
    links: Vec<Vec<Link>>,
}

fn edge_len(topo: &Topology, f: usize, e: usize) -> f64 {
    let face = topo.faces[f];
    len3(sub3(
        topo.positions[face[(e + 1) % 3] as usize],
        topo.positions[face[e] as usize],
    ))
}

/// Face normals smoothed by a few rounds of area-weighted averaging with
/// edge neighbors, never across creases. Scanned meshes have noisy normals;
/// growing charts on the raw ones gives comb-like boundaries where faces
/// alternate in and out of the normal cone.
fn smoothed_normals(topo: &Topology, rounds: usize) -> Vec<V3> {
    let cos_crease = 3f64.sqrt() / 2.0; // cos 30°
    let smooth_neighbors: Vec<[u32; 3]> = (0..topo.faces.len())
        .map(|f| {
            topo.neighbors[f].map(|g| {
                if g != NONE && dot3(topo.normals[f], topo.normals[g as usize]) >= cos_crease {
                    g
                } else {
                    NONE
                }
            })
        })
        .collect();
    let mut n: Vec<V3> = topo
        .normals
        .iter()
        .zip(&topo.areas)
        .map(|(&n, &a)| scale3(n, a))
        .collect();
    for _ in 0..rounds {
        n = (0..n.len())
            .map(|f| {
                let mut s = n[f];
                for &g in &smooth_neighbors[f] {
                    if g != NONE {
                        s = add3(s, n[g as usize]);
                    }
                }
                s
            })
            .collect();
    }
    n.iter()
        .enumerate()
        .map(|(f, &v)| normalize3(v).unwrap_or(topo.normals[f]))
        .collect()
}

/// Builds the planar regions of `subset` (faces outside it count as
/// boundary), using `smooth` normals for the region normals.
fn build_regions(topo: &Topology, smooth: &[V3], subset: &[u32]) -> Regions {
    // Faces whose normals differ by less than ~0.25° count as coplanar.
    const COPLANAR: f64 = 1.0 - 1e-5;
    let nf = topo.faces.len();
    // NONE: outside the subset; PENDING: in it but not assigned yet.
    const PENDING: u32 = NONE - 1;
    let mut of_face = vec![NONE; nf];
    for &f in subset {
        of_face[f as usize] = PENDING;
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut stack = Vec::new();
    for &seed in subset {
        let seed = seed as usize;
        if of_face[seed] != PENDING {
            continue;
        }
        let id = faces.len() as u32;
        let n = topo.normals[seed];
        let mut members = vec![seed as u32];
        of_face[seed] = id;
        stack.push(seed as u32);
        while let Some(f) = stack.pop() {
            for &g in &topo.neighbors[f as usize] {
                if g != NONE
                    && of_face[g as usize] == PENDING
                    && dot3(topo.normals[g as usize], n) >= COPLANAR
                {
                    of_face[g as usize] = id;
                    members.push(g);
                    stack.push(g);
                }
            }
        }
        faces.push(members);
    }

    let nr = faces.len();
    let mut normal = vec![[0.0; 3]; nr];
    let mut area = vec![0.0; nr];
    let mut perimeter = vec![0.0; nr];
    let mut links: Vec<Vec<Link>> = vec![Vec::new(); nr];
    for (r, members) in faces.iter().enumerate() {
        let mut nsum = [0.0; 3];
        for &f in members {
            let f = f as usize;
            nsum = add3(nsum, scale3(smooth[f], topo.areas[f]));
            area[r] += topo.areas[f];
            for e in 0..3 {
                let g = topo.neighbors[f][e];
                let other = if g == NONE { NONE } else { of_face[g as usize] };
                if other == r as u32 {
                    continue;
                }
                let len = edge_len(topo, f, e);
                perimeter[r] += len;
                if other != NONE {
                    let crease =
                        len * (1.0 - dot3(topo.normals[f], topo.normals[g as usize])).max(0.0);
                    match links[r].iter_mut().find(|l| l.other == other) {
                        Some(l) => {
                            l.len += len;
                            l.crease += crease;
                        }
                        None => links[r].push(Link { other, len, crease }),
                    }
                }
            }
        }
        normal[r] = normalize3(nsum).unwrap_or(topo.normals[members[0] as usize]);
    }
    Regions {
        of_face,
        faces,
        normal,
        area,
        perimeter,
        links,
    }
}

#[derive(Clone, Copy)]
struct ChartState {
    nsum: V3,
    area: f64,
    boundary: f64,
}

impl ChartState {
    fn normal(&self) -> V3 {
        normalize3(self.nsum).unwrap_or([0.0, 0.0, 1.0])
    }
}

/// Cost of adding region `r` to chart `c`, or `None` if it isn't allowed.
fn cost(
    reg: &Regions,
    chart_of: &[u32],
    c: u32,
    st: &ChartState,
    r: usize,
    cos_max: f64,
) -> Option<f64> {
    let d = dot3(reg.normal[r], st.normal());
    if d < cos_max {
        return None;
    }
    let (mut l_in, mut crease) = (0.0, 0.0);
    for l in &reg.links[r] {
        if chart_of[l.other as usize] == c {
            l_in += l.len;
            crease += l.crease;
        }
    }
    if l_in <= 0.0 {
        return None;
    }
    let l_out = reg.perimeter[r] - l_in;
    let new_area = st.area + reg.area[r];
    let new_boundary = (st.boundary + l_out - l_in).max(0.0);
    let roundness = if st.boundary > 0.0 && new_boundary > 0.0 {
        1.0 - (st.boundary * st.boundary / st.area) / (new_boundary * new_boundary / new_area)
    } else {
        0.0
    };
    let straightness = ((l_out - l_in) / (l_out + l_in)).min(0.0);
    let c = NORMAL_DEVIATION_WEIGHT * (1.0 - d)
        + ROUNDNESS_WEIGHT * roundness
        + STRAIGHTNESS_WEIGHT * straightness
        + CREASE_WEIGHT * crease / l_in;
    (c <= MAX_COST).then_some(c)
}

/// Connected components of the non-excluded faces (the UV islands, for a mesh
/// with input UVs), each in ascending face order.
pub(crate) fn components(topo: &Topology) -> Vec<Vec<u32>> {
    let mut seen: Vec<bool> = topo.excluded.clone();
    let mut out = Vec::new();
    for seed in 0..topo.faces.len() {
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        let mut members = vec![seed as u32];
        let mut i = 0;
        while i < members.len() {
            for &g in &topo.neighbors[members[i] as usize] {
                if g != NONE && !seen[g as usize] {
                    seen[g as usize] = true;
                    members.push(g);
                }
            }
            i += 1;
        }
        members.sort_unstable();
        out.push(members);
    }
    out
}

/// Splits the non-excluded faces into connected charts whose (smoothed)
/// face normals lie within `max_angle` radians of the chart's average normal.
pub(crate) fn segment(topo: &Topology, max_angle: f64) -> Vec<Vec<u32>> {
    let smooth = smoothed_normals(topo, SMOOTHING_ROUNDS);
    let all: Vec<u32> = (0..topo.faces.len() as u32)
        .filter(|&f| !topo.excluded[f as usize])
        .collect();
    segment_regions(topo, &build_regions(topo, &smooth, &all), &all, max_angle)
}

fn segment_regions(
    topo: &Topology,
    reg: &Regions,
    subset: &[u32],
    max_angle: f64,
) -> Vec<Vec<u32>> {
    let cos_max = crate::geom::cos(max_angle);
    let nr = reg.faces.len();
    let mut chart_of = vec![NONE; nr];
    let mut order: Vec<u32> = (0..nr as u32).collect();
    order.sort_by(|&a, &b| {
        reg.area[b as usize]
            .total_cmp(&reg.area[a as usize])
            .then(a.cmp(&b))
    });

    let mut charts: Vec<Vec<u32>> = Vec::new();
    let mut states: Vec<ChartState> = Vec::new();
    let mut heap = BinaryHeap::new();
    for &seed in &order {
        if chart_of[seed as usize] != NONE {
            continue;
        }
        let c = charts.len() as u32;
        let s = seed as usize;
        chart_of[s] = c;
        let mut st = ChartState {
            nsum: scale3(reg.normal[s], reg.area[s]),
            area: reg.area[s],
            boundary: reg.perimeter[s],
        };
        let mut members = vec![seed];
        heap.clear();
        let push =
            |heap: &mut BinaryHeap<Candidate>, chart_of: &[u32], st: &ChartState, r: usize| {
                for l in &reg.links[r] {
                    let o = l.other as usize;
                    if chart_of[o] == NONE {
                        if let Some(cost) = cost(reg, chart_of, c, st, o, cos_max) {
                            heap.push(Candidate {
                                cost,
                                region: o as u32,
                            });
                        }
                    }
                }
            };
        push(&mut heap, &chart_of, &st, s);
        while let Some(Candidate {
            cost: queued,
            region,
        }) = heap.pop()
        {
            let r = region as usize;
            if chart_of[r] != NONE {
                continue;
            }
            // The chart changed since this was queued: re-evaluate lazily.
            let Some(now) = cost(reg, &chart_of, c, &st, r, cos_max) else {
                continue;
            };
            if now > queued + 1e-9 {
                heap.push(Candidate { cost: now, region });
                continue;
            }
            let l_in: f64 = reg.links[r]
                .iter()
                .filter(|l| chart_of[l.other as usize] == c)
                .map(|l| l.len)
                .sum();
            chart_of[r] = c;
            members.push(region);
            st.nsum = add3(st.nsum, scale3(reg.normal[r], reg.area[r]));
            st.area += reg.area[r];
            st.boundary = (st.boundary + reg.perimeter[r] - 2.0 * l_in).max(0.0);
            push(&mut heap, &chart_of, &st, r);
        }
        charts.push(members);
        states.push(st);
    }

    merge_charts(reg, &mut chart_of, &mut charts, &mut states, cos_max);

    let mut face_chart: Vec<u32> = reg
        .of_face
        .iter()
        .map(|&r| {
            if r == NONE {
                NONE
            } else {
                chart_of[r as usize]
            }
        })
        .collect();
    smooth_boundaries(topo, reg, subset, &mut face_chart, &states, cos_max);

    let mut out: Vec<Vec<u32>> = vec![Vec::new(); charts.len()];
    for &f in subset {
        let c = face_chart[f as usize];
        if c != NONE {
            out[c as usize].push(f);
        }
    }
    out.retain(|c| !c.is_empty());
    out
}

/// Moves single-face teeth to the chart they stick into: a face with at most
/// one edge neighbor in its own chart and at least two in another one. Such a
/// face is a leaf of its chart, so moving it never disconnects the chart.
fn smooth_boundaries(
    topo: &Topology,
    reg: &Regions,
    subset: &[u32],
    face_chart: &mut [u32],
    states: &[ChartState],
    cos_max: f64,
) {
    let normals: Vec<V3> = states.iter().map(|s| s.normal()).collect();
    for _ in 0..4 {
        let mut moved = false;
        for &f in subset {
            let f = f as usize;
            let own = face_chart[f];
            let r = reg.of_face[f];
            if own == NONE || reg.faces[r as usize].len() != 1 {
                continue;
            }
            let nb = topo.neighbors[f].map(|g| {
                if g == NONE {
                    NONE
                } else {
                    face_chart[g as usize]
                }
            });
            let own_count = nb.iter().filter(|&&c| c == own).count();
            if own_count > 1 {
                continue;
            }
            let other = if nb[0] != own && nb[0] != NONE && (nb[0] == nb[1] || nb[0] == nb[2]) {
                nb[0]
            } else if nb[1] != own && nb[1] != NONE && nb[1] == nb[2] {
                nb[1]
            } else {
                continue;
            };
            if dot3(reg.normal[r as usize], normals[other as usize]) >= cos_max {
                face_chart[f] = other;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}

/// Merges each small chart into the neighbor it shares most of its boundary
/// with, if the merged chart still satisfies the normal cone.
fn merge_charts(
    reg: &Regions,
    chart_of: &mut [u32],
    charts: &mut [Vec<u32>],
    states: &mut [ChartState],
    cos_max: f64,
) {
    let total_area: f64 = states.iter().map(|s| s.area).sum();
    let live = charts.len();
    if live < 2 {
        return;
    }
    let small = total_area / live as f64 * 0.25;
    loop {
        let mut order: Vec<usize> = (0..charts.len())
            .filter(|&c| !charts[c].is_empty())
            .collect();
        order.sort_by(|&a, &b| states[a].area.total_cmp(&states[b].area).then(a.cmp(&b)));
        let mut merged_any = false;
        for c in order {
            if charts[c].is_empty() || states[c].boundary <= 0.0 {
                continue;
            }
            // Shared boundary length with each neighboring chart.
            let mut shared: Vec<(u32, f64)> = Vec::new();
            for &r in &charts[c] {
                for l in &reg.links[r as usize] {
                    let o = chart_of[l.other as usize];
                    if o != c as u32 {
                        match shared.iter_mut().find(|s| s.0 == o) {
                            Some(s) => s.1 += l.len,
                            None => shared.push((o, l.len)),
                        }
                    }
                }
            }
            let Some(&(n, len)) = shared
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            else {
                continue;
            };
            let ratio = len / states[c].boundary;
            let worth = ratio >= 0.5
                || (states[c].area < small && ratio >= 0.25)
                || states[c].area < small * 0.1;
            if !worth {
                continue;
            }
            let n = n as usize;
            let nsum = add3(states[c].nsum, states[n].nsum);
            let Some(axis) = normalize3(nsum) else {
                continue;
            };
            let fits = |members: &[u32]| {
                members
                    .iter()
                    .all(|&r| dot3(reg.normal[r as usize], axis) >= cos_max)
            };
            if !fits(&charts[c]) || !fits(&charts[n]) {
                continue;
            }
            let moved = std::mem::take(&mut charts[c]);
            for &r in &moved {
                chart_of[r as usize] = n as u32;
            }
            charts[n].extend(moved);
            states[n] = ChartState {
                nsum,
                area: states[n].area + states[c].area,
                boundary: (states[n].boundary + states[c].boundary - 2.0 * len).max(0.0),
            };
            merged_any = true;
        }
        if !merged_any {
            break;
        }
    }
}
