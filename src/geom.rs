//! Small f64 vector helpers and 2D computational geometry.
//!
//! Output must be bit-identical on every platform, so nothing here (or
//! anywhere in the crate) calls the platform math library: `sin`, `cos`,
//! `atan2` etc. may differ in the last bit between platforms, while `+ - * /`
//! and `sqrt` are correctly rounded everywhere. [`cos`] and [`angle`] are
//! deterministic replacements built from those.

pub(crate) type V2 = [f64; 2];
pub(crate) type V3 = [f64; 3];

#[inline]
pub(crate) fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn scale3(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn cross3(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn len3(a: V3) -> f64 {
    dot3(a, a).sqrt()
}

/// Normalizes `a`, returning `None` for (near) zero vectors.
#[inline]
pub(crate) fn normalize3(a: V3) -> Option<V3> {
    let l = len3(a);
    (l > 0.0 && l.is_finite()).then(|| scale3(a, 1.0 / l))
}

/// Twice the signed area of a 2D triangle (positive = counter-clockwise).
#[inline]
pub(crate) fn signed_area2(a: V2, b: V2, c: V2) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point.
pub(crate) fn convex_hull(points: &[V2]) -> Vec<V2> {
    let mut pts: Vec<V2> = points.to_vec();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: V2, a: V2, b: V2| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut hull: Vec<V2> = Vec::with_capacity(pts.len() * 2);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev().skip(1) {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Deterministic cosine for `|x| <= π/2` (Taylor series to `x^20`, error
/// below 1e-16).
pub(crate) fn cos(x: f64) -> f64 {
    debug_assert!(x.abs() <= std::f64::consts::FRAC_PI_2 + 1e-12);
    let x2 = x * x;
    // Horner over 1 - x²/2! + x⁴/4! - ..., innermost term first.
    let mut r = 1.0;
    for k in (1..=10).rev() {
        r = 1.0 - x2 / ((2 * k - 1) * (2 * k)) as f64 * r;
    }
    r
}

/// Deterministic arctangent for `|t| <= 1`.
fn atan_unit(t: f64) -> f64 {
    use std::f64::consts::FRAC_PI_6;
    let (sign, t) = if t < 0.0 { (-1.0, -t) } else { (1.0, t) };
    // Reduce to |u| <= tan(π/12) ≈ 0.268 with atan t = π/6 + atan((√3t - 1)/(√3 + t)).
    let sqrt3 = 3f64.sqrt();
    let (base, u) = if t > 0.2679491924311227 { (FRAC_PI_6, (sqrt3 * t - 1.0) / (sqrt3 + t)) } else { (0.0, t) };
    // u - u³/3 + u⁵/5 - ... to u^29 (error below 1e-17).
    let u2 = u * u;
    let mut r = 0.0;
    for k in (0..15).rev() {
        r = 1.0 / (2 * k + 1) as f64 - u2 * r;
    }
    sign * (base + u * r)
}

/// Deterministic `atan2(y, x)` for `y >= 0`, i.e. the angle in `[0, π]`
/// between a vector and the x axis, as needed for the angle between two
/// vectors (`y = |a × b|`, `x = a · b`).
pub(crate) fn angle(y: f64, x: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_2, PI};
    debug_assert!(y >= 0.0);
    if y == 0.0 {
        if x < 0.0 {
            PI
        } else {
            0.0
        }
    } else if x >= y {
        atan_unit(y / x)
    } else if -x >= y {
        PI - atan_unit(y / -x)
    } else {
        FRAC_PI_2 - atan_unit(x / y)
    }
}

/// Unit direction `(cos θ, sin θ)` such that rotating points by `-θ` (see
/// [`rotate`]) minimizes the area of their axis-aligned bounding box. Uses the
/// fact that an optimal rectangle shares an edge with the convex hull.
pub(crate) fn min_area_rect_dir(hull: &[V2]) -> (f64, f64) {
    let mut best = (f64::INFINITY, (1.0, 0.0));
    if hull.len() < 3 {
        return best.1;
    }
    for i in 0..hull.len() {
        let p = hull[i];
        let q = hull[(i + 1) % hull.len()];
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if !(len > 0.0) {
            continue;
        }
        let (c, s) = (dx / len, dy / len);
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for &h in hull {
            let r = rotate(h, c, s);
            for k in 0..2 {
                lo[k] = lo[k].min(r[k]);
                hi[k] = hi[k].max(r[k]);
            }
        }
        let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
        if area < best.0 {
            best = (area, (c, s));
        }
    }
    best.1
}

/// Rotates `p` by `-θ` given `c = cos θ`, `s = sin θ`.
#[inline]
pub(crate) fn rotate(p: V2, c: f64, s: f64) -> V2 {
    [p[0] * c + p[1] * s, -p[0] * s + p[1] * c]
}

/// Separating-axis test for two 2D triangles. Returns `true` only if their
/// interiors overlap by more than `eps`; triangles that merely touch (shared
/// edge or vertex) are not considered overlapping.
pub(crate) fn triangles_overlap(a: &[V2; 3], b: &[V2; 3], eps: f64) -> bool {
    for tri in [a, b] {
        for i in 0..3 {
            let p = tri[i];
            let q = tri[(i + 1) % 3];
            let axis = [p[1] - q[1], q[0] - p[0]];
            let len = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
            if len == 0.0 {
                continue;
            }
            let axis = [axis[0] / len, axis[1] / len];
            let project = |t: &[V2; 3]| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for v in t {
                    let d = v[0] * axis[0] + v[1] * axis[1];
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
                (lo, hi)
            };
            let (alo, ahi) = project(a);
            let (blo, bhi) = project(b);
            if ahi <= blo + eps || bhi <= alo + eps {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_square_with_interior_point() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
        assert_eq!(convex_hull(&pts).len(), 4);
    }

    #[test]
    fn min_rect_of_rotated_square() {
        let a = 0.3f64;
        let (s, c) = a.sin_cos();
        let pts: Vec<V2> = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| [p[0] * c - p[1] * s, p[0] * s + p[1] * c])
            .collect();
        let (c2, s2) = min_area_rect_dir(&convex_hull(&pts));
        let r: Vec<V2> = pts.iter().map(|&p| rotate(p, c2, s2)).collect();
        let w = r.iter().map(|p| p[0]).fold(f64::MIN, f64::max) - r.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
        let h = r.iter().map(|p| p[1]).fold(f64::MIN, f64::max) - r.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
        assert!((w * h - 2.0).abs() < 1e-9);
    }

    #[test]
    fn deterministic_trig_matches_std() {
        for i in 0..=1000 {
            let x = std::f64::consts::FRAC_PI_2 * i as f64 / 1000.0;
            assert!((cos(x) - x.cos()).abs() < 1e-15, "cos({x})");
            for r in [0.5, 1.0, 2.0] {
                let (y, xx) = (x.sin() * r, x.cos() * r);
                let (y2, x2) = ((2.0 * x).sin() * r, (2.0 * x).cos() * r);
                assert!((angle(y, xx) - y.atan2(xx)).abs() < 1e-15, "angle({y}, {xx})");
                assert!((angle(y2.abs(), x2) - y2.abs().atan2(x2)).abs() < 1e-15, "angle({y2}, {x2})");
            }
        }
        assert_eq!(angle(0.0, -1.0), std::f64::consts::PI);
        assert_eq!(angle(0.0, 0.0), 0.0);
    }

    #[test]
    fn overlap_cases() {
        let t1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let shared_edge = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let shared_vertex = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0]];
        let overlapping = [[0.2, 0.2], [1.0, 0.2], [0.2, 1.0]];
        assert!(!triangles_overlap(&t1, &shared_edge, 1e-12));
        assert!(!triangles_overlap(&t1, &shared_vertex, 1e-12));
        assert!(triangles_overlap(&t1, &overlapping, 1e-12));
    }
}
