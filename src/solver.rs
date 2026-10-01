//! Sparse least squares via CGLS (conjugate gradient on the normal equations,
//! without ever forming `AᵀA`).

/// Row-compressed sparse matrix.
pub(crate) struct SparseRows {
    pub row_ptr: Vec<usize>,
    pub cols: Vec<u32>,
    pub vals: Vec<f64>,
    pub ncols: usize,
}

impl SparseRows {
    pub fn new(ncols: usize) -> Self {
        SparseRows { row_ptr: vec![0], cols: Vec::new(), vals: Vec::new(), ncols }
    }

    pub fn push(&mut self, col: usize, val: f64) {
        self.cols.push(col as u32);
        self.vals.push(val);
    }

    pub fn end_row(&mut self) {
        self.row_ptr.push(self.cols.len());
    }

    pub fn nrows(&self) -> usize {
        self.row_ptr.len() - 1
    }

    /// `out = A x`
    fn mul(&self, x: &[f64], out: &mut [f64]) {
        for (r, o) in out.iter_mut().enumerate() {
            let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
            *o = self.cols[s..e].iter().zip(&self.vals[s..e]).map(|(&c, &v)| v * x[c as usize]).sum();
        }
    }

    /// `out = Aᵀ y`
    fn mul_t(&self, y: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        for (r, &yr) in y.iter().enumerate() {
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                out[self.cols[k] as usize] += self.vals[k] * yr;
            }
        }
    }
}

/// Least squares solve of `A x ≈ b`. Uses a sparse Cholesky factorization of
/// the normal equations when the `direct-solver` feature is on (falling back to
/// CGLS if factorization fails), else CGLS warm-started from `x`.
pub(crate) fn least_squares(a: &mut SparseRows, b: &[f64], x: &mut [f64]) {
    #[cfg(feature = "direct-solver")]
    if cholesky(a, b, x) {
        return;
    }
    let max_iter = (a.ncols * 2 + 100).min(5000);
    cgls(a, b, x, max_iter, 1e-9);
}

/// Solves `AᵀA x = Aᵀb` by sparse Cholesky. Returns `false` if `AᵀA` isn't
/// numerically positive definite.
#[cfg(feature = "direct-solver")]
fn cholesky(a: &SparseRows, b: &[f64], x: &mut [f64]) -> bool {
    use faer::linalg::solvers::Solve;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, Side};

    let n = a.ncols;
    let mut triplets = Vec::with_capacity(a.vals.len() * 4);
    let mut atb = Mat::<f64>::zeros(n, 1);
    for r in 0..a.nrows() {
        let (s, e) = (a.row_ptr[r], a.row_ptr[r + 1]);
        for i in s..e {
            let (ci, vi) = (a.cols[i] as usize, a.vals[i]);
            atb[(ci, 0)] += vi * b[r];
            for j in s..e {
                let cj = a.cols[j] as usize;
                if ci >= cj {
                    triplets.push(Triplet::new(ci, cj, vi * a.vals[j]));
                }
            }
        }
    }
    let Ok(m) = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets) else {
        return false;
    };
    let Ok(llt) = m.sp_cholesky(Side::Lower) else {
        return false;
    };
    llt.solve_in_place(atb.as_mut());
    if (0..n).any(|i| !atb[(i, 0)].is_finite()) {
        return false;
    }
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = atb[(i, 0)];
    }
    true
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimizes `|A x - b|²` starting from `x`, with Jacobi (column-norm)
/// preconditioning: solves for `y = D x` on `A D⁻¹`, then maps back. Returns
/// the number of iterations used.
pub(crate) fn cgls(a: &mut SparseRows, b: &[f64], x: &mut [f64], max_iter: usize, tol: f64) -> usize {
    let mut norm = vec![0.0; a.ncols];
    for (&c, &v) in a.cols.iter().zip(&a.vals) {
        norm[c as usize] += v * v;
    }
    norm.iter_mut().for_each(|n| *n = if *n > 0.0 { n.sqrt() } else { 1.0 });
    for (&c, v) in a.cols.iter().zip(a.vals.iter_mut()) {
        *v /= norm[c as usize];
    }
    x.iter_mut().zip(&norm).for_each(|(xi, n)| *xi *= n);
    let iters = cgls_raw(a, b, x, max_iter, tol);
    x.iter_mut().zip(&norm).for_each(|(xi, n)| *xi /= n);
    iters
}

/// Unpreconditioned CGLS. Stops when the normal-equation residual has dropped
/// by `tol` relative to the start, or after `max_iter`.
fn cgls_raw(a: &SparseRows, b: &[f64], x: &mut [f64], max_iter: usize, tol: f64) -> usize {
    let (m, n) = (a.nrows(), a.ncols);
    let mut r = vec![0.0; m];
    a.mul(x, &mut r);
    r.iter_mut().zip(b).for_each(|(ri, bi)| *ri = bi - *ri);
    let mut s = vec![0.0; n];
    a.mul_t(&r, &mut s);
    let mut p = s.clone();
    let mut q = vec![0.0; m];
    let mut gamma = dot(&s, &s);
    let stop = gamma * tol * tol;
    for it in 0..max_iter {
        if gamma <= stop || gamma == 0.0 {
            return it;
        }
        a.mul(&p, &mut q);
        let qq = dot(&q, &q);
        if qq == 0.0 {
            return it;
        }
        let alpha = gamma / qq;
        x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
        r.iter_mut().zip(&q).for_each(|(ri, qi)| *ri -= alpha * qi);
        a.mul_t(&r, &mut s);
        let gamma_new = dot(&s, &s);
        let beta = gamma_new / gamma;
        p.iter_mut().zip(&s).for_each(|(pi, si)| *pi = si + beta * *pi);
        gamma = gamma_new;
    }
    max_iter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_overdetermined_system() {
        // x + y = 3, x - y = 1, 2x = 4  ->  x = 2, y = 1
        let mut a = SparseRows::new(2);
        for row in [[(0, 1.0), (1, 1.0)], [(0, 1.0), (1, -1.0)], [(0, 2.0), (1, 0.0)]] {
            for (c, v) in row {
                a.push(c, v);
            }
            a.end_row();
        }
        let mut x = vec![0.0; 2];
        cgls(&mut a, &[3.0, 1.0, 4.0], &mut x, 50, 1e-12);
        assert!((x[0] - 2.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9);
    }
}
