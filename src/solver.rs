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
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::sparse::linalg::cholesky::{factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering};
    use faer::sparse::linalg::SupernodalThreshold;
    use faer::sparse::{SparseColMat, SymbolicSparseColMat};
    use faer::{Conj, MatMut, Par, Side};

    let (lower, mut atb) = normal_equations(a, b);
    let n = a.ncols;
    let symbolic = SymbolicSparseColMat::<usize>::new_checked(n, n, lower.col_ptr, None, lower.row_idx);
    let m = SparseColMat::<usize, f64>::new(symbolic, lower.vals);
    // Supernodal is faster than simplicial on LSCM systems of every size we
    // see, including small ones, so don't let faer choose.
    let params = CholeskySymbolicParams {
        supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL,
        ..Default::default()
    };
    let Ok(sym) = factorize_symbolic_cholesky(m.symbolic(), Side::Lower, SymmetricOrdering::Amd, params) else {
        return false;
    };
    let par = Par::Seq;
    let mut values = vec![0.0; sym.len_val()];
    let scratch = sym.factorize_numeric_llt_scratch::<f64>(par, Default::default()).or(sym.solve_in_place_scratch::<f64>(1, par));
    let mut buf = MemBuffer::new(scratch);
    let Ok(llt) = sym.factorize_numeric_llt(&mut values, m.as_ref(), Side::Lower, Default::default(), par, MemStack::new(&mut buf), Default::default())
    else {
        return false;
    };
    llt.solve_in_place_with_conj(Conj::No, MatMut::from_column_major_slice_mut(&mut atb, n, 1), par, MemStack::new(&mut buf));
    if atb.iter().any(|v| !v.is_finite()) {
        return false;
    }
    x.copy_from_slice(&atb);
    true
}

/// Lower triangle of a symmetric matrix in compressed sparse column form,
/// row indices sorted within each column.
#[cfg_attr(not(feature = "direct-solver"), allow(dead_code))]
pub(crate) struct LowerCsc {
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub vals: Vec<f64>,
}

/// Assembles the lower triangle of `AᵀA` and `Aᵀb`, column by column with a
/// dense accumulator (no global sort of `O(nnz²)` triplets).
#[cfg_attr(not(feature = "direct-solver"), allow(dead_code))]
pub(crate) fn normal_equations(a: &SparseRows, b: &[f64]) -> (LowerCsc, Vec<f64>) {
    let n = a.ncols;
    // Aᵀ as column lists of (row, value).
    let mut t_ptr = vec![0usize; n + 1];
    for &c in &a.cols {
        t_ptr[c as usize + 1] += 1;
    }
    for i in 0..n {
        t_ptr[i + 1] += t_ptr[i];
    }
    let mut fill = t_ptr.clone();
    let mut t_row = vec![0u32; a.cols.len()];
    let mut t_val = vec![0.0; a.cols.len()];
    let mut atb = vec![0.0; n];
    for r in 0..a.nrows() {
        for k in a.row_ptr[r]..a.row_ptr[r + 1] {
            let c = a.cols[k] as usize;
            t_row[fill[c]] = r as u32;
            t_val[fill[c]] = a.vals[k];
            fill[c] += 1;
            atb[c] += a.vals[k] * b[r];
        }
    }

    let mut out = LowerCsc { col_ptr: Vec::with_capacity(n + 1), row_idx: Vec::new(), vals: Vec::new() };
    out.col_ptr.push(0);
    let mut acc = vec![0.0; n];
    let mut seen = vec![usize::MAX; n];
    let mut touched: Vec<usize> = Vec::new();
    for j in 0..n {
        touched.clear();
        for p in t_ptr[j]..t_ptr[j + 1] {
            let (r, arj) = (t_row[p] as usize, t_val[p]);
            for k in a.row_ptr[r]..a.row_ptr[r + 1] {
                let i = a.cols[k] as usize;
                if i < j {
                    continue;
                }
                if seen[i] != j {
                    seen[i] = j;
                    acc[i] = 0.0;
                    touched.push(i);
                }
                acc[i] += arj * a.vals[k];
            }
        }
        touched.sort_unstable();
        for &i in &touched {
            out.row_idx.push(i);
            out.vals.push(acc[i]);
        }
        out.col_ptr.push(out.row_idx.len());
    }
    (out, atb)
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
    fn normal_equations_match_dense() {
        // A = [[1, 2, 0], [0, 3, 4], [5, 0, 6], [0, 7, 0]], b = [1, 2, 3, 4]
        let rows: [&[(usize, f64)]; 4] = [&[(0, 1.0), (1, 2.0)], &[(1, 3.0), (2, 4.0)], &[(0, 5.0), (2, 6.0)], &[(1, 7.0)]];
        let mut a = SparseRows::new(3);
        for row in rows {
            for &(c, v) in row {
                a.push(c, v);
            }
            a.end_row();
        }
        let b = [1.0, 2.0, 3.0, 4.0];
        let (lower, atb) = normal_equations(&a, &b);
        let dense = |i: usize, j: usize| -> f64 {
            rows.iter().map(|r| {
                let get = |c| r.iter().find(|e| e.0 == c).map_or(0.0, |e| e.1);
                get(i) * get(j)
            }).sum()
        };
        for j in 0..3 {
            let col: Vec<(usize, f64)> = (lower.col_ptr[j]..lower.col_ptr[j + 1]).map(|k| (lower.row_idx[k], lower.vals[k])).collect();
            let expect: Vec<(usize, f64)> = (j..3).map(|i| (i, dense(i, j))).filter(|e| e.1 != 0.0).collect();
            assert_eq!(col, expect, "column {j}");
        }
        assert_eq!(atb, vec![16.0, 36.0, 26.0]);
    }

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
