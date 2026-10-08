//! Data parallelism with rayon when the `parallel` feature is on, serial
//! otherwise. Results keep input order either way, so output never depends on
//! the thread count.

/// `items.map(f)`, in input order.
pub(crate) fn map<T, R>(items: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R>
where
    T: Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.into_iter().map(f).collect()
    }
}

/// `items.iter().map(f)`, in input order.
pub(crate) fn map_ref<T, R>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        items.par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter().map(f).collect()
    }
}

/// Concatenation of `f(item)` over `items`, in input order.
pub(crate) fn flat_map<T, R>(items: Vec<T>, f: impl Fn(T) -> Vec<R> + Sync + Send) -> Vec<R>
where
    T: Send,
    R: Send,
{
    map(items, f).into_iter().flatten().collect()
}

/// Runs both closures, potentially in parallel.
pub(crate) fn join<A, B>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}
