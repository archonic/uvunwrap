//! Exports `torus_hash() -> u64`: the `unwrap` example's output hash for the
//! built-in torus, as a library, so CI can compare it on targets without a
//! process or stdout (wasm32-unknown-unknown, run by `torus_hash.mjs`).
//!
//! ```text
//! cargo build --release --no-default-features --example torus_hash --target wasm32-unknown-unknown
//! node examples/torus_hash.mjs target/wasm32-unknown-unknown/release/examples/torus_hash.wasm
//! ```

mod common;

/// The output hash of the built-in torus with default options.
#[no_mangle]
pub extern "C" fn torus_hash() -> u64 {
    let (positions, indices) = common::torus();
    let mesh = uvunwrap::MeshInput::new(&positions, &indices);
    let atlas = uvunwrap::generate(&[mesh], &uvunwrap::Options::default()).expect("unwrap failed");
    common::hash(&atlas)
}
