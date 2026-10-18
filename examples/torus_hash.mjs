// Runs the `torus_hash` example's wasm module and prints its output hash in
// the `unwrap` example's format: node torus_hash.mjs path/to/torus_hash.wasm
import { readFileSync } from "node:fs";

const { instance } = await WebAssembly.instantiate(readFileSync(process.argv[2]));
const hash = BigInt.asUintN(64, instance.exports.torus_hash());
console.log(`output hash ${hash.toString(16).padStart(16, "0")}`);
