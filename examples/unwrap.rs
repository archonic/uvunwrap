//! Unwraps an OBJ file and writes the result as OBJ (with `vt`) plus an SVG
//! drawing of each atlas page.
//!
//! ```text
//! cargo run --release --example unwrap -- [input.obj] [output_prefix]
//! ```
//!
//! Without arguments, a built-in torus is unwrapped into `out/torus*`.

use std::fmt::Write as _;
use std::time::Instant;
use std::{env, fs};

fn load_obj(text: &str) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).map(|s| s.parse().unwrap_or(0.0)).collect();
                positions.push([c[0], c[1], c[2]]);
            }
            Some("f") => {
                // "f a/b/c ..." with 1-based or negative indices; fan-triangulate polygons.
                let n = positions.len() as i64;
                let poly: Vec<u32> = it
                    .filter_map(|t| t.split('/').next()?.parse::<i64>().ok())
                    .map(|i| if i < 0 { n + i } else { i - 1 } as u32)
                    .collect();
                for k in 1..poly.len().saturating_sub(1) {
                    indices.extend([poly[0], poly[k], poly[k + 1]]);
                }
            }
            _ => {}
        }
    }
    (positions, indices)
}

fn torus() -> (Vec<[f32; 3]>, Vec<u32>) {
    let (nu, nv) = (64u32, 32u32);
    let mut positions = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let (a, b) = (i as f32 / nu as f32 * std::f32::consts::TAU, j as f32 / nv as f32 * std::f32::consts::TAU);
            let r = 1.0 + 0.4 * b.cos();
            positions.push([r * a.cos(), r * a.sin(), 0.4 * b.sin()]);
        }
    }
    let id = |i: u32, j: u32| (j % nv) * nu + (i % nu);
    let mut indices = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            indices.extend([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    (positions, indices)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let (positions, indices, prefix) = match args.first() {
        Some(path) => {
            let (p, i) = load_obj(&fs::read_to_string(path).expect("read input"));
            let prefix = args.get(1).cloned().unwrap_or_else(|| path.trim_end_matches(".obj").to_string() + "_unwrapped");
            (p, i, prefix)
        }
        None => {
            let (p, i) = torus();
            fs::create_dir_all("out").unwrap();
            (p, i, "out/torus".to_string())
        }
    };

    let start = Instant::now();
    let mesh = uvunwrap::MeshInput { positions: &positions, indices: &indices };
    let atlas = uvunwrap::generate(&[mesh], &uvunwrap::Options::default()).expect("unwrap failed");
    let elapsed = start.elapsed();
    let out = &atlas.meshes[0];
    println!(
        "{} triangles -> {} charts, {} vertices ({} in), {}x{} x{} page(s), {:.1}% utilization, {:.1} ms",
        indices.len() / 3,
        atlas.chart_count,
        out.vertices.len(),
        positions.len(),
        atlas.width,
        atlas.height,
        atlas.page_count,
        atlas.utilization * 100.0,
        elapsed.as_secs_f64() * 1000.0
    );

    let mut obj = String::new();
    for v in &out.vertices {
        let p = positions[v.xref as usize];
        writeln!(obj, "v {} {} {}", p[0], p[1], p[2]).unwrap();
    }
    for v in &out.vertices {
        writeln!(obj, "vt {} {}", v.uv[0], 1.0 - v.uv[1]).unwrap(); // OBJ's v points up
    }
    for t in out.indices.chunks_exact(3) {
        let [a, b, c] = [t[0] + 1, t[1] + 1, t[2] + 1];
        writeln!(obj, "f {a}/{a} {b}/{b} {c}/{c}").unwrap();
    }
    fs::write(format!("{prefix}.obj"), obj).unwrap();

    for page in 0..atlas.page_count {
        let scale = 1024.0 / atlas.width.max(atlas.height) as f32;
        let (w, h) = (atlas.width as f32 * scale, atlas.height as f32 * scale);
        let mut svg = format!(
            "<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>\
             <rect width='100%' height='100%' fill='#1b1b1f'/>"
        );
        for t in out.indices.chunks_exact(3) {
            let vs = [0, 1, 2].map(|k| out.vertices[t[k] as usize]);
            let (Some(chart), true) = (vs[0].chart, vs[0].page == page) else { continue };
            let hue = (chart as f32 * 137.508) % 360.0; // golden-angle palette
            let pts: Vec<String> =
                vs.iter().map(|v| format!("{:.2},{:.2}", v.uv[0] * atlas.width as f32 * scale, v.uv[1] * atlas.height as f32 * scale)).collect();
            write!(svg, "<polygon points='{}' fill='hsl({hue:.0},65%,60%)' stroke='hsl({hue:.0},65%,35%)' stroke-width='0.3'/>", pts.join(" ")).unwrap();
        }
        svg.push_str("</svg>");
        fs::write(format!("{prefix}_page{page}.svg"), svg).unwrap();
    }
    println!("wrote {prefix}.obj and {} SVG page(s)", atlas.page_count);
}
