//! Unwraps an OBJ file and writes the result as OBJ (with `vt`) plus an SVG
//! drawing of each atlas page.
//!
//! ```text
//! cargo run --release --example unwrap -- [--fast] [--flat] [--block-align]
//!     [--texels-per-unit X] [--max-chart-size N] [--padding N] [--page-size N]
//!     [--instances 1,2,0.5] [--keep-uvs] [input.obj] [output_prefix]
//! ```
//!
//! Vertices are split per distinct `v/vt/vn` corner, as an engine import
//! would, so UV seams and hard edges reach the unwrapper. `--flat` gives every
//! triangle its own vertices instead (flat shading). Quads and n-gons stay
//! whole: each OBJ face is passed as one polygon. `--keep-uvs` repacks the
//! OBJ's own `vt` islands instead of unwrapping. Without arguments, a
//! built-in torus is unwrapped into `out/torus*`.

mod common;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;
use std::{env, fs};

/// Positions, triangle indices, each triangle's polygon (its `f` line) and
/// (if complete) per-vertex UVs.
type Obj = (Vec<[f32; 3]>, Vec<u32>, Vec<u32>, Option<Vec<[f32; 2]>>);

/// Loads an OBJ with one vertex per distinct `v/vt/vn` corner, plus each
/// vertex's `vt` (if every vertex has one). Quads and n-gons are
/// fan-triangulated, remembering which polygon each triangle came from.
fn load_obj(text: &str) -> Obj {
    let mut v: Vec<[f32; 3]> = Vec::new();
    let mut vt: Vec<[f32; 2]> = Vec::new();
    let mut uvs: Vec<Option<[f32; 2]>> = Vec::new();
    let nn = &mut 0i64;
    let mut corners: HashMap<[i64; 3], u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut polygons = Vec::new();
    let mut face_count = 0u32;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).map(|s| s.parse().unwrap_or(0.0)).collect();
                v.push([c[0], c[1], c[2]]);
            }
            Some("vt") => {
                let c: Vec<f32> = it.take(2).map(|s| s.parse().unwrap_or(0.0)).collect();
                vt.push([c[0], 1.0 - c.get(1).copied().unwrap_or(0.0)]); // OBJ's v points up
            }
            Some("vn") => *nn += 1,
            Some("f") => {
                // "f a/b/c ..." with 1-based or negative indices; fan-triangulate polygons.
                let polygon = face_count;
                face_count += 1;
                let counts = [v.len() as i64, vt.len() as i64, *nn];
                let poly: Vec<u32> = it
                    .filter_map(|t| {
                        let mut key = [-1i64; 3];
                        for (k, part) in t.split('/').take(3).enumerate() {
                            if let Ok(i) = part.parse::<i64>() {
                                key[k] = if i < 0 { counts[k] + i } else { i - 1 };
                            }
                        }
                        let p = *v.get(usize::try_from(key[0]).ok()?)?;
                        let uv = usize::try_from(key[1])
                            .ok()
                            .and_then(|i| vt.get(i).copied());
                        Some(*corners.entry(key).or_insert_with(|| {
                            positions.push(p);
                            uvs.push(uv);
                            (positions.len() - 1) as u32
                        }))
                    })
                    .collect();
                for k in 1..poly.len().saturating_sub(1) {
                    indices.extend([poly[0], poly[k], poly[k + 1]]);
                    polygons.push(polygon);
                }
            }
            _ => {}
        }
    }
    let uvs = uvs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .filter(|u| !u.is_empty());
    (positions, indices, polygons, uvs)
}

/// Gives every triangle its own three vertices.
fn flatten(positions: &[[f32; 3]], indices: &[u32]) -> (Vec<[f32; 3]>, Vec<u32>) {
    (
        indices.iter().map(|&i| positions[i as usize]).collect(),
        (0..indices.len() as u32).collect(),
    )
}

fn main() {
    let mut opts = uvunwrap::Options::default();
    let mut args: Vec<String> = env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--fast") {
        opts.packing = uvunwrap::Packing::Fast;
        args.remove(i);
    }
    let flat = args
        .iter()
        .position(|a| a == "--flat")
        .map(|i| args.remove(i))
        .is_some();
    let keep_uvs = args
        .iter()
        .position(|a| a == "--keep-uvs")
        .map(|i| args.remove(i))
        .is_some();
    opts.block_align = args
        .iter()
        .position(|a| a == "--block-align")
        .map(|i| args.remove(i))
        .is_some();
    let mut value = |name: &str| {
        let i = args.iter().position(|a| a == name)?;
        args.remove(i);
        Some(args.remove(i))
    };
    let texels_per_unit: Option<f32> =
        value("--texels-per-unit").map(|v| v.parse().expect("--texels-per-unit"));
    if let Some(v) = value("--max-chart-size") {
        opts.max_chart_size = v.parse().expect("--max-chart-size");
    }
    let instances: Vec<f32> = value("--instances")
        .map(|v| {
            v.split(',')
                .map(|k| k.parse().expect("--instances"))
                .collect()
        })
        .unwrap_or_default();
    let page_size: Option<u32> = value("--page-size").map(|v| v.parse().expect("--page-size"));
    opts.density = match texels_per_unit {
        Some(texels_per_unit) => uvunwrap::Density::Fixed {
            texels_per_unit,
            page_size,
        },
        None => uvunwrap::Density::Auto {
            page_size: page_size.unwrap_or(1024),
        },
    };
    if let Some(v) = value("--padding") {
        opts.padding = v.parse().expect("--padding");
    }
    let (positions, indices, polygons, uvs, prefix) = match args.first() {
        Some(path) => {
            let (p, i, poly, uv) = load_obj(&fs::read_to_string(path).expect("read input"));
            let prefix = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| path.trim_end_matches(".obj").to_string() + "_unwrapped");
            (p, i, Some(poly), uv, prefix)
        }
        None => {
            let (p, i) = common::torus();
            fs::create_dir_all("out").unwrap();
            (p, i, None, None, "out/torus".to_string())
        }
    };

    let (positions, indices) = if flat {
        flatten(&positions, &indices)
    } else {
        (positions, indices)
    };
    let uvs = if keep_uvs && !flat {
        Some(uvs.expect("--keep-uvs: the OBJ has no vt for every vertex"))
    } else {
        None
    };

    let start = Instant::now();
    let mesh = uvunwrap::MeshInput {
        instance_scales: &instances,
        existing_uvs: uvs.as_deref(),
        polygons: polygons.as_deref(),
        ..uvunwrap::MeshInput::new(&positions, &indices)
    };
    let atlas = uvunwrap::generate(&[mesh], &opts).expect("unwrap failed");
    let elapsed = start.elapsed();
    let out = &atlas.meshes[0];
    println!(
        "{} triangles -> {} charts, {} vertices ({} in), {}x{} x{} page(s), {:.4} texels/unit, {:.1}% utilization, {:.1} ms",
        indices.len() / 3,
        atlas.chart_count,
        out.vertices.len(),
        positions.len(),
        atlas.width,
        atlas.height,
        atlas.page_count,
        atlas.texels_per_unit,
        atlas.utilization * 100.0,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("output hash {:016x}", common::hash(&atlas));

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
        // Draw the mesh once, or once per instance rectangle.
        let views: Vec<Option<uvunwrap::InstanceRect>> = if out.instances.is_empty() {
            vec![None]
        } else {
            out.instances.iter().copied().map(Some).collect()
        };
        for view in views {
            for t in out.indices.chunks_exact(3) {
                let vs = [0, 1, 2].map(|k| out.vertices[t[k] as usize]);
                let Some(chart) = vs[0].chart else { continue };
                let on_page = view.map_or(vs[0].page, |r| Some(r.page));
                if on_page != Some(page) {
                    continue;
                }
                let uv = |uv: [f32; 2]| match view {
                    None => uv,
                    Some(r) => [0, 1].map(|k| r.min[k] + uv[k] * (r.max[k] - r.min[k])),
                };
                let hue = (chart as f32 * 137.508) % 360.0; // golden-angle palette
                let pts: Vec<String> = vs
                    .iter()
                    .map(|v| {
                        let p = uv(v.uv);
                        format!(
                            "{:.2},{:.2}",
                            p[0] * atlas.width as f32 * scale,
                            p[1] * atlas.height as f32 * scale
                        )
                    })
                    .collect();
                write!(svg, "<polygon points='{}' fill='hsl({hue:.0},65%,60%)' stroke='hsl({hue:.0},65%,35%)' stroke-width='0.3'/>", pts.join(" ")).unwrap();
            }
        }
        svg.push_str("</svg>");
        fs::write(format!("{prefix}_page{page}.svg"), svg).unwrap();
    }
    println!("wrote {prefix}.obj and {} SVG page(s)", atlas.page_count);
}
