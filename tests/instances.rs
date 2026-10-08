//! Instanced meshes: one layout, one atlas rectangle per instance.

mod common;
use common::*;

#[test]
fn instances_get_their_own_rectangles() {
    let (t, c) = (torus(32, 16), cube());
    let scales = [1.0, 2.0, 0.5];
    let one = [1.0];
    for opts in [
        Options::default(),
        Options { texels_per_unit: Some(12.0), resolution: 192, ..Options::default() },
        Options { packing: Packing::Rect, block_align: true, ..Options::default() },
    ] {
        let inputs = [MeshInput { instances: &scales, ..t.input() }, c.input(), MeshInput { instances: &one, ..c.input() }];
        // `check` maps every instance into the atlas and checks them all
        // together with the plain cube: valid, padded, non-overlapping.
        let atlas = run_inputs(&inputs, &opts);
        let (torus_out, cube_out, single) = (&atlas.meshes[0], &atlas.meshes[1], &atlas.meshes[2]);
        assert_eq!((torus_out.instances.len(), cube_out.instances.len(), single.instances.len()), (3, 0, 1));
        // Local UVs fill [0, 1] (up to padding) on both axes.
        for out in [torus_out, single] {
            for k in 0..2 {
                let lo = out.vertices.iter().map(|v| v.uv[k]).fold(f32::MAX, f32::min);
                let hi = out.vertices.iter().map(|v| v.uv[k]).fold(f32::MIN, f32::max);
                assert!((0.0..0.05).contains(&lo) && hi <= 1.0 && hi > 0.95, "local uv range {lo}..{hi}");
            }
        }
        // Rectangles scale with the instance scale.
        let size = |r: InstanceRect| ((r.max[0] - r.min[0]) * atlas.width as f32, (r.max[1] - r.min[1]) * atlas.height as f32);
        let base = size(torus_out.instances[0]);
        for (i, &k) in scales.iter().enumerate() {
            let (w, h) = size(torus_out.instances[i]);
            assert!((w / base.0 - k).abs() < 1e-3 && (h / base.1 - k).abs() < 1e-3, "instance {i}: {w}x{h} vs {base:?}");
        }
        if opts.resolution == 192 {
            assert!(atlas.page_count > 1, "expected several pages");
        }
    }
    let bad = MeshInput { instances: &[1.0, 0.0], ..c.input() };
    assert!(matches!(generate(&[bad], &Options::default()), Err(Error::InvalidScale { mesh: 0 })));
}

#[test]
fn instance_layouts_respect_the_page() {
    let c = cube();
    let opts = |tpu| Options { texels_per_unit: Some(tpu), resolution: 128, ..Options::default() };
    run_inputs(&[MeshInput { instances: &[1.0, 0.5], ..c.input() }], &opts(12.0));
    // At 20 texels/unit the six 44-texel faces can't share one 128 rectangle
    // (the plain cube only fits by spreading over two pages). The error must
    // report the most compact layout, not a 44 x 264 column.
    let result = generate(&[MeshInput { instances: &[1.0], ..c.input() }], &opts(20.0));
    let Err(Error::ChartTooLarge { width, height, .. }) = result else { panic!("{result:?}") };
    assert!(width.max(height) <= 140, "{width} x {height}");
}

#[test]
fn instance_scale_ratios_fit_or_fail_cleanly() {
    let t = torus(32, 16);
    // A 10x ratio wastes space (the largest instance gets 10x the padding) but
    // fits.
    run_inputs(&[MeshInput { instances: &[1.0, 0.1], ..t.input() }], &Options::default());
    // At 10^4 the largest instance's padding alone exceeds the page.
    let start = std::time::Instant::now();
    let result = generate(&[MeshInput { instances: &[1.0, 1e-4], ..t.input() }], &Options::default());
    assert!(matches!(result, Err(Error::ChartTooLarge { .. })), "{result:?}");
    assert!(start.elapsed().as_secs_f64() < 2.0, "took {:?}", start.elapsed());
}

#[test]
fn max_chart_size_applies_to_instances() {
    let t = torus(32, 16);
    let opts = Options { texels_per_unit: Some(100.0), resolution: 0, max_chart_size: 16, ..Options::default() };
    let atlas = run_inputs(&[MeshInput { instances: &[1.0, 2.0], ..t.input() }], &opts);
    let out = &atlas.meshes[0];
    for i in 0..out.instances.len() {
        let m = instance_mesh(out, i);
        let mut extent: std::collections::HashMap<u32, ([f32; 2], [f32; 2])> = std::collections::HashMap::new();
        for v in &m.vertices {
            let p = [v.uv[0] * atlas.width as f32, v.uv[1] * atlas.height as f32];
            let (lo, hi) = extent.entry(v.chart.unwrap()).or_insert((p, p));
            *lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            *hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }
        for (chart, (lo, hi)) in extent {
            let side = (hi[0] - lo[0]).max(hi[1] - lo[1]);
            assert!(side <= 16.01, "instance {i} chart {chart}: {side} texels");
        }
    }
}

#[test]
fn instances_of_repacked_uvs() {
    let m = soup(&plane(12));
    let right = right_of(&m, 1.0);
    let uvs: Vec<[f32; 2]> = m.positions.iter().enumerate().map(|(i, p)| [p[0] + if right[i / 3] { 5.0 } else { 0.0 }, p[1]]).collect();
    let atlas = run_inputs(&[MeshInput { uvs: Some(&uvs), instances: &[1.0, 0.5], ..m.input() }], &Options::default());
    assert_eq!((atlas.chart_count, atlas.meshes[0].instances.len()), (2, 2));
}
