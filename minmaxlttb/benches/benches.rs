use criterion::{criterion_group, criterion_main, Criterion};
use minmaxlttb::{lttb as lttb_fn, Binning, LttbBuilder, LttbMethod, Point};
use std::hint::black_box;

fn make_series(n: usize) -> Vec<Point> {
    // Generate a simple deterministic waveform outside the benchmark to avoid RNG cost in the hot path
    (0..n)
        .map(|i| {
            let x = i as f64;
            let y = (i as f64 * 0.001).sin() + (i as f64 * 0.0001).cos();
            Point::new(x, y)
        })
        .collect()
}

fn bench_minmaxlttb_50m(c: &mut Criterion) {
    let n = 50_000_000usize;
    let threshold = 2_000usize;
    let ratio = 30usize;
    let data = make_series(n);

    // Use the public builder to avoid relying on internal function signatures
    let sampler = LttbBuilder::new()
        .threshold(threshold)
        .method(LttbMethod::MinMax)
        .ratio(ratio)
        .build();

    c.bench_function("minmaxlttb_50M", |b| {
        b.iter(|| {
            let _ = black_box(sampler.downsample(black_box(&data)));
        })
    });
}

fn bench_lttb_50m(c: &mut Criterion) {
    let n = 50_000_000usize;
    let threshold = 2_000usize;
    let data = make_series(n);

    c.bench_function("lttb_50M", |b| {
        b.iter(|| {
            let _ = lttb_fn(black_box(&data), black_box(threshold), Binning::ByRange);
        })
    });
}

criterion_group!(benches, bench_minmaxlttb_50m, bench_lttb_50m);
criterion_main!(benches);
