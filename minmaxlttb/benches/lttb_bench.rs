use criterion::{criterion_group, criterion_main, Criterion};
use lttb as external_lttb;
use minmaxlttb::{lttb as local_lttb, Point};
use std::hint::black_box;

fn generate_data(n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| Point::new(i as f64, (i as f64).sin()))
        .collect()
}

fn bench_lttb(c: &mut Criterion) {
    let data = generate_data(10_000);
    let threshold = 500;

    c.bench_function("lttb (local)", |b| {
        b.iter(|| {
            let _ = local_lttb(black_box(&data), black_box(threshold));
        })
    });

    // Convert to external_lttb::DataPoint for external crate
    let ext_data: Vec<external_lttb::DataPoint> = data
        .iter()
        .map(|p| external_lttb::DataPoint::new(p.x(), p.y()))
        .collect();

    c.bench_function("lttb (jeromefroe/lttb-rs)", |b| {
        b.iter(|| {
            let _ = external_lttb::lttb(black_box(ext_data.clone()), black_box(threshold));
        })
    });
}

criterion_group!(benches, bench_lttb);
criterion_main!(benches);
