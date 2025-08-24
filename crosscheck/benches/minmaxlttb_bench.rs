use criterion::{criterion_group, criterion_main, Criterion};
use minmaxlttb::{minmaxlttb as local_minmaxlttb, Point};
use std::hint::black_box;

fn generate_data(n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| Point::new(i as f64, (i as f64).sin()))
        .collect()
}

fn bench_minmaxlttb(c: &mut Criterion) {
    let data = generate_data(10_000_000);
    let threshold = 20_000;
    let ratio = 8;

    // Prepare x/y slices for upstream implementation using x-ranges
    let xs: Vec<f64> = data.iter().map(|p| p.x()).collect();
    let ys: Vec<f64> = data.iter().map(|p| p.y()).collect();

    c.bench_function("minmaxlttb (local)", |b| {
        b.iter(|| {
            let _ = local_minmaxlttb(black_box(&data), black_box(threshold), black_box(ratio));
        })
    });

    c.bench_function("minmaxlttb (tsdownsample)", |b| {
        b.iter(|| {
            let _ = downsample_rs::minmaxlttb_with_x(
                black_box(&xs),
                black_box(&ys),
                black_box(threshold),
                black_box(ratio),
            );
        })
    });
}

criterion_group!(benches, bench_minmaxlttb);
criterion_main!(benches);
