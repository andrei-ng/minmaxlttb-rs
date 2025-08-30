use minmaxlttb::{minmaxlttb, Point};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn generate_random_series(n: usize) -> Vec<Point> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut y = 0.0;
    (0..n)
        .map(|i| {
            y += rng.random_range(-1.0..1.0);
            Point::new(i as f64, y)
        })
        .collect()
}

#[test]
fn compare_with_tsdownsample_minmaxlttb() {
    let n = 10_000usize;
    let threshold = 1_000usize;
    let ratio = 2usize;
    let series = generate_random_series(n);

    let ours = minmaxlttb(&series, threshold, ratio).unwrap();
    let our_indices: Vec<usize> = ours.iter().map(|p| p.x().round() as usize).collect();

    let x: Vec<i32> = (0..n as i32).collect();
    let y: Vec<f32> = series.iter().map(|p| p.y() as f32).collect();
    let theirs = downsample_rs::minmaxlttb_with_x(&x, &y, threshold, ratio);

    assert_eq!(
        our_indices, theirs,
        "Index selection differs from tsdownsample"
    );
}
