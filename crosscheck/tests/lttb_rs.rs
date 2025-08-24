use lttb::{lttb as external_lttb, DataPoint};
use minmaxlttb::{lttb as local_lttb, Binning, Point};
use rand::Rng;

fn generate_random_series(n: usize) -> Vec<Point> {
    let mut rng = rand::rng();
    let mut y = 0.0;
    (0..n)
        .map(|i| {
            y += rng.random_range(-1.0..1.0);
            Point::new(i as f64, y)
        })
        .collect()
}

fn to_datapoints(points: &[Point]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.x(), p.y()))
        .collect()
}

#[test]
fn compare_lttb_rs_crate() {
    const EPS_TOL: f64 = 1e-8;

    let n = 10_000;
    let threshold = 1_000;
    let series = generate_random_series(n);

    let local_impl = local_lttb(&series, threshold, Binning::ByCount).unwrap();
    let external_impl = external_lttb(to_datapoints(&series), threshold);

    assert_eq!(
        local_impl.len(),
        external_impl.len(),
        "downsampling result has different lengths"
    );

    assert!(
        (local_impl[0].x() - external_impl[0].x).abs() < EPS_TOL,
        "start point has different x values"
    );

    assert!(
        (local_impl[0].y() - external_impl[0].y).abs() < EPS_TOL,
        "start point has different y values"
    );

    assert!(
        (local_impl.last().unwrap().x() - external_impl.last().unwrap().x).abs() < EPS_TOL,
        "end point has different x values"
    );

    assert!(
        (local_impl.last().unwrap().y() - external_impl.last().unwrap().y).abs() < EPS_TOL,
        "end point has different y values"
    );

    for (our_p, their_p) in local_impl.iter().zip(external_impl) {
        assert_eq!(our_p.x(), their_p.x);
        assert_eq!(our_p.y(), their_p.y);
    }
}
