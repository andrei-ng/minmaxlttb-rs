use lttb::{lttb as external_lttb, DataPoint};
use minmaxlttb::{lttb as local_lttb, Point};
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

fn is_peak(y: &[f64], i: usize) -> bool {
    i > 0 && i + 1 < y.len() && y[i] > y[i - 1] && y[i] > y[i + 1]
}

fn is_valley(y: &[f64], i: usize) -> bool {
    i > 0 && i + 1 < y.len() && y[i] < y[i - 1] && y[i] < y[i + 1]
}

fn find_peaks_and_valleys(y: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut peaks = Vec::new();
    let mut valleys = Vec::new();
    for i in 1..y.len() - 1 {
        if is_peak(y, i) {
            peaks.push(i);
        } else if is_valley(y, i) {
            valleys.push(i);
        }
    }
    (peaks, valleys)
}

fn main() {
    let n = 1000;
    let threshold = 100;
    let series = generate_random_series(n);

    let local_impl = local_lttb(&series, threshold);
    let external_impl = external_lttb(to_datapoints(&series), threshold);

    // Check length
    let ok_len = local_impl.len() == external_impl.len();
    // Check start/end
    let ok_start = (local_impl[0].x() - external_impl[0].x).abs() < 1e-8
        && (local_impl[0].y() - external_impl[0].y).abs() < 1e-8;
    let ok_end = (local_impl.last().unwrap().x() - external_impl.last().unwrap().x).abs() < 1e-8
        && (local_impl.last().unwrap().y() - external_impl.last().unwrap().y).abs() < 1e-8;

    // Check peaks/valleys
    let local_y: Vec<f64> = local_impl.iter().map(|p| p.y()).collect();
    let external_y: Vec<f64> = external_impl.iter().map(|p| p.y).collect();

    let (local_peaks, local_valleys) = find_peaks_and_valleys(&local_y);
    let (external_peaks, ext_valleys) = find_peaks_and_valleys(&external_y);

    let ok_peaks = local_peaks == external_peaks;
    let ok_valleys = local_valleys == ext_valleys;

    println!("LTTB comparison (local vs lttb-rs):");
    println!(
        "  Length: {} (local: {}, ext: {})",
        ok_len,
        local_impl.len(),
        external_impl.len()
    );
    println!("  Start: {}", ok_start);
    println!("  End: {}", ok_end);
    println!(
        "  Peaks: {} \n   local: {:?}\n  extern: {:?}",
        ok_peaks, local_peaks, external_peaks
    );
    println!(
        "  Valleys: {}\n   local: {:?}\n  extern: {:?}",
        ok_valleys, local_valleys, ext_valleys
    );

    if ok_len && ok_start && ok_end && ok_peaks && ok_valleys {
        println!("SUCCESS: Both implementations match on all criteria.");
    } else {
        println!("FAILURE: Mismatch detected. See details above.");
    }
}
