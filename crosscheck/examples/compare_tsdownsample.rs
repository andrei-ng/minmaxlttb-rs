use csv::ReaderBuilder;
use minmaxlttb::{lttb as lttb_local, LttbBuilder, LttbMethod, Point};
use plotly::{Plot, Scatter};

const DATA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../examples/assets/timeseries.csv"
);

fn load_timeseries_data(path: &str) -> Vec<Point> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("open csv");
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result.expect("read record");
        let x: f64 = record.get(0).unwrap().parse().expect("x parse");
        let y: f64 = record.get(1).unwrap().parse().expect("y parse");
        data.push(Point::new(x, y));
    }
    data
}

fn to_xy(points: &[Point]) -> (Vec<f64>, Vec<f64>) {
    let xs = points.iter().map(|p| p.x()).collect();
    let ys = points.iter().map(|p| p.y()).collect();
    (xs, ys)
}

fn main() {
    let threshold = 500usize;
    let ratio = 8usize;
    let series = load_timeseries_data(DATA_PATH);

    // Our MinMaxLTTB (via builder)
    let sampler = LttbBuilder::new()
        .threshold(threshold)
        .method(LttbMethod::MinMax)
        .ratio(ratio)
        .build();
    let ours_minmax = sampler.downsample(&series).expect("minmaxlttb");

    // Our LTTB (range buckets)
    let ours_lttb = lttb_local(&series, threshold, minmaxlttb::Binning::ByRange).expect("lttb");

    // Upstream tsdownsample
    let x: Vec<f64> = series.iter().map(|p| p.x()).collect();
    let y: Vec<f64> = series.iter().map(|p| p.y()).collect();
    let theirs_minmax_idx = downsample_rs::minmaxlttb_with_x(&x, &y, threshold, ratio);
    let theirs_lttb_idx = downsample_rs::lttb_with_x(&x, &y, threshold);
    let theirs_minmax: Vec<Point> = theirs_minmax_idx
        .iter()
        .map(|&i| Point::new(i as f64, series[i].y()))
        .collect();
    let theirs_lttb: Vec<Point> = theirs_lttb_idx
        .iter()
        .map(|&i| Point::new(i as f64, series[i].y()))
        .collect();

    // Build plot
    let (x_full, y_full) = to_xy(&series);
    let (x_ours_mm, y_ours_mm) = to_xy(&ours_minmax);
    let (x_ours_lttb, y_ours_lttb) = to_xy(&ours_lttb);
    let (x_theirs_mm, y_theirs_mm) = to_xy(&theirs_minmax);
    let (x_theirs_lttb, y_theirs_lttb) = to_xy(&theirs_lttb);

    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(x_full, y_full).name("original").opacity(0.3));
    plot.add_trace(Scatter::new(x_ours_mm, y_ours_mm).name("ours-minmax"));
    plot.add_trace(Scatter::new(x_ours_lttb, y_ours_lttb).name("ours-lttb"));
    plot.add_trace(Scatter::new(x_theirs_mm, y_theirs_mm).name("upstream-minmax"));
    plot.add_trace(Scatter::new(x_theirs_lttb, y_theirs_lttb).name("upstream-lttb"));

    let out_dir = "output";
    std::fs::create_dir_all(out_dir).unwrap();
    let out_path = format!("{out_dir}/compare_result.html");
    plot.write_html(&out_path);
    println!("Plot saved to {out_path}");
    plot.show_html(&out_path);
    println!("Wrote {out_path}");
}
