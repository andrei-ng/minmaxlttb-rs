use csv::ReaderBuilder;
use minmaxlttb::{LttbBuilder, Point};
use plotly::{Layout, Plot, Scatter};
use std::error::Error;

const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/timeseries.csv");

fn load_timeseries_data(path: &str) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let x: f64 = record.get(0).unwrap().parse()?;
        let y: f64 = record.get(1).unwrap().parse()?;
        data.push(Point::new(x, y));
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = load_timeseries_data(DATA_PATH)?;
    let threshold = 500;

    let standard = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::Standard)
        .build()
        .downsample(&data.clone());
    let minmax_ratio_2 = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::MinMax)
        .ratio(2)
        .build()
        .downsample(&data.clone());
    let minmax_ratio_8 = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::MinMax)
        .ratio(8)
        .build()
        .downsample(&data.clone());
    let minmax_ratio_16 = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::MinMax)
        .ratio(16)
        .build()
        .downsample(&data);

    println!("Original points: {}", data.len());
    println!("Standard LTTB: {} points", standard.len());
    println!("MinMax LTTB (ratio=2): {} points", minmax_ratio_2.len());
    println!("MinMax LTTB (ratio=8): {} points", minmax_ratio_8.len());
    println!("MinMax LTTB (ratio=16): {} points", minmax_ratio_16.len());

    let mut plot = Plot::new();
    let x_orig: Vec<f64> = data.iter().map(|p| p.x()).collect();
    let y_orig: Vec<f64> = data.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_orig, y_orig)
            .name("Original")
            .line(plotly::common::Line::new().color("lightgray").width(1.2)),
    );

    let x_std: Vec<f64> = standard.iter().map(|p| p.x()).collect();
    let y_std: Vec<f64> = standard.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_std, y_std)
            .name("Standard LTTB")
            .line(plotly::common::Line::new().color("blue").width(2.0)),
    );

    let x_mm_2: Vec<f64> = minmax_ratio_2.iter().map(|p| p.x()).collect();
    let y_mm_2: Vec<f64> = minmax_ratio_2.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_mm_2, y_mm_2)
            .name("MinMax LTTB (ratio=2)")
            .line(plotly::common::Line::new().color("red").width(2.0)),
    );

    let x_mm_8: Vec<f64> = minmax_ratio_8.iter().map(|p| p.x()).collect();
    let y_mm_8: Vec<f64> = minmax_ratio_8.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_mm_8, y_mm_8)
            .name("MinMax LTTB (ratio=8)")
            .line(plotly::common::Line::new().color("green").width(2.0)),
    );

    let x_mm_16: Vec<f64> = minmax_ratio_16.iter().map(|p| p.x()).collect();
    let y_mm_16: Vec<f64> = minmax_ratio_16.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_mm_16, y_mm_16)
            .name("MinMax LTTB (ratio=16)")
            .line(plotly::common::Line::new().color("purple").width(2.0)),
    );

    let layout = Layout::new()
        .title(plotly::common::Title::with_text(
            "LTTB vs MinMaxLTTB Comparison on timeseries.csv",
        ))
        .height(900)
        .show_legend(true)
        .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::with_text("Time")))
        .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::with_text("Value")));
    plot.set_layout(layout);

    let out_dir = "output";
    std::fs::create_dir_all(out_dir)?;
    let out_path = format!("{out_dir}/compare_minmax_vs_standard.html");
    plot.write_html(&out_path);
    println!("Plot saved to {out_path}");
    plot.show_html(out_path);
    Ok(())
}
