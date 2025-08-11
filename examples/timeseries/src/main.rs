use minmaxlttb::{LttbBuilder, Point};
use plotly::{Layout, Plot, Scatter};
use std::error::Error;
use std::fs::File;

const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/timeseries.csv");

fn main() -> Result<(), Box<dyn Error>> {
    let original_data = load_timeseries_data()?;

    println!("Loaded timeseries data with {} points", original_data.len());

    let thresholds = vec![100, 500, 1000, 2000];
    let mut downsampled_results = Vec::new();
    for &threshold in &thresholds {
        let downsampled = LttbBuilder::new()
            .threshold(threshold)
            .build()
            .downsample(&original_data);
        println!(
            "LTTB (threshold = {}): downsampled = {} points",
            threshold,
            downsampled.len()
        );
        downsampled_results.push((threshold, downsampled));
    }

    let mut plot = Plot::new();
    let x_orig: Vec<f64> = original_data.iter().map(|p| p.x()).collect();
    let y_orig: Vec<f64> = original_data.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_orig, y_orig)
            .name("Original Data")
            .line(plotly::common::Line::new().color("lightgray").width(1.0)),
    );
    let colors = ["red", "blue", "green", "purple"];
    for (i, (threshold, downsampled)) in downsampled_results.iter().enumerate() {
        let x: Vec<f64> = downsampled.iter().map(|p| p.x()).collect();
        let y: Vec<f64> = downsampled.iter().map(|p| p.y()).collect();
        plot.add_trace(
            Scatter::new(x, y)
                .name(format!("LTTB ({threshold})"))
                .line(plotly::common::Line::new().color(colors[i]).width(2.0)),
        );
    }
    let layout = Layout::new()
        .title(plotly::common::Title::with_text(
            "LTTB Downsampling on timeseries.csv Data",
        ))
        .show_legend(true)
        .height(900)
        .x_axis(
            plotly::layout::Axis::new()
                .title(plotly::common::Title::with_text("Time"))
                .range(vec![0.0, original_data.last().unwrap().x()]),
        )
        .y_axis(
            plotly::layout::Axis::new()
                .title(plotly::common::Title::with_text("Value"))
                .range(vec![
                    original_data
                        .iter()
                        .map(|p| p.y())
                        .fold(f64::INFINITY, f64::min)
                        * 0.95,
                    original_data
                        .iter()
                        .map(|p| p.y())
                        .fold(f64::NEG_INFINITY, f64::max)
                        * 1.05,
                ]),
        );
    plot.set_layout(layout);
    // plot.set_configuration(Configuration::default().responsive(true).fill_frame(true));

    let out_dir = "./output";
    std::fs::create_dir_all(out_dir).unwrap();

    let out_path = format!("{out_dir}/lttb_timeseries_visualization.html");
    plot.write_html(&out_path);
    println!("Plot saved as {out_dir}/lttb_timeseries_visualization.html");
    plot.show_html(out_path);

    Ok(())
}

fn load_timeseries_data() -> Result<Vec<Point>, Box<dyn Error>> {
    let file = File::open(DATA_PATH)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let x: f64 = record.get(0).ok_or("Missing X column")?.parse()?;
        let y: f64 = record.get(1).ok_or("Missing Y column")?.parse()?;
        data.push(Point::new(x, y));
    }
    Ok(data)
}
