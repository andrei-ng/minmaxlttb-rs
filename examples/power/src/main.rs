use chrono::{DateTime, NaiveDateTime};
use csv::ReaderBuilder;
use csv::StringRecord;
use minmaxlttb::{LttbBuilder, Point};
use num_format::{Locale, ToFormattedString};
use plotly::{common::DashType, Configuration, Layout, Plot, Scatter};
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

const DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip";
const DATA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../assets/household_power_consumption.txt"
);

fn main() -> Result<(), Box<dyn Error>> {
    download_data_file()?;

    let t0 = Instant::now();
    let original_data = load_power_consumption_data()?;
    let elapsed = t0.elapsed();
    println!(
        "Loaded power consumption data with {} points in {:.3?}",
        original_data.len().to_formatted_string(&Locale::en),
        elapsed
    );

    let thresholds = [100, 500, 1000, 2000, 5000, 10_000, 25000];
    let mut downsampled_results = Vec::new();

    for &threshold in &thresholds {
        let t0 = Instant::now();
        let downsampled = LttbBuilder::new()
            .threshold(threshold)
            .ratio(8)
            .build()
            .downsample(&original_data)
            .unwrap();
        let elapsed = t0.elapsed();
        println!(
            "LTTB (threshold={}): original={} points,downsampled={} points, runtime {:.3?}",
            threshold,
            original_data.len().to_formatted_string(&Locale::en),
            downsampled.len().to_formatted_string(&Locale::en),
            elapsed
        );
        downsampled_results.push((threshold, downsampled));
    }

    let mut plot = Plot::new();

    // Add original data (downsampled for visualization)
    let original_viz = LttbBuilder::new()
        .threshold(500000)
        .build()
        .downsample(&original_data)
        .unwrap();
    let x_orig: Vec<f64> = original_viz.iter().map(|p| p.x()).collect();
    let y_orig: Vec<f64> = original_viz.iter().map(|p| p.y()).collect();

    plot.add_trace(
        Scatter::new(x_orig, y_orig)
            .web_gl_mode(true)
            .name("Power Data (downsampled to 500k points)")
            .line(
                plotly::common::Line::new()
                    .color("lightgray")
                    .width(1.5)
                    .dash(DashType::Dash),
            ),
    );

    // Add downsampled traces with different colors
    let colors = ["red", "blue", "green", "purple", "orange", "pink", "brown"];
    for (i, (threshold, downsampled)) in downsampled_results.iter().enumerate() {
        let x: Vec<f64> = downsampled.iter().map(|p| p.x()).collect();
        let y: Vec<f64> = downsampled.iter().map(|p| p.y()).collect();

        plot.add_trace(
            Scatter::new(x, y)
                .name(format!("MinMaxLTTB ({threshold}, ratio=8)"))
                .line(plotly::common::Line::new().color(colors[i]).width(1.5)),
        );
    }

    let layout = Layout::new()
        .title(plotly::common::Title::with_text(
            "LTTB Downsampling on Real Household Power Consumption Data,<br>Original data: 2.04M points</br>",
        ))
        .show_legend(true)
        .height(900)
        .x_axis(plotly::layout::Axis::new()
            .title(plotly::common::Title::with_text("Time (hours)"))
            .range(vec![0.0, original_viz.last().unwrap().x()]))
        .y_axis(plotly::layout::Axis::new()
            .title(plotly::common::Title::with_text("Power (kW)"))
            .range(vec![
                original_viz.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min) * 0.95,
                original_viz.iter().map(|p| p.y()).fold(f64::NEG_INFINITY, f64::max) * 1.05
            ]));
    plot.set_layout(layout);
    plot.set_configuration(Configuration::default().responsive(true));

    let out_dir = "./output";
    std::fs::create_dir_all(out_dir).unwrap();

    let out_path = format!("{out_dir}/lttb_power_consumption_visualization.html");
    plot.write_html(&out_path);
    println!("Plot saved as {out_dir}/lttb_power_consumption_visualization.html");
    plot.show_html(out_path);

    println!("\n=== Real Power Consumption Data Statistics ===");
    println!("Original data points: {}", original_data.len());
    println!(
        "Data span: {:.1} hours",
        original_data.last().unwrap().x() - original_data.first().unwrap().x()
    );
    println!(
        "Power range: {:.1} - {:.1} kW",
        original_data
            .iter()
            .map(|p| p.y())
            .fold(f64::INFINITY, f64::min),
        original_data
            .iter()
            .map(|p| p.y())
            .fold(f64::NEG_INFINITY, f64::max)
    );

    for (threshold, downsampled) in &downsampled_results {
        let compression_ratio = original_data.len() as f64 / downsampled.len() as f64;
        println!(
            "LTTB (threshold={}): original={} points, downsampled={} points, compression ratio: {:.1}x",
            threshold,
            original_data.len().to_formatted_string(&Locale::en),
            downsampled.len().to_formatted_string(&Locale::en),
            compression_ratio
        );
    }

    Ok(())
}

fn download_data_file() -> Result<(), Box<dyn Error>> {
    if Path::new(DATA_PATH).exists() {
        return Ok(());
    }
    println!("Downloading dataset from {DATA_URL}...");
    let resp = reqwest::blocking::get(DATA_URL)?;
    let bytes = resp.bytes()?;
    let reader = Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(reader)?;
    let mut file = zip.by_name("household_power_consumption.txt")?;
    std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/../assets"))?;
    let mut out = std::fs::File::create(DATA_PATH)?;
    std::io::copy(&mut file, &mut out)?;
    println!("Dataset downloaded and extracted to {DATA_PATH}");
    Ok(())
}

fn load_power_consumption_data() -> Result<Vec<Point>, Box<dyn Error>> {
    let file = File::open(DATA_PATH)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_reader(file);

    // Collect all records first
    let records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;

    // Parallel parse
    let mut data: Vec<Point> = records
        .par_iter()
        .filter_map(|record: &StringRecord| {
            // Skip records with missing values (marked with ?)
            let power_str = record.get(2).unwrap_or("?");
            if power_str == "?" || power_str.is_empty() {
                return None;
            }
            // Parse date and time
            let date_str = record.get(0).unwrap_or_default();
            let time_str = record.get(1).unwrap_or_default();
            let datetime_str = format!("{date_str} {time_str}");
            let naive_datetime =
                NaiveDateTime::parse_from_str(&datetime_str, "%d/%m/%Y %H:%M:%S").ok()?;
            let datetime =
                DateTime::<chrono::Utc>::from_naive_utc_and_offset(naive_datetime, chrono::Utc);
            let power = power_str.parse::<f64>().ok()?;
            Some(Point::new(datetime.timestamp() as f64, power))
        })
        .collect();

    // Normalize Unix timestamp to hours
    if !data.is_empty() {
        let min_ts = data.iter().map(|p| p.x()).fold(f64::INFINITY, f64::min);
        data = data
            .into_iter()
            .map(|p| Point::new((p.x() - min_ts) / 3600.0, p.y()))
            .collect();
    }

    Ok(data)
}
