use clap::Parser;
use csv::ReaderBuilder;
use minmaxlttb::{LttbBuilder, Point};
use plotly::{Layout, Plot, Scatter};
use std::error::Error;

const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/timeseries.csv");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Ratio to use for MinMax LTTB (must >= 2)
    #[arg(short, long, default_value_t = 4)]
    ratio: usize,

    /// Show bucket boundaries
    #[arg(long, default_value_t = true)]
    show_buckets: bool,

    /// Show partition boundaries
    #[arg(long, default_value_t = true)]
    show_partitions: bool,

    /// Show next vertices (mean points of next buckets)
    #[arg(long, default_value_t = true)]
    show_next_vertices: bool,

    /// Show min/max points from partitions
    #[arg(long, default_value_t = true)]
    show_min_max: bool,

    /// Show final selected points
    #[arg(long, default_value_t = true)]
    show_selected: bool,
}

// Helper function to get next vertices (mean points of next buckets) for visualization
fn get_last_vertices(points: &[Point], n_out: usize) -> Vec<Point> {
    (1..n_out - 1)
        .filter_map(|i| minmaxlttb::third_vertex(points, n_out, i))
        .collect()
}

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
    let args = Args::parse();

    if args.ratio < 2 {
        eprintln!("Error: ratio must be >= 2");
        std::process::exit(1);
    }

    let data = load_timeseries_data(DATA_PATH)?;
    let threshold = 500;

    let standard = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::Standard)
        .build()
        .downsample(&data.clone());
    let minmax = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::MinMax)
        .ratio(args.ratio)
        .build()
        .downsample(&data);

    println!("Original points: {}", data.len());
    println!("Standard LTTB: {} points", standard.len());
    println!(
        "MinMax LTTB (ratio={}): {} points",
        args.ratio,
        minmax.len()
    );

    // Get next vertices for visualization
    let last_vertices = get_last_vertices(&data, threshold);
    println!("next vertices: {} points", last_vertices.len());

    let mut plot = Plot::new();
    let x_orig: Vec<f64> = data.iter().map(|p| p.x()).collect();
    let y_orig: Vec<f64> = data.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_orig, y_orig)
            .name("Original")
            .line(plotly::common::Line::new().color("lightgray").width(1.5)),
    );

    // Standard LTTB
    let x_std: Vec<f64> = standard.iter().map(|p| p.x()).collect();
    let y_std: Vec<f64> = standard.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_std, y_std)
            .name("Standard LTTB")
            .line(plotly::common::Line::new().color("blue").width(2.0)),
    );

    // MinMax LTTB
    let x_mm: Vec<f64> = minmax.iter().map(|p| p.x()).collect();
    let y_mm: Vec<f64> = minmax.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_mm, y_mm)
            .name(format!("MinMax LTTB (ratio={})", args.ratio))
            .line(plotly::common::Line::new().color("red").width(2.0)),
    );

    if args.show_min_max {
        // Get min/max points from partitions for visualization
        let minmax_points = minmaxlttb::preselect_extrema(&data, threshold, args.ratio);
        println!(
            "Min/Max points from partitions: {} points",
            minmax_points.len()
        );

        // Separate min and max points from partitions
        let mut min_points = Vec::new();
        let mut max_points = Vec::new();

        // Get min/max points from partitions for visualization
        for bucket_idx in 1..threshold - 1 {
            let (bucket_start, bucket_end) =
                minmaxlttb::bucket_boundaries(data.len(), threshold, bucket_idx);
            let num_partitions = args.ratio / 2;
            for partition_idx in 0..num_partitions {
                let (s, e) = minmaxlttb::partition_boundaries(
                    bucket_end - bucket_start,
                    num_partitions,
                    partition_idx,
                );
                let start = bucket_start + s;
                let end = bucket_start + e;
                if end - start > 1 {
                    let (min_p, max_p) = minmaxlttb::minmax_partition(&data[start..end]);
                    min_points.push(min_p);
                    max_points.push(max_p);
                } else if end - start == 1 {
                    min_points.push(data[start]);
                    max_points.push(data[start]);
                }
            }
        }

        // Min points from partitions
        let x_min: Vec<f64> = min_points.iter().map(|p| p.x()).collect();
        let y_min: Vec<f64> = min_points.iter().map(|p| p.y()).collect();
        plot.add_trace(
            Scatter::new(x_min, y_min)
                .name("Min Points from Partitions")
                .web_gl_mode(true)
                .mode(plotly::common::Mode::Markers)
                .marker(plotly::common::Marker::new().color("blue").size(4))
                .visible(if args.show_min_max {
                    plotly::common::Visible::True
                } else {
                    plotly::common::Visible::False
                }),
        );

        // Max points from partitions
        let x_max: Vec<f64> = max_points.iter().map(|p| p.x()).collect();
        let y_max: Vec<f64> = max_points.iter().map(|p| p.y()).collect();
        plot.add_trace(
            Scatter::new(x_max, y_max)
                .name("Max Points from Partitions")
                .web_gl_mode(true)
                .mode(plotly::common::Mode::Markers)
                .marker(plotly::common::Marker::new().color("red").size(4))
                .visible(if args.show_min_max {
                    plotly::common::Visible::True
                } else {
                    plotly::common::Visible::False
                }),
        );
    }

    // Add bucket and partition boundaries as aggregated traces for proper legend toggling
    {
        // Get y-axis range for boundary lines
        let y_min = data.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min);
        let y_max = data.iter().map(|p| p.y()).fold(f64::NEG_INFINITY, f64::max);

        // Aggregate all bucket boundary vertical lines into a single trace using NaN separators
        if args.show_buckets {
            let mut x_bucket_lines: Vec<f64> = Vec::new();
            let mut y_bucket_lines: Vec<f64> = Vec::new();
            for bucket_idx in 1..threshold - 1 {
                let (bucket_start, _bucket_end) =
                    minmaxlttb::bucket_boundaries(data.len(), threshold, bucket_idx);
                let x_bucket_start = data[bucket_start].x();
                x_bucket_lines.push(x_bucket_start);
                x_bucket_lines.push(x_bucket_start);
                x_bucket_lines.push(f64::NAN);
                y_bucket_lines.push(y_min);
                y_bucket_lines.push(y_max);
                y_bucket_lines.push(f64::NAN);
            }
            if !x_bucket_lines.is_empty() {
                plot.add_trace(
                    Scatter::new(x_bucket_lines, y_bucket_lines)
                        .web_gl_mode(true)
                        .mode(plotly::common::Mode::Lines)
                        .name("Bucket Boundaries")
                        .line(
                            plotly::common::Line::new()
                                .color("black")
                                .width(2.0)
                                .dash(plotly::common::DashType::Dash),
                        )
                        .show_legend(true)
                        .visible(plotly::common::Visible::True),
                );
            }
        }

        // Aggregate all partition boundary vertical lines into a single trace
        if args.show_partitions {
            let mut x_partition_lines: Vec<f64> = Vec::new();
            let mut y_partition_lines: Vec<f64> = Vec::new();
            for bucket_idx in 1..threshold - 1 {
                let (bucket_start, bucket_end) =
                    minmaxlttb::bucket_boundaries(data.len(), threshold, bucket_idx);
                let num_partitions = args.ratio / 2;
                for partition_idx in 0..num_partitions {
                    let (s, e) = minmaxlttb::partition_boundaries(
                        bucket_end - bucket_start,
                        num_partitions,
                        partition_idx,
                    );
                    let start = bucket_start + s;
                    let end = bucket_start + e;
                    if start < data.len() && end <= data.len() {
                        let x_partition_start = data[start].x();
                        x_partition_lines.push(x_partition_start);
                        x_partition_lines.push(x_partition_start);
                        x_partition_lines.push(f64::NAN);
                        y_partition_lines.push(y_min);
                        y_partition_lines.push(y_max);
                        y_partition_lines.push(f64::NAN);
                    }
                }
            }
            if !x_partition_lines.is_empty() {
                plot.add_trace(
                    Scatter::new(x_partition_lines, y_partition_lines)
                        .web_gl_mode(true)
                        .mode(plotly::common::Mode::Lines)
                        .name("Partition Boundaries")
                        .line(
                            plotly::common::Line::new()
                                .color("orange")
                                .width(1.0)
                                .dash(plotly::common::DashType::Dot),
                        )
                        .show_legend(true)
                        .visible(plotly::common::Visible::True),
                );
            }
        }
    }

    // Add final selected points from each bucket (the actual LTTB result)
    if args.show_selected {
        let x_final: Vec<f64> = minmax.iter().map(|p| p.x()).collect();
        let y_final: Vec<f64> = minmax.iter().map(|p| p.y()).collect();
        plot.add_trace(
            Scatter::new(x_final, y_final)
                .name("Final Selected Points")
                .web_gl_mode(true)
                .mode(plotly::common::Mode::Markers)
                .marker(
                    plotly::common::Marker::new()
                        .color("green")
                        .size(8)
                        .symbol(plotly::common::MarkerSymbol::Diamond),
                )
                .visible(if args.show_selected {
                    plotly::common::Visible::True
                } else {
                    plotly::common::Visible::False
                }),
        );
    }

    // Add next vertices (mean points of next buckets)
    if args.show_next_vertices {
        let x_last_vertices: Vec<f64> = last_vertices.iter().map(|p| p.x()).collect();
        let y_last_vertices: Vec<f64> = last_vertices.iter().map(|p| p.y()).collect();
        plot.add_trace(
            Scatter::new(x_last_vertices, y_last_vertices)
                .name("Next Vertices (Next Bucket Means)")
                .web_gl_mode(true)
                .mode(plotly::common::Mode::Markers)
                .marker(plotly::common::Marker::new().color("purple").size(6))
                .visible(if args.show_next_vertices {
                    plotly::common::Visible::True
                } else {
                    plotly::common::Visible::False
                }),
        );
    }

    let layout = Layout::new()
        .title(plotly::common::Title::with_text(format!(
            "LTTB vs MinMaxLTTB (ratio={}) with Bucket/Partition Boundaries and Next Vertices",
            args.ratio
        )))
        .height(900)
        .show_legend(true)
        .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::with_text("Time")))
        .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::with_text("Value")));
    plot.set_layout(layout);

    let out_dir = "output";
    std::fs::create_dir_all(out_dir)?;
    let out_path = format!("{out_dir}/lttb_analysis.html");
    plot.write_html(&out_path);
    println!("Plot saved to {out_path}");
    plot.show_html(out_path);
    Ok(())
}
