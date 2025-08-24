use clap::builder::BoolishValueParser;
use clap::{ArgAction, Parser};
use csv::ReaderBuilder;
use minmaxlttb::{LttbBuilder, Point};
use plotly::{common::DashType, Configuration, Layout, Plot, Scatter};
use std::error::Error;

const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/timeseries.csv");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Ratio to use for MinMax LTTB (must >= 2)
    #[arg(short, long, default_value_t = 4)]
    ratio: usize,

    /// Downsampling threshold (number of output points)
    #[arg(short = 't', long, default_value_t = 500)]
    threshold: usize,

    /// Show bucket boundaries
    #[arg(long, default_value_t = true, action = ArgAction::Set, value_parser = BoolishValueParser::new())]
    show_buckets: bool,

    /// Show partition boundaries
    #[arg(long, default_value_t = true, action = ArgAction::Set, value_parser = BoolishValueParser::new())]
    show_partitions: bool,

    /// Show next vertices (mean points of next buckets)
    #[arg(long, default_value_t = true, action = ArgAction::Set, value_parser = BoolishValueParser::new())]
    show_next_vertices: bool,

    /// Show min/max points from partitions
    #[arg(long, default_value_t = true, action = ArgAction::Set, value_parser = BoolishValueParser::new())]
    show_min_max: bool,

    /// Show final selected points
    #[arg(long, default_value_t = true, action = ArgAction::Set, value_parser = BoolishValueParser::new())]
    show_selected: bool,
}

// Helper function to get next vertices (mean points of next buckets) for visualization
fn compute_next_vertices(points: &[Point], n_out: usize) -> Vec<Point> {
    let edges = minmaxlttb::bucket_limits_by_count(points, n_out).unwrap();
    if edges.len() < 3 {
        return Vec::new();
    }
    (1..=n_out - 2)
        .filter_map(|i| {
            let (ns, ne) = (edges[i + 1], edges[i + 2]);
            minmaxlttb::mean_point_bucket(&points[ns..ne])
        })
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
    let threshold = args.threshold;

    let classic = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::Classic)
        .build()
        .downsample(&data)
        .unwrap();
    let minmax = LttbBuilder::new()
        .threshold(threshold)
        .method(minmaxlttb::LttbMethod::MinMax)
        .ratio(args.ratio)
        .build()
        .downsample(&data)
        .unwrap();

    println!("Original points: {}", data.len());
    println!("Classic LTTB: {} points", classic.len());
    println!(
        "MinMax LTTB (ratio={}): {} points",
        args.ratio,
        minmax.len()
    );

    let bucket_size = data.len() / threshold;
    let used_preselection = bucket_size > args.ratio;
    if used_preselection {
        println!(
            "MinMax preselection: ACTIVE\n\t n_in={}\n\t threshold={}\n\t bucket_size={} > ratio={}",
            data.len(),
            threshold,
            bucket_size,
            args.ratio
        );
    } else {
        println!(
            "MinMax preselection: NON-ACTIVE\n\t n_in={}\n\t threshold={}\n\t bucket_size={} <= ratio={}",
            data.len(),
            threshold,
            bucket_size,
            args.ratio
        );
    }
    let analysis_points: Vec<Point> = if used_preselection {
        minmaxlttb::extrema_selection(&data, threshold, args.ratio)?
    } else {
        data.clone()
    };
    // Compute global partition bounds only if preselection is used (for visualization only)
    let (num_partitions, global_partition_bounds): (usize, Option<Vec<usize>>) =
        if used_preselection {
            let np = threshold.saturating_mul(args.ratio / 2);
            let gb = minmaxlttb::partition_bounds_by_range(&data[1..(data.len() - 1)], 1, np)?;
            (np, Some(gb))
        } else {
            (0, None)
        };

    // Get next vertices for visualization (based on count-buckets over the points used by LTTB)
    let last_vertices = compute_next_vertices(&analysis_points, threshold);

    let mut plot = Plot::new();
    let x_orig: Vec<f64> = data.iter().map(|p| p.x()).collect();
    let y_orig: Vec<f64> = data.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_orig, y_orig).name("Original").line(
            plotly::common::Line::new()
                .color("black")
                .width(1.5)
                .dash(DashType::Dash),
        ),
    );

    // Classic LTTB (original)
    let x_std: Vec<f64> = classic.iter().map(|p| p.x()).collect();
    let y_std: Vec<f64> = classic.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_std, y_std)
            .name("Classic LTTB")
            .line(plotly::common::Line::new().color("blue").width(1.5)),
    );

    // MinMax LTTB
    let x_mm: Vec<f64> = minmax.iter().map(|p| p.x()).collect();
    let y_mm: Vec<f64> = minmax.iter().map(|p| p.y()).collect();
    plot.add_trace(
        Scatter::new(x_mm, y_mm)
            .name(format!("MinMax LTTB (ratio={})", args.ratio))
            .line(plotly::common::Line::new().color("red").width(1.5)),
    );

    if args.show_min_max && used_preselection {
        // Get min/max points from all range partitions (inner span), mirroring MinMax preselection
        let mut min_points = Vec::new();
        let mut max_points = Vec::new();

        let bounds = global_partition_bounds
            .as_ref()
            .expect("bounds present when used_preselection");
        for i_p in 0..num_partitions {
            let start = bounds[i_p];
            let end = bounds[i_p + 1];
            let minmax = minmaxlttb::find_minmax(&data[start..end]);
            if minmax.len() == 1 {
                min_points.push(minmax[0]);
                max_points.push(minmax[0]);
            } else {
                min_points.push(minmax[0]);
                max_points.push(minmax[1]);
            }
        }
        println!(
            "Min/Max points from partitions: {} points",
            min_points.len() + max_points.len()
        );

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
            let edges = minmaxlttb::bucket_limits_by_count(&analysis_points, threshold)?;
            for bucket_idx in 1..threshold - 1 {
                let (bucket_start, _bucket_end) = (edges[bucket_idx], edges[bucket_idx + 1]);
                let x_bucket_start = analysis_points[bucket_start].x();
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

        // Aggregate all partition boundary vertical lines into a single trace (range partitions)
        if args.show_partitions && used_preselection {
            let mut x_partition_lines: Vec<f64> = Vec::new();
            let mut y_partition_lines: Vec<f64> = Vec::new();
            // Draw inner boundaries only (skip the very first and last which coincide with endpoints)
            let bounds = global_partition_bounds
                .as_ref()
                .expect("bounds present when used_preselection");
            for &b in bounds.iter().take(num_partitions).skip(1) {
                if b < data.len() {
                    let x_partition_start = data[b].x();
                    x_partition_lines.push(x_partition_start);
                    x_partition_lines.push(x_partition_start);
                    x_partition_lines.push(f64::NAN);
                    y_partition_lines.push(y_min);
                    y_partition_lines.push(y_max);
                    y_partition_lines.push(f64::NAN);
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
            "MinMaxLTTB (ratio={}) with Bucket/Partition Boundaries and Next Vertices",
            args.ratio
        )))
        .height(900)
        .show_legend(true)
        .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::with_text("Time")))
        .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::with_text("Value")));
    plot.set_layout(layout);
    plot.set_configuration(Configuration::default().responsive(true));

    let out_dir = "output";
    std::fs::create_dir_all(out_dir)?;
    let out_path = format!("{out_dir}/minmaxlttb_analysis.html");
    plot.write_html(&out_path);
    println!("Plot saved to {out_path}");
    plot.show_html(out_path);
    Ok(())
}
