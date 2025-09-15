# MinMaxLTTB

[![Crates.io](https://img.shields.io/crates/v/minmaxlttb.svg)](https://crates.io/crates/minmaxlttb)
[![Docs.rs](https://docs.rs/minmaxlttb/badge.svg)](https://docs.rs/minmaxlttb)
[![CI](https://github.com/andrei-ng/minmaxlttb-rs/actions/workflows/build.yml/badge.svg)](https://github.com/andrei-ng/minmaxlttb-rs/actions/workflows/build.yml)

A Rust crate for downsampling timeseries data using the LTTB (Largest Triangle Three Buckets) and MinMaxLTTB algorithms.

The classic LTTB algorithm is implemented as described in the original paper [Downsampling Time Series for Visual Representation](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf). 
The MinMaxLTTB algorithm follows [MinMaxLTTB: Leveraging MinMax-Preselection to Scale LTTB](https://arxiv.org/abs/2305.00332).

The crate draws inspiration from other implementations of LTTB and its variants 
 - https://github.com/jeromefroe/lttb-rs
 - https://github.com/predict-idlab/MinMaxLTTB
 - https://github.com/predict-idlab/tsdownsample
 - https://github.com/cpbotha/lttb-bench/

## Variants

- **Classic LTTB**: Classic implementation of LTTB downsampling with bucket binning based on equal count of points
- **Standard LTTB**: Alternative implementation of LTTB downsampling with bucket binning based on equidistant x-range for all buckets  
- **MinMax LTTB**: MinMax variant that preserves local minima and maxima and is more computationally efficient


The restriction on the input data is that points must represent a timeseries, with strictly monotonically increasing values of `x`.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
minmaxlttb = "0.1"
```

```rust
use minmaxlttb::{Point, Lttb, LttbBuilder, LttbMethod, Binning};

// E.g., usage with convenience functions
let points = vec![
    Point::new(0.0, 1.0),
    Point::new(1.0, 2.0),
    Point::new(2.0, 3.0),
    Point::new(3.0, 4.0),
];

// Classic LTTB (equal-count buckets)
let classic = minmaxlttb::lttb(&points, 3, Binning::ByCount).unwrap();

// Standard LTTB (equal x-range buckets)
let standard = minmaxlttb::lttb(&points, 3, Binning::ByRange).unwrap();

// Advanced usage with builder pattern, e.g., using MinMax LTTB with ratio=3
let lttb = LttbBuilder::new()
    .threshold(3)
    .method(LttbMethod::MinMax)
    .ratio(3)
    .build();

let result = lttb.downsample(&points).unwrap();

// Reuse the same configuration for multiple datasets
let dataset1 = vec![Point::new(0.0, 10.0), Point::new(1.0, 20.0), Point::new(2.0, 30.0), Point::new(3.0, 40.0)];
let dataset2 = vec![Point::new(0.0, 30.0), Point::new(1.0, 40.0), Point::new(2.0, 50.0), Point::new(3.0, 60.0)];

let lttb = LttbBuilder::new()
    .threshold(3)
    .method(LttbMethod::Classic)
    .build();

let result1 = lttb.downsample(&dataset1).unwrap();
let result2 = lttb.downsample(&dataset2).unwrap();
```

## Result of MinMaxLTTB on 5.000-point timeseries

Below images show the effects of downsampling a 5.000-point timeseries with MinMaxLTTB with different thresholds.

The plots have been exported from the `timeseries` example using the `--method minmax` argument
```Bash
cargo run -p example_timeseries -- --method minmax
```


<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/minmax_n5000_d100.png" alt="MinMaxLTTB threshold 100 on 5000 points" width="1100">
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/minmax_n5000_d250.png" alt="MinMaxLTTB threshold 250 on 5000 points" width="1100">
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/minmax_n5000_d500.png" alt="MinMaxLTTB threshold 500 on 5000 points" width="1100">
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/minmax_n5000_d1000.png" alt="MinMaxLTTB threshold 1000 on 5000 points" width="1100">
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/minmax_n5000_d2000.png" alt="MinMaxLTTB threshold 2000 on 5000 points" width="1100">

## Examples

Check the `examples` directory for examples that generate plots using the `plotly-rs` crate and are rendered in your system's default browser:
 - power: example of downsampling a real-world power consumption dataset with MinMaxLTTB
 - timeseries: example of downsampling a synthetic dataset with classic LTTB
 - minmax_vs_classic: comparison of LTTB and MinMaxLTTB downsampling
 - minmaxlttb_analysis: a visual analysis of the point selections and bucket/partition division for the MinMaxLTTB algorithm


### Timeseries example

```Bash
cargo run -p example_timeseries --release
```
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/timeseries-example.png" alt="Timeseries example" width="1100">

### Powerdata series example

```Bash
cargo run -p example_power --release
``` 
The plot below is a zoom-in on a region of the plot to emphasize the differences   
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/powerseries-example.png" alt="Powerdata example" width="1100">

### LTTB vs MinMaxLttb example

```Bash
cargo run -p example_minmax_vs_classic --release
``` 
<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main/docs/assets/lttb-vs-minmax-example.png" alt="MinMaxLTTB vs LTTB" width="1100">

###  MinMaxLttb analysis example

This example generates a plot that illustrates how the MinMaxLTTB algorithm selects points for each bucket, highlighting the next average point, the preselected points, partition boundaries, and bucket boundaries.

```Bash
cargo run -p example_minmaxlttb_analysis --release -- --show-buckets=on --show-partitions=on --show-min-max=on --show-selected=on --show-next-vertices=on --ratio=4
``` 

<img src="https://github.com/andrei-ng/minmaxlttb-rs/raw/main//docs/assets/minmaxlttb-analysis-example.png" alt="MinMaxLTTB analysis" width="1100">

## License

MIT 
