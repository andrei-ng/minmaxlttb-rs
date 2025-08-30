# MinMaxLTTB

A Rust crate for downsampling timeseries data using the LTTB (Largest Triangle Three Buckets) and MinMaxLTTB algorithms.

The classic LTTB algorithm is implemented as described in the original paper [Downsampling Time Series for Visual Representation](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf). 
The MinMaxLTTB algorithm follows [MinMaxLTTB: Leveraging MinMax-Preselection to Scale LTTB](https://arxiv.org/abs/2305.00332).

The crate draws inspiration from other implementations of LTTB and its variants 
 - https://github.com/jeromefroe/lttb-rs
 - https://github.com/predict-idlab/MinMaxLTTB
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
minmaxlttb = "0.1.0"
```

```rust
use minmaxlttb::{Point, LttbBuilder, LttbMethod};

// Simple usage with convenience functions
let points = vec![Point::new(0.0, 1.0), Point::new(1.0, 2.0), Point::new(2.0, 3.0)];
let downsampled = minmaxlttb::lttb(&points, 2, Binning::ByCount);

// Advanced usage with builder pattern
let lttb = LttbBuilder::new()
    .threshold(2)
    .method(LttbMethod::MinMax)
    .ratio(3)
    .build();

let result = lttb.downsample(&points);
```

## Examples

Check the `examples` directory for examples that generate plots using the `plotly-rs` crate and are rendered in your system's default browser:
 - power: example of downsampling a real-world power consumption dataset with MinMaxLTTB
 - timeseries: example of downsampling a synthetic dataset with classic LTTB
 - minmax_vs_classic: comparison of LTTB and MinMaxLTTB downsampling
 - minmaxlttb_analysis: a visual analysis of the point selections and bucket/partition division for the MinMaxLTTB algorithm


Run any of the examples by using one of the commands 
```Bash
cargo run -p example_timeseries --release
```

```Bash
cargo run -p example_power --release
``` 


```Bash
cargo run -p example_minmax_vs_classic --release
``` 

```Bash
cargo run -p example_minmaxlttb_analysis --release -- --show-buckets --show-partitions --show-min-max --show-selected --show-next-vertices --ratio=4
``` 

## License

MIT 