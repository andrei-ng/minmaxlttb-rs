# MinMaxLTTB

A Rust crate for downsampling timeseries data using the LTTB (Largest Triangle Three Buckets) and MinMaxLTTB algorithms.

The standard LTTB algorithm is implemented as described in the original paper [Downsampling Time Series for Visual Representation](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf). 
The MinMaxLTTB algorithm follows [MinMaxLTTB: Leveraging MinMax-Preselection to Scale LTTB](https://arxiv.org/abs/2305.00332).

The crate draws inspiration from other implementations of LTTB and its variants 
 - https://github.com/jeromefroe/lttb-rs
 - https://github.com/predict-idlab/MinMaxLTTB
 - https://github.com/cpbotha/lttb-bench/

## Variants

- **Standard LTTB**: Classic implementation of LTTB downsampling
- **MinMax LTTB**: MinMax variant that better preserves local minima and maxima

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
let downsampled = minmaxlttb::lttb(&points, 2);

// Advanced usage with builder pattern
let lttb = LttbBuilder::new()
    .threshold(2)
    .method(LttbMethod::MinMax)
    .ratio(3)
    .build();

let result = lttb.downsample(&points);
```

## Examples

Check the `examples` directory for a few examples that generate plots using the `plotly-rs` crate and are rendered in your system's default browser:
 - power: example of downsampling a real-world power consumption dataset with MinMaxLTTB
 - timeseries: example of downsampling a synthetic dataset with standard LTTB
 - minmax_vs_standard: comparison of LTTB and MinMaxLTTB downsampling
 - lttb_analysis: a visual analysis of the point selections and bucket/partition division for the MinMaxLTTB algorithm

## License

MIT 