## LTTB Analysis

This example is rather an analysis into the inner workings of the MinMaxLTTB algorithm. Using the `plotly-rs` crate it allows to visualize bucket/partition boundaries and selections of points.

It uses the same synthetic dataset in `examples/assets/timeseries.csv`.

Run the example with 
```
cargo run -- --show-buckets --show-partitions --show-min-max --show-selected --show-next-vertices --ratio=8
``` 


