//! # MinMaxLTTB - MinMax Largest Triangle Three Buckets
//!
//! This crate provides implementations of the LTTB (Largest Triangle Three Buckets) and MinMaxLTTB algorithm
//! for downsampling timeseries data for visualization purposes.
//!
//! ## Variants
//!
//! - **Classic LTTB**: Classic implementation of LTTB downsampling using buckets with equal number of points
//! - **Standard LTTB**: Alternative implementation of classic LTTB downsampling using buckets with equal x-axis range
//! - **MinMax LTTB**: MinMax variant that preserves local minima and maxima and is more computationally efficient
//!
//! ## Usage
//!
//! ```rust
//! use minmaxlttb::{Point, Lttb, LttbBuilder, LttbMethod, Binning};
//!
//! // E.g., usage with convenience functions
//! let points = vec![
//!     Point::new(0.0, 1.0),
//!     Point::new(1.0, 2.0),
//!     Point::new(2.0, 3.0),
//!     Point::new(3.0, 4.0),
//! ];
//!
//! // Classic LTTB (equal-count buckets)
//! let classic = minmaxlttb::lttb(&points, 3, Binning::ByCount).unwrap();
//!
//! // Standard LTTB (equal x-range buckets)
//! let standard = minmaxlttb::lttb(&points, 3, Binning::ByRange).unwrap();
//!
//! // Advanced usage with builder pattern, e.g., using MinMax LTTB with ratio=3
//! let lttb = LttbBuilder::new()
//!     .threshold(3)
//!     .method(LttbMethod::MinMax)
//!     .ratio(3)
//!     .build();
//!
//! let result = lttb.downsample(&points).unwrap();
//!
//! // Reuse the same configuration for multiple datasets
//! let dataset1 = vec![Point::new(0.0, 10.0), Point::new(1.0, 20.0), Point::new(2.0, 30.0), Point::new(3.0, 40.0)];
//! let dataset2 = vec![Point::new(0.0, 30.0), Point::new(1.0, 40.0), Point::new(2.0, 50.0), Point::new(3.0, 60.0)];
//!
//! let lttb = LttbBuilder::new()
//!     .threshold(3)
//!     .method(LttbMethod::Classic)
//!     .build();
//!
//! let result1 = lttb.downsample(&dataset1).unwrap();
//! let result2 = lttb.downsample(&dataset2).unwrap();
//! ```

use std::{error::Error, fmt};
pub type Result<T> = std::result::Result<T, LttbError>;

#[derive(Debug, PartialEq)]
/// Error returned by LTTB downsampling
pub enum LttbError {
    /// Error returned when the provided threshold is invalid
    /// `n_in` is the number of points in the original set
    /// `n_out` is the number of points to downsample to (the threshold)
    /// `n_out` must be greater than 2 and less than `n_in`
    InvalidThreshold { n_in: usize, n_out: usize },
    /// Error returned when the provided ratio is invalid
    /// `ratio` is the number of extrema points to preselect from each `bucket` before running the LTTB algorithm
    /// `ratio` must be greater than 2
    InvalidRatio { ratio: usize },
    /// Error returned when requested to partition an empty bucket
    EmptyBucketPartitioning,
    /// Error returned when the the boundaries of a bucket are invalid (non-increasing or out of range)
    /// `start` is the start index of the bucket in the original timeseries
    /// `end` is the end index of the bucket in the original timeseries
    /// `start` must be less than `end`
    InvalidBucketLimits { start: usize, end: usize },
}

impl fmt::Display for LttbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LttbError::InvalidThreshold { n_in, n_out } => write!(
                f,
                "threshold n_out={n_out} invalid; must be 2 < n_out < n_in={n_in}"
            ),
            LttbError::InvalidRatio { ratio } => {
                write!(f, "ratio is invalid; must be >= 2 (got {ratio})")
            }
            LttbError::EmptyBucketPartitioning => write!(f, "cannot partition an empty bucket"),
            LttbError::InvalidBucketLimits { start, end } => {
                write!(f, "evaluated invalid bucket with limits at [{start},{end})")
            }
        }
    }
}
impl Error for LttbError {}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
/// Defines a `Point` with x and y coordinates
pub struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl Point {
    /// Create a new `Point` with the given x and y coordinates
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Get the x coordinate of the `Point`
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get the y coordinate of the `Point`
    pub fn y(&self) -> f64 {
        self.y
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
/// Method to use for downsampling
pub enum LttbMethod {
    /// Classic LTTB algorithm as described in the original paper
    /// where the bucket size is based on counting the number of points required, i.e., `Binning::ByCount`.
    /// [Downsampling Time Series for Visual Representation](https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf)
    Classic,

    /// Standard LTTB algorithm improves upon the classic algorithm by using
    /// buckets that have equal x-axis range, i.e., `Binning::ByRange`.
    Standard,

    /// MinMax LTTB algorithm as described in the original paper
    /// [MinMaxLTTB: Leveraging MinMax-Preselection to Scale LTTB](https://arxiv.org/abs/2305.00332).
    /// MinMax LTTB uses MinMax preselection over equal x-axis range partitions to choose extrema points for buckets.
    /// It produces better visual results since it preserves better the original shape of the data.
    #[default]
    MinMax,
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
/// Method to use for splitting the points into buckets
pub enum Binning {
    /// Equal number of points in each bucket
    #[default]
    ByCount,
    /// Equal x-axis range in each bucket
    ByRange,
}

/// Builder for configuring LTTB downsampling parameters with MinMax LTTB as default
#[derive(Default, Debug, Clone)]
pub struct LttbBuilder {
    lttb: Lttb,
}

impl LttbBuilder {
    /// Create a new builder with default configuration (default is MinMax LTTB with ratio=2)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the downsampling method
    pub fn method(mut self, method: LttbMethod) -> Self {
        self.lttb.method = method;
        self
    }

    /// Set the ratio for MinMaxLTTB (only used by MinMax algorithm variant)
    pub fn ratio(mut self, ratio: usize) -> Self {
        self.lttb.ratio = ratio;
        self
    }

    /// Set the threshold for the downsampling process
    pub fn threshold(mut self, threshold: usize) -> Self {
        self.lttb.threshold = threshold;
        self
    }

    /// Build the LTTB downsampler
    pub fn build(self) -> Lttb {
        self.lttb
    }
}

/// LTTB downsampler that can be used on a `Vec<Point>` to downsample it to the selected threshold
#[derive(Debug, Clone)]
pub struct Lttb {
    /// Number of points to downsample to
    threshold: usize,
    /// Method to use for downsampling
    method: LttbMethod,
    /// Ratio for MinMaxLTTB (only used by MinMax algorithm variant)
    /// Default is `DEFAULT_RATIO = 2`
    ratio: usize,
}

impl Default for Lttb {
    fn default() -> Self {
        Self {
            threshold: 0,
            method: LttbMethod::MinMax,
            ratio: Self::DEFAULT_RATIO,
        }
    }
}

impl Lttb {
    const DEFAULT_RATIO: usize = 2;

    /// Downsample the given points to the target size `threshold` and `ratio` provided by the builder
    ///
    /// `points` is the original set of points to downsample
    ///
    /// Preconditions: `points` must be strictly increasing in `x` (monotone with `x[i] < x[i+1]`).
    ///
    /// The first and last points are always preserved in the downsampled timeseries.
    pub fn downsample(&self, points: &[Point]) -> Result<Vec<Point>> {
        match self.method {
            LttbMethod::MinMax => minmaxlttb(points, self.threshold, self.ratio),
            LttbMethod::Classic => lttb(points, self.threshold, Binning::ByCount),
            LttbMethod::Standard => lttb(points, self.threshold, Binning::ByRange),
        }
    }
}

/// Downsample using the MinMax LTTB algorithm.
///
/// This algorithm is a variant of the LTTB algorithm that uses the MinMax pre-selection to choose extrema points for buckets.
/// It produces better visual results since it preserves better the original shape of the data.
///
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
/// `ratio` is the number of extrema points to preselect from globally defined, equidistant
///        x-range partitions across the inner range `[1..n-1]` before running LTTB algorithm
///
/// Preconditions: `points` must be strictly increasing in `x` (i.e., `x[i] < x[i+1]`).
///
/// Note:
/// - The first and last points are always included in the downsampled timeseries.
/// - When `ratio` approaches the bucket size (`points.len()/n_out`), all points in a bucket
///   are effectively selected, so MinMax converges to the classic LTTB behavior.
pub fn minmaxlttb(points: &[Point], n_out: usize, ratio: usize) -> Result<Vec<Point>> {
    debug_assert!(
        points.windows(2).all(|w| w[0].x() < w[1].x()),
        "points must be sorted by x"
    );
    if n_out >= points.len() || n_out < 3 {
        return Err(LttbError::InvalidThreshold {
            n_in: points.len(),
            n_out,
        });
    }

    if ratio < 2 {
        return Err(LttbError::InvalidRatio { ratio });
    }

    // Apply MinMax preselect only when bucket size > ratio
    let bucket_size = points.len() / n_out;
    if bucket_size > ratio {
        let selected = extrema_selection(points, n_out, ratio)?;
        lttb(&selected, n_out, Binning::ByCount)
    } else {
        lttb(points, n_out, Binning::ByCount)
    }
}

/// Downsample using the LTTB algorithm with a given binning method.
///
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
/// `binning_method` is the method to use for binning the points, i.e.,
/// `Binning::ByCount` for equal number of points in each bucket or
/// `Binning::ByRange` for equal x-axis range in each bucket
///
/// Preconditions: `points` must be strictly increasing in `x` (i.e., `x[i] < x[i+1]`).
///
/// The first and last points are always preserved in the downsampled timeseries.
///
pub fn lttb(points: &[Point], n_out: usize, binning_method: Binning) -> Result<Vec<Point>> {
    debug_assert!(
        points.windows(2).all(|w| w[0].x() < w[1].x()),
        "points must be sorted by x"
    );
    if n_out >= points.len() || n_out < 3 {
        return Err(LttbError::InvalidThreshold {
            n_in: points.len(),
            n_out,
        });
    }

    let bucket_bounds = match binning_method {
        Binning::ByCount => bucket_limits_by_count(points, n_out)?,
        Binning::ByRange => bucket_limits_by_range(points, n_out)?,
    };

    let mut downsampled = Vec::with_capacity(n_out);
    downsampled.push(points[0]); // Push first point

    // Iterate over all the buckets except the first and last
    for i in 1..n_out - 1 {
        let (start, end) = (bucket_bounds[i], bucket_bounds[i + 1]);
        let (next_s, next_e) = (bucket_bounds[i + 1], bucket_bounds[i + 2]);

        let first_vertex = downsampled[i - 1];
        let third_vertex =
            mean_point_bucket(&points[next_s..next_e]).ok_or(LttbError::InvalidBucketLimits {
                start: next_s,
                end: next_e,
            })?;

        let best_vertex = vertex_by_max_area(&points[start..end], first_vertex, third_vertex)
            .ok_or(LttbError::InvalidBucketLimits { start, end })?;

        downsampled.push(best_vertex);
    }
    // Push last point
    downsampled.push(points[points.len() - 1]);
    Ok(downsampled)
}

/// Preselect the extrema points for each bucket using the MinMax algorithm
///
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
/// `ratio` is the number of extrema to preselect from globally equidistant x-range partitions
///        across the inner range `[1..n-1]` before running the LTTB algorithm
///
/// Preconditions: `points` must be strictly increasing in `x` (i.e., `x[i] < x[i+1]`).
///
/// The first and last points are always preserved in the selected points.
pub fn extrema_selection(points: &[Point], n_out: usize, ratio: usize) -> Result<Vec<Point>> {
    if n_out >= points.len() || n_out < 3 {
        return Err(LttbError::InvalidThreshold {
            n_in: points.len(),
            n_out,
        });
    }

    if ratio < 2 {
        return Err(LttbError::InvalidRatio { ratio });
    }

    // Global equidistant x-axis partitions across inner range [1..n-1]
    // Number of partitions equals (n_out * ratio) / 2, selecting 2 points per partition
    const NUM_PTS_PER_PARTITION: usize = 2;
    let num_partitions = n_out.saturating_mul(ratio / NUM_PTS_PER_PARTITION);

    let n_in = points.len();
    let mut selected: Vec<Point> = Vec::with_capacity(n_out * ratio);
    selected.push(points[0]);
    let bounds = partition_bounds_by_range(&points[1..(n_in - 1)], 1, num_partitions)?;
    for i in 0..num_partitions {
        let start = bounds[i];
        let end = bounds[i + 1];
        selected.extend(find_minmax(&points[start..end]));
    }
    selected.push(points[n_in - 1]);
    Ok(selected)
}

/// Returns the best candidate Point from the provided slice of points by maximizing the area
/// of the triangle formed by the first vertex, the next vertex and any vertex from the provided points.
///
/// The best candidate is the vertex that maximizes the area of the triangle,
/// hence the Largest Triangle Three Buckets (LTTB) algorithm name.
///
/// `points` is the slice of points to consider (usually a bucket)
/// `first_vertex` is the best candidate Point of the previous adjacent bucket
/// `next_vertex` is the mean Point of the all points in the next adjacent bucket
///
/// Returns `None` if the slice of points is empty
fn vertex_by_max_area(points: &[Point], first_vertex: Point, next_vertex: Point) -> Option<Point> {
    let mut max_area = f64::MIN;
    let mut best_candidate = None;
    for p in points.iter() {
        let area = triangle_area(&first_vertex, p, &next_vertex);
        if area >= max_area {
            max_area = area;
            best_candidate = Some(*p);
        }
    }
    best_candidate
}

/// Returns the mean `Point` for a slice of points by computing the average of the x and y coordinates
/// Returns `None` if the slice of points is empty
pub fn mean_point_bucket(points: &[Point]) -> Option<Point> {
    if points.is_empty() {
        return None;
    }

    let mut mean_p = Point::new(0.0, 0.0);
    for p in points {
        mean_p.x += p.x;
        mean_p.y += p.y;
    }
    Some(Point {
        x: mean_p.x / points.len() as f64,
        y: mean_p.y / points.len() as f64,
    })
}

/// Returns the MIN and MAX points in a slice of points as a vector
/// where the MIN point has the lowest Y value and the MAX point has the highest Y value.
///
/// `points` is the slice of points to consider
///
/// Returns:
/// - `points.to_vec()` when the input has at most 2 points
/// - `[min_p, max_p]` when the input has more than 2 points
///
/// Tie-breaking and order:
/// - If multiple points share the minimum Y, the first (leftmost in x) is chosen.
/// - If multiple points share the maximum Y, the last (rightmost in x) is chosen.
/// - The output is always ordered by increasing x.
///
pub fn find_minmax(points: &[Point]) -> Vec<Point> {
    let mut result = Vec::with_capacity(2);
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut min_p = points[0];
    let mut max_p = points[0];

    for p in points.iter() {
        if p.y < min_p.y {
            min_p = *p;
        }
        if p.y >= max_p.y {
            max_p = *p;
        }
    }
    if min_p.x < max_p.x {
        result.push(min_p);
        result.push(max_p);
    } else {
        result.push(max_p);
        result.push(min_p);
    }
    result
}

/// Returns a vector of all bucket boundaries (indices) using floating-point arithmetic
/// such that the number of points in each bucket is equal (by count)
///
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
///
/// Output format:
/// - Returns `n_out + 1` boundaries `bounds` such that bucket `i` corresponds to
///   the slice `points[bounds[i]..bounds[i+1]]`.
/// - Boundaries are strictly increasing;
/// - First bucket contains only the first point,i.e., first two elements of the output are
///   always [0,1, ...]
/// - Last bucket contains only the last point,i.e., last two elements of the output are
///   always [n_in-1, n_in].
///
pub fn bucket_limits_by_count(points: &[Point], n_out: usize) -> Result<Vec<usize>> {
    let n_in = points.len();
    if n_out >= n_in || n_out < 3 {
        return Err(LttbError::InvalidThreshold { n_in, n_out });
    }

    // Exclude the end points from bucket calculations
    let n_in_exclusive = (n_in - 2) as f64;
    let n_out_exclusive = (n_out - 2) as f64;
    let bucket_size = n_in_exclusive / n_out_exclusive;

    let mut bounds = Vec::with_capacity(n_out + 1);

    bounds.push(0);
    for i in 0..n_out - 1 {
        let edge = (1.0 + i as f64 * bucket_size) as usize;
        bounds.push(edge);
    }
    bounds.push(n_in);
    Ok(bounds)
}

/// Return a vector of all partition boundaries (indices) using floating-point arithmetic
/// such that the number of points in each partition is equal (by count)
///
/// `start` is the start index of the current partition
/// `end` is the end index of the current partition
/// `n` is the number of partitions to create
///
/// Output format:
/// - `n + 1` strictly increasing absolute indices covering `[start, end)`
/// - returns 2 absolute indices `[start, end]` when `n == 0`
///
/// The indices can be used to slice partitions as `points[b[i]..b[i+1]]`.
pub fn partition_limits_by_count(start: usize, end: usize, n: usize) -> Result<Vec<usize>> {
    if start >= end {
        return Err(LttbError::InvalidBucketLimits { start, end });
    }

    if n == 0 {
        return Ok(vec![start, end]);
    }

    let size = (end - start) as f64 / n as f64;

    let mut bounds = Vec::with_capacity(n + 1);
    for i in 0..n {
        let edge = (i as f64 * size) as usize;
        bounds.push(start + edge);
    }
    bounds.push(end);
    Ok(bounds)
}

/// Returns a vector of all bucket boundaries (indices) such that
/// all buckets have the same x-axis range
///
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
///
/// Output format:
/// - Returns `n_out + 1` boundaries `bounds` such that bucket `i` corresponds to
///   the slice `points[bounds[i]..bounds[i+1]]`.
/// - Boundaries are strictly increasing;
/// - First bucket contains only the first point,i.e., first two elements of the output are
///   always [0,1, ...]
/// - Last bucket contains only the last point,i.e., last two elements of the output are
///   always [n_in-1, n_in].
pub fn bucket_limits_by_range(points: &[Point], n_out: usize) -> Result<Vec<usize>> {
    let n_in = points.len();
    if n_out >= n_in || n_out < 3 {
        return Err(LttbError::InvalidThreshold { n_in, n_out });
    }

    // Exclude the end points from bucket calculations
    let first_point: usize = 1;
    let last_point: usize = n_in - 2;
    let n_out_exclusive = (n_out - 2) as f64;
    let start_x = points[first_point].x();
    let end_x = points[last_point].x();

    let step_size = ((end_x - start_x) / n_out_exclusive).abs();
    let mut bounds = Vec::with_capacity(n_out + 1);

    bounds.push(0);
    bounds.push(1);

    let mut idx = 1;
    let mut prev = 1;
    for i in 1..n_out - 2 {
        let edge_x = start_x + step_size * i as f64;
        while idx < n_in - 1 && points[idx].x() < edge_x {
            idx += 1;
        }
        // Make sure we don't duplicate boundaries and enforce strictly increasing edges
        if idx <= prev {
            idx = (prev + 1).min(n_in - 2);
        }
        bounds.push(idx);
        prev = idx;
    }
    bounds.push(n_in - 1);
    bounds.push(n_in);
    Ok(bounds)
}

/// Return a vector of all partition boundaries (indices) such that the
/// all partitions have the same x-axis range
///
/// `start` is the start index of the current partition
/// `end` is the end index of the current partition
/// `n` is the number of partitions to create
///
/// Output format:
/// - `n + 1` strictly increasing absolute indices covering `[start, start + points.len())`
/// - returns 2 absolute indices `[start, start + points.len()]` when `n == 0`
///
/// The indices can be used to slice partitions as `points[b[i]..b[i+1]]`.
pub fn partition_bounds_by_range(points: &[Point], start: usize, n: usize) -> Result<Vec<usize>> {
    if n == 0 {
        return Ok(vec![start, start + points.len()]);
    }
    if points.is_empty() {
        return Err(LttbError::EmptyBucketPartitioning);
    }

    let start_x = points[0].x();
    let end_x = points[points.len() - 1].x();

    let step_size = ((end_x - start_x) / n as f64).abs();

    // n partitions => n+1 boundaries: [start, inner..., end]
    let mut bounds = Vec::with_capacity(n + 1);
    bounds.push(start);

    let mut idx = 0; // index in slice
    let mut prev_abs = start; // previous boundary (absolute index)
    for i in 1..n {
        let edge_x = start_x + step_size * i as f64;
        while idx < points.len() && points[idx].x() < edge_x {
            idx += 1;
        }
        // Make sure we don't duplicate boundaries and enforce strictly increasing edges
        let mut abs = start + idx;
        if abs <= prev_abs {
            abs = (prev_abs + 1).min(start + points.len() - 1);
            idx = abs - start;
        }
        bounds.push(abs);
        prev_abs = abs;
    }

    bounds.push(start + points.len());
    Ok(bounds)
}

#[inline(always)]
/// Returns the area of the triangle formed by the three points
///
/// `p1`, `p2`, `p3` are the three points of the triangle
///
fn triangle_area(p1: &Point, p2: &Point, p3: &Point) -> f64 {
    let a = p1.x * (p2.y - p3.y);
    let b = p2.x * (p3.y - p1.y);
    let c = p3.x * (p1.y - p2.y);
    (a + b + c).abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(always)]
    /// Helper method to get the edges of a bucket given an index
    fn bucket_edges_by_count(data: &[Point], n_out: usize, bucket_index: usize) -> (usize, usize) {
        let bucket_bounds = bucket_limits_by_count(data, n_out).unwrap();
        (bucket_bounds[bucket_index], bucket_bounds[bucket_index + 1])
    }

    #[test]
    fn threshold_conditions() {
        let data = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
        ];
        let n_out = 5;
        let result = lttb(&data, n_out, Binning::ByCount);
        assert_eq!(
            result,
            Err(LttbError::InvalidThreshold { n_in: 4, n_out: 5 })
        );

        let n_out = 2;
        let result = lttb(&data, n_out, Binning::ByCount);
        assert_eq!(
            result,
            Err(LttbError::InvalidThreshold { n_in: 4, n_out: 2 })
        );
    }

    #[test]
    fn bucket_mean_point() {
        assert!(mean_point_bucket(&[]).is_none());

        let data = vec![
            Point::new(0.0, 4.0),
            Point::new(1.0, 5.0),
            Point::new(2.0, 6.0),
            Point::new(3.0, 7.0),
        ];

        assert!(mean_point_bucket(&data[1..1]).is_none());

        assert_eq!(
            mean_point_bucket(&data).unwrap(),
            Point::new(6.0 / 4.0, 22.0 / 4.0)
        )
    }

    #[test]
    fn minmax_partition_check() {
        let data = vec![Point::new(0.0, 4.0)];
        assert_eq!(find_minmax(&data), vec![Point::new(0.0, 4.0)]);

        let data = vec![
            Point::new(0.0, 4.0),
            Point::new(1.0, 5.0),
            Point::new(2.0, 7.0),
            Point::new(3.0, 6.0),
        ];

        assert_eq!(find_minmax(&[]), vec![]);
        assert_eq!(find_minmax(&data[0..0]), vec![]);
        assert_eq!(find_minmax(&data[0..1]), vec![Point::new(0.0, 4.0)]);

        // Reverse order of points
        let data = vec![
            Point::new(0.0, 6.0),
            Point::new(1.0, 5.0),
            Point::new(2.0, 4.0),
            Point::new(3.0, 3.0),
        ];

        assert_eq!(
            find_minmax(&data),
            vec![Point::new(0.0, 6.0), Point::new(3.0, 3.0)]
        );

        let data = vec![
            Point::new(0.0, 4.0),
            Point::new(1.0, 4.0),
            Point::new(2.0, 4.0),
            Point::new(3.0, 4.0),
        ];

        assert_eq!(
            find_minmax(&data),
            vec![Point::new(0.0, 4.0), Point::new(3.0, 4.0)]
        );
    }

    #[test]
    fn right_vertex_for_first_bucket() {
        struct TestCase {
            name: &'static str,
            bucket_index: usize,
            expected_vertex: Option<Point>,
        }

        let cases = [
            TestCase {
                name: "Right vertex for 1st bucket",
                bucket_index: 0,
                expected_vertex: Some(Point::new(1.5, 2.5)), // Should return the average of the second bucket
            },
            TestCase {
                name: "Right vertex for 2nd bucket",
                bucket_index: 1,
                expected_vertex: Some(Point::new(3.0, 4.0)), // Should return the last point
            },
        ];

        let data = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
        ];
        let n_out = 3;

        for c in cases {
            let (next_start, next_end) = bucket_edges_by_count(&data, n_out, c.bucket_index + 1);
            let result = mean_point_bucket(&data[next_start..next_end]);
            assert_eq!(result, c.expected_vertex, "test case: {}", c.name,);
        }
    }

    #[test]
    fn right_vertex_for_middle_bucket() {
        let data = vec![
            Point::new(0.0, 1.0), // bucket 0
            Point::new(1.0, 2.0), // bucket 1 - start
            Point::new(2.0, 3.0), // bucket 1 - end
            Point::new(3.0, 4.0), // bucket 2 - start
            Point::new(4.0, 5.0), // bucket 2 - end
            Point::new(5.0, 6.0), // bucket 3
        ];
        let n_out = 4;
        let bucket_index = 1; // Middle bucket

        let (next_start, next_end) = bucket_edges_by_count(&data, n_out, bucket_index + 1);
        let result = mean_point_bucket(&data[next_start..next_end]);
        // Should return mean of bucket 2: (3.0+4.0)/2, (4.0+5.0)/2
        assert_eq!(result, Some(Point::new(3.5, 4.5)));
    }

    #[test]
    fn right_vertex_for_penultimate_bucket() {
        let data = vec![
            Point::new(0.0, 1.0), // bucket 0
            Point::new(1.0, 2.0), // bucket 1
            Point::new(2.0, 3.0), // bucket 1
            Point::new(3.0, 4.0), // bucket 2
        ];
        let n_out = 3;
        let bucket_index = n_out - 2; // Penultimate bucket is bucket 1

        let (next_start, next_end) = bucket_edges_by_count(&data, n_out, bucket_index + 1);
        let result = mean_point_bucket(&data[next_start..next_end]);

        assert_eq!(result, Some(Point::new(3.0, 4.0))); // Should return the last point
    }

    #[test]
    fn best_candidate_bucket() {
        let data = vec![
            Point::new(0.0, 0.0), // bucket 0
            Point::new(1.0, 1.0), // bucket 1 - candidate 1
            Point::new(1.0, 2.0), // bucket 1 - candidate 2 (higher area)
            Point::new(2.0, 0.0), // bucket 2
        ];
        let n_out = 3;
        let bucket_index = 1;
        let first = data[0];
        let third = data[3];

        let (start, end) = bucket_edges_by_count(&data, n_out, bucket_index);

        let result = vertex_by_max_area(&data[start..end], first, third);
        assert_eq!(result, Some(Point::new(1.0, 2.0))); // Should pick the point with higher triangle area
    }

    #[test]
    fn partition_bounds_by_count_check() {
        // Invalid inputs
        assert_eq!(
            partition_limits_by_count(0, 0, 3),
            Err(LttbError::InvalidBucketLimits { start: 0, end: 0 })
        );
        assert_eq!(
            partition_limits_by_count(4, 0, 3),
            Err(LttbError::InvalidBucketLimits { start: 4, end: 0 })
        );

        assert_eq!(partition_limits_by_count(4, 10, 0), Ok(vec![4, 10]));

        // 10 points, 3 partitions
        // Should split as: 4, 3, 3
        assert_eq!(partition_limits_by_count(0, 10, 3), Ok(vec![0, 3, 6, 10]));

        // 5 points, 2 partitions: 3, 2
        assert_eq!(partition_limits_by_count(0, 5, 2), Ok(vec![0, 2, 5]));

        // 7 points, 7 partitions: all size 1
        assert_eq!(
            partition_limits_by_count(0, 7, 7),
            Ok(vec![0, 1, 2, 3, 4, 5, 6, 7])
        );
    }

    #[test]
    fn minmax_preselect_preserves_extrema() {
        // Data with peaks, valleys, and intermediate points
        let data = vec![
            Point::new(0.0, 0.0),  // first
            Point::new(0.5, 2.0),  // bucket 1 - valley
            Point::new(1.0, 10.0), // bucket 1 - peak
            Point::new(1.5, 5.0),  // bucket 2 - peak
            Point::new(2.0, -5.0), // bucket 2 - valley
            Point::new(2.5, 0.0),  // bucket 3 - valley
            Point::new(3.0, 8.0),  // bucket 3 - peak
            Point::new(3.5, 4.0),  // bucket 3 - intermediate
            Point::new(4.0, 0.0),  // last
        ];
        // n_out = 5, ratio = 2 (global partitions over inner range)
        let selected = extrema_selection(&data, 5, 2).unwrap();
        // For this configuration, the expected output is:
        // - First and last points are always included
        // - (0.5, 2.0) and (1.0, 10.0) are included because they are extrema in the first bucket
        // - (1.5, 5.0) and (2.0, -5.0) are included because they are extrema in the second bucket
        // - (2.5, 0.0) and (3.0, 8.0) are included because they are extrema in the third bucket
        let expected = vec![
            Point::new(0.0, 0.0),
            Point::new(0.5, 2.0),
            Point::new(1.0, 10.0),
            Point::new(1.5, 5.0),
            Point::new(2.0, -5.0),
            Point::new(2.5, 0.0),
            Point::new(3.0, 8.0),
            Point::new(3.5, 4.0),
            Point::new(4.0, 0.0),
        ];
        assert_eq!(selected, expected);
    }

    #[test]
    fn minmax_preselect_handles_duplicates() {
        // All y values the same
        let data = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 1.0),
            Point::new(3.0, 1.0),
        ];
        let selected = extrema_selection(&data, 3, 2).unwrap();
        // Should include first and last
        assert_eq!(selected[0], data[0]);
        assert_eq!(selected[selected.len() - 1], data[3]);
        // Should not panic or duplicate points unnecessarily
        assert!(selected.iter().all(|p| p.y == 1.0));
    }

    #[test]
    fn minmax_preselect_small_buckets() {
        // Fewer points than n_out
        let data = vec![Point::new(0.0, 1.0), Point::new(1.0, 2.0)];
        let selected = extrema_selection(&data, 5, 2);
        // Should just return the original data
        assert_eq!(
            selected,
            Err(LttbError::InvalidThreshold { n_in: 2, n_out: 5 })
        );
    }

    #[test]
    fn minmaxlttb_invalid_inputs() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
        ];
        // n_out < 3
        assert_eq!(
            minmaxlttb(&points, 2, 2),
            Err(LttbError::InvalidThreshold {
                n_in: points.len(),
                n_out: 2
            })
        );
        assert_eq!(
            extrema_selection(&points, 2, 2),
            Err(LttbError::InvalidThreshold {
                n_in: points.len(),
                n_out: 2
            })
        );
        // n_out >= points.len()
        assert_eq!(
            minmaxlttb(&points, 4, 2),
            Err(LttbError::InvalidThreshold {
                n_in: points.len(),
                n_out: 4
            })
        );
        assert_eq!(
            extrema_selection(&points, 4, 2),
            Err(LttbError::InvalidThreshold {
                n_in: points.len(),
                n_out: 4
            })
        );
        // ratio < 2
        assert_eq!(
            minmaxlttb(&points, 3, 1),
            Err(LttbError::InvalidRatio { ratio: 1 })
        );
        assert_eq!(
            extrema_selection(&points, 3, 1),
            Err(LttbError::InvalidRatio { ratio: 1 })
        );
    }

    #[test]
    fn point_new() {
        let p = Point::new(1.0, 2.0);
        assert_eq!(p.x(), 1.0);
        assert_eq!(p.y(), 2.0);
    }

    #[test]
    fn downsample_classic_lttb_check() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
            Point::new(4.0, 5.0),
        ];
        let result = lttb(&points, 3, Binning::ByCount).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn builder_pattern() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
            Point::new(4.0, 5.0),
        ];

        // Test builder pattern with classic LTTB
        let result_classic = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::Classic)
            .build();
        assert_eq!(result_classic.downsample(&points).unwrap().len(), 3);

        // Test builder pattern with MinMaxLTTB and custom ratio
        let result_minmax = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::MinMax)
            .ratio(4)
            .build();
        assert_eq!(result_minmax.downsample(&points).unwrap().len(), 3);

        // Test builder pattern with MinMaxLTTB and default ratio
        let result_minmax_default = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::MinMax)
            .build();
        assert_eq!(result_minmax_default.downsample(&points).unwrap().len(), 3);

        let result_standard = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::Standard)
            .build()
            .downsample(&points)
            .unwrap();
        assert_eq!(result_standard.len(), 3);
    }

    #[test]
    fn bucket_limits_by_count_check() {
        // Invalid inputs
        let bounds = bucket_limits_by_count(&[Point::default(); 6], 2);
        let expected = Err(LttbError::InvalidThreshold { n_in: 6, n_out: 2 });
        assert_eq!(bounds, expected);

        let bounds = bucket_limits_by_count(&[Point::default(); 6], 6);
        let expected = Err(LttbError::InvalidThreshold { n_in: 6, n_out: 6 });
        assert_eq!(bounds, expected);

        let bounds = bucket_limits_by_count(&[Point::default(); 6], 4);
        let expected = vec![0, 1, 3, 5, 6];
        assert_eq!(bounds.unwrap(), expected);

        let bounds = bucket_limits_by_count(&[Point::default(); 6], 5);
        let expected = vec![0, 1, 2, 3, 5, 6];
        assert_eq!(bounds.unwrap(), expected);

        let bounds = bucket_limits_by_count(&[Point::default(); 10], 5);
        let expected = vec![0, 1, 3, 6, 9, 10];
        assert_eq!(bounds.unwrap(), expected);

        let bounds = bucket_limits_by_count(&[Point::default(); 15], 10);
        let expected = vec![0, 1, 2, 4, 5, 7, 9, 10, 12, 14, 15];
        assert_eq!(bounds.unwrap(), expected);
    }

    #[test]
    fn bucket_limits_by_range_early_return_conditions() {
        let data = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        ];
        // n_out >= n_in
        assert_eq!(
            bucket_limits_by_range(&data, 4),
            Err(LttbError::InvalidThreshold { n_in: 4, n_out: 4 })
        );
        // n_out < 3
        assert_eq!(
            bucket_limits_by_range(&data, 2),
            Err(LttbError::InvalidThreshold { n_in: 4, n_out: 2 })
        );
    }

    #[test]
    fn bucket_limits_by_range_non_uniform_spacing() {
        // Inner x-range is split in half; boundary falls between 0.2 and 5.0
        let data = vec![
            Point::new(0.0, 0.0), // first
            Point::new(0.1, 0.0), // start of inner range
            Point::new(0.2, 0.0),
            Point::new(5.0, 0.0),
            Point::new(5.1, 0.0),  // end of inner range
            Point::new(10.0, 0.0), // last
        ];
        // start = 0.1, end = 5.1, inner width = (5.1 - 0.1)/2 = 2.5
        // boundary1 = 3 → first point >= 0.1 + 2.5 is 5.0 at idx 3
        // boundary2 = 5 → last point, but bucket_limits_by_range already pushes n_in-1 and n_in at end
        let bounds = bucket_limits_by_range(&data, 4).unwrap();
        let expected = vec![0, 1, 3, 5, 6];
        assert_eq!(bounds, expected);
    }

    #[test]
    fn bucket_limits_by_range_negative_x_and_offset() {
        // Check correctness with negative x
        let data = vec![
            Point::new(-5.0, 0.0), // first
            Point::new(-4.5, 0.0), // start of inner range
            Point::new(-4.0, 0.0),
            Point::new(-1.0, 0.0),
            Point::new(3.0, 0.0),  // end of inner range
            Point::new(10.0, 0.0), // last
        ];
        // start = -4.5, end = 10.0, inner width = (3.0 - (-4.5))/2 = 3.75
        // boundary1 = 4 → first idx >= -4.5 + 3.75 is -0.75 at idx 4
        // boundary2 = 5 → last point, but bucket_limits_by_range already pushes n_in-1 and n_in at end
        let bounds = bucket_limits_by_range(&data, 4).unwrap();
        let expected = vec![0, 1, 4, 5, 6];
        assert_eq!(bounds, expected);
    }

    #[test]
    fn bucket_limits_by_range_matches_count_when_uniform() {
        // Uniform x-spacing → range buckets should equal count buckets
        let data: Vec<Point> = (0..=10).map(|i| Point::new(i as f64, 0.0)).collect();
        let n_out = 6;
        let by_range = bucket_limits_by_range(&data, n_out).unwrap();
        let by_count = bucket_limits_by_count(&data, n_out).unwrap();
        assert_eq!(by_range, by_count);
    }

    #[test]
    fn partition_bounds_by_range_edges() {
        // n == 0 returns [start, start + len]
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        ];
        assert_eq!(
            partition_bounds_by_range(&points, 5, 0).unwrap(),
            vec![5, 8]
        );

        // Empty points returns error
        let empty: Vec<Point> = vec![];
        assert_eq!(
            partition_bounds_by_range(&empty, 0, 3),
            Err(LttbError::EmptyBucketPartitioning)
        );
    }
    #[test]
    fn downsample_minmax_check() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
            Point::new(4.0, 5.0),
        ];
        let result = minmaxlttb(&points, 3, 2).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn force_extrema_selection_branch() {
        // Create enough points so bucket_size > ratio and extrema selection branch triggers
        let points: Vec<Point> = (0..100)
            .map(|i| Point::new(i as f64, (i % 7) as f64))
            .collect();
        let n_out = 10;
        let ratio = 2; // bucket_size = 100/10 = 10 > 2 → triggers extremaSelection branch
        let result = minmaxlttb(&points, n_out, ratio).unwrap();
        assert_eq!(result.len(), n_out);
    }

    #[test]
    fn lttberror_format() {
        let e1 = LttbError::InvalidThreshold { n_in: 4, n_out: 5 };
        assert_eq!(
            format!("{}", e1),
            "threshold n_out=5 invalid; must be 2 < n_out < n_in=4"
        );

        let e2 = LttbError::InvalidRatio { ratio: 1 };
        assert_eq!(format!("{}", e2), "ratio is invalid; must be >= 2 (got 1)");

        let e3 = LttbError::EmptyBucketPartitioning;
        assert_eq!(format!("{}", e3), "cannot partition an empty bucket");

        let e4 = LttbError::InvalidBucketLimits { start: 2, end: 1 };
        assert_eq!(
            format!("{}", e4),
            "evaluated invalid bucket with limits at [2,1)"
        );
    }
}
