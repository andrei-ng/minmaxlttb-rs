//! # MinMaxLTTB - MinMax Largest Triangle Three Buckets
//!
//! This crate provides implementations of the LTTB (Largest Triangle Three Buckets) and MinMaxLTTB algorithm
//! for downsampling time series data for visualization purposes.
//!
//! ## Variants
//!
//! - **Standard LTTB**: Classic implementation of LTTB downsampling
//! - **MinMax LTTB**: MinMax variant that better preserves local minima and maxima
//!
//! ## Usage
//!
//! ```rust
//! use minmaxlttb::{Point, Lttb, LttbBuilder, LttbMethod};
//!
//! // Simple usage with convenience functions
//! let points = vec![Point::new(0.0, 1.0), Point::new(1.0, 2.0), Point::new(2.0, 3.0)];
//! let downsampled = minmaxlttb::lttb(&points, 2);
//!
//! // Advanced usage with builder pattern
//! let lttb = LttbBuilder::new()
//!     .threshold(2)
//!     .method(LttbMethod::MinMax)
//!     .ratio(3)
//!     .build();
//!
//! let result = lttb.downsample(&points);
//!
//! // Reuse the same configuration for multiple datasets
//! let dataset1 = vec![Point::new(0.0, 1.0), Point::new(1.0, 2.0)];
//! let dataset2 = vec![Point::new(0.0, 3.0), Point::new(1.0, 4.0)];
//!
//! let lttb = LttbBuilder::new()
//!     .threshold(1)
//!     .method(LttbMethod::Standard)
//!     .build();
//!
//! let result1 = lttb.downsample(&dataset1);
//! let result2 = lttb.downsample(&dataset2);
//! ```

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub enum LttbMethod {
    Standard,
    #[default]
    MinMax,
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
        self.lttb.ratio = Some(ratio);
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
#[derive(Default, Debug, Clone)]
pub struct Lttb {
    /// Number of points to downsample to
    threshold: usize,
    /// Method to use for downsampling
    method: LttbMethod,
    /// Ratio for MinMaxLTTB (only used by MinMax algorithm variant)
    /// Default when not set is `DEFAULT_RATIO = 2`
    ratio: Option<usize>,
}

impl Lttb {
    const DEFAULT_RATIO: usize = 2;
    /// Downsample the given points to the target size `threshold`
    pub fn downsample(&self, points: &[Point]) -> Vec<Point> {
        match self.method {
            LttbMethod::MinMax => {
                let ratio = self.ratio.unwrap_or(Self::DEFAULT_RATIO);
                minmaxlttb(points, self.threshold, ratio)
            }
            LttbMethod::Standard => lttb(points, self.threshold),
        }
    }
}

/// Downsample using the MinMax algorithm.
/// This algorithm is a variant of the LTTB algorithm that uses the MinMax pre-selection to choose extrema points for buckets.
/// It produces better visual results since it preserves better the original shape of the data.
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
/// `ratio` is the number of extrema points to preselect from each `bucket` before running the LTTB algorithm
//
/// Returns the original data if any of the following holds true:
/// - `n_out` is greater than or equal to the number of input points
/// - `n_out` is less than 3
/// - `ratio` is less than 2
///  
/// Note: Select `ratio` with care. When `ratio` approaches the bucket size (`points.len()/n_out`),
/// partitions shrink to 1–2 points. This causes all points in a bucket to be selected,
/// making MinMax equivalent to standard LTTB and render the maxima/minima preselection step void.
pub fn minmaxlttb(points: &[Point], n_out: usize, ratio: usize) -> Vec<Point> {
    if n_out >= points.len() || n_out <= 2 || ratio < 2 {
        return points.to_vec();
    }

    let result = preselect_extrema(points, n_out, ratio);

    lttb(&result, n_out)
}

/// Downsample using the standard LTTB algorithm.
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
///
/// Returns the original data if any of the following holds true:
/// - `n_out` is greater than or equal to the number of input points
/// - `n_out` is less than 3
///
/// If `n_out` is greater than the number of points in the original data or less than 3, the function will return the original data.
pub fn lttb(points: &[Point], n_out: usize) -> Vec<Point> {
    if n_out >= points.len() || n_out <= 2 {
        return points.to_vec();
    }

    let mut downsampled = Vec::with_capacity(n_out);

    // Push first point
    downsampled.push(points[0]);

    for bucket_idx in 1..n_out - 1 {
        let first_vertex = downsampled[bucket_idx - 1];
        let third_vertex = match third_vertex(points, n_out, bucket_idx) {
            Some(vertex) => vertex,
            None => break, // We have reached the end, no more buckets
        };

        let best_vertex =
            match max_area_vertex(points, n_out, bucket_idx, first_vertex, third_vertex) {
                Some(vertex) => vertex,
                None => {
                    // most likely we have reached the end
                    break;
                }
            };

        downsampled.push(best_vertex);
    }
    // Push last point
    downsampled.push(points[points.len() - 1]);
    downsampled
}

/// Preselect the extrema points for each bucket using the MinMax algorithm
///
/// `points` is the original set of points to downsample
/// `n_out` is the number of points to downsample to (also known as the threshold)
/// `ratio` is the number of extrema points to preselect from each `bucket` before running the LTTB algorithm
pub fn preselect_extrema(points: &[Point], n_out: usize, ratio: usize) -> Vec<Point> {
    if n_out >= points.len() || n_out <= 2 || ratio < 2 {
        return points.to_vec();
    }

    const NUM_ENDS: usize = 2;
    const NUM_PTS_PER_PARTITION: usize = 2;

    let mut selected = Vec::with_capacity((n_out - NUM_ENDS) * ratio + NUM_ENDS);

    let num_partitions = ratio / NUM_PTS_PER_PARTITION;

    // Push first point
    selected.push(points[0]);

    for bucket_idx in 1..n_out - 1 {
        let (bucket_start, bucket_end) = bucket_boundaries(points.len(), n_out, bucket_idx);
        for partition_idx in 0..num_partitions {
            let (s, e) =
                partition_boundaries(bucket_end - bucket_start, num_partitions, partition_idx);
            let start = bucket_start + s;
            let end = bucket_start + e;
            // if the partition has only one point, no need to perform MinMax selection
            if end - start == 1 {
                selected.push(points[start]);
            } else {
                // perform MinMax selection on the partition
                let (min_p, max_p) = minmax_partition(&points[start..end]);
                if min_p.x < max_p.x {
                    selected.push(min_p);
                    selected.push(max_p);
                } else {
                    selected.push(max_p);
                    selected.push(min_p);
                }
            }
        }
    }
    // Push last point
    selected.push(points[points.len() - 1]);
    selected
}

/// Returns the third (next) triangle vertex, computed as the mean of the points in the right-adjacent bucket.
/// For the last bucket (which has no right neighbor), returns `None`.
pub fn third_vertex(points: &[Point], n_out: usize, bucket_index: usize) -> Option<Point> {
    if bucket_index < n_out - 1 {
        let next_bucket = bucket_index + 1;
        let (start, end) = bucket_boundaries(points.len(), n_out, next_bucket);
        mean_point_bucket(&points[start..end])
    } else {
        None // For the last bucket, there is no next candidate
    }
}

/// Returns the best candidate vertex for the current bucket by maximizing the area of the triangle formed by
/// the current bucket's points, the left bucket's selected point and the right adjacent bucket's mean point.
fn max_area_vertex(
    points: &[Point],
    n_out: usize,
    bucket_index: usize,
    first_vertex: Point,
    next_vertex: Point,
) -> Option<Point> {
    if bucket_index < n_out - 1 {
        let (start, end) = bucket_boundaries(points.len(), n_out, bucket_index);
        // Start and end should never be the same as this will result in an empty bucket
        // This should never happen in practice since this function is only called
        // from higher level functions that checks the necessary conditions for
        // `n_out` to be higher than the number of original data points, thus making
        // sure that the bucket sizes are always non-zero
        debug_assert!(start != end, "buckets should be non-empty");

        let mut max_area = 0.0;
        let mut best_candidate = None;

        for p in points[start..end].iter() {
            let area = triangle_area(&first_vertex, p, &next_vertex);
            if area >= max_area {
                max_area = area;
                best_candidate = Some(*p);
            }
        }
        best_candidate
    } else {
        points.last().cloned()
    }
}

/// Returns the mean `Point` for a list of points by computing the average of the x and y coordinates
/// Returns `None` if the list of points is empty
pub fn mean_point_bucket(points: &[Point]) -> Option<Point> {
    if points.is_empty() {
        return None;
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;

    for p in points {
        mean_x += p.x;
        mean_y += p.y;
    }
    Some(Point {
        x: mean_x / points.len() as f64,
        y: mean_y / points.len() as f64,
    })
}

/// Returns the MIN and MAX points in a partition of points as a tuple
/// where the MIN point has the lowest y value and the MAX point has the highest y value.
///
/// Returns `(points[0], points[0])` if the partition has only one point
///
/// Panics if the partition is empty.
pub fn minmax_partition(points: &[Point]) -> (Point, Point) {
    debug_assert!(
        !points.is_empty(),
        "each partition must have at least one point"
    );
    if points.len() == 1 {
        return (points[0], points[0]);
    }

    let mut min_p = points[0];
    // use second point as max_p to avoid returning both min and max the same point
    // in case all points have the same y value
    let mut max_p = points[1];

    for p in points.iter() {
        if p.y < min_p.y {
            min_p = *p;
        }
        if p.y > max_p.y {
            max_p = *p;
        }
    }
    (min_p, max_p)
}

/// Returns the start and end indices of a bucket using floating-point arithmetic
///
/// `n_in` is the number of points in the original data
/// `n_out` is the number of partitions to create
/// `i_bucket` is the index of the bucket to find the start and end indices for
///
/// The first bucket is always the first point in the original data.
/// The last bucket is always the last point in the original data.
///
/// Returns `(0, 1)` if `i_bucket` is 0.
/// Returns `(n_in - 1, n_in)` if `i_bucket` is `n_out - 1`.
pub fn bucket_boundaries(n_in: usize, n_out: usize, i_bucket: usize) -> (usize, usize) {
    if i_bucket == 0 {
        return (0, 1);
    }

    if i_bucket >= n_out - 1 {
        return (n_in - 1, n_in);
    }

    // Exclude the end points from bucket calculations
    let n_in_exclusive = (n_in - 2) as f64;
    let n_out_exclusive = (n_out - 2) as f64;
    let bucket_size = n_in_exclusive / n_out_exclusive;

    let start = ((i_bucket - 1) as f64 * bucket_size + 1.0) as usize;
    let end = (i_bucket as f64 * bucket_size + 1.0) as usize;

    if i_bucket == n_out - 2 {
        // For the penultimate bucket, cover all remaining points up to the last point
        (start, n_in - 1)
    } else {
        (start, end)
    }
}

/// Return the start and end indices of a partition of points as a tuple
///
/// `n_in` is the number of points in the original data
/// `n_out` is the number of partitions to create
/// `i_partition` is the index of the partition to find the start and end indices for
///
/// Returns `(0, 0)` if either `n_in` or `n_out` is 0
///
pub fn partition_boundaries(n_in: usize, n_out: usize, i_partition: usize) -> (usize, usize) {
    if (n_in == 0) || (n_out == 0) {
        return (0, 0);
    }

    let size = n_in as f64 / n_out as f64;

    let start = (i_partition as f64 * size) as usize;
    let end = ((i_partition + 1) as f64 * size) as usize;

    if i_partition == n_out - 1 {
        // For the last bucket, cover all remaining points up to the last point
        (start, n_in)
    } else {
        (start, end)
    }
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

    #[test]
    fn threshold_conditions() {
        let data = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
        ];
        let n_out = 5;
        let result = lttb(&data, n_out);
        assert_eq!(result, data);

        let n_out = 2;
        let result = lttb(&data, n_out);
        assert_eq!(result, data);
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

        assert_eq!(
            mean_point_bucket(&data).unwrap(),
            Point::new(6.0 / 4.0, 22.0 / 4.0)
        )
    }

    #[test]
    fn minmax_partition_check() {
        let data = vec![
            Point::new(0.0, 4.0),
            Point::new(1.0, 5.0),
            Point::new(2.0, 7.0),
            Point::new(3.0, 6.0),
        ];

        let (min_p, max_p) = minmax_partition(&data);
        assert_eq!(min_p, Point::new(0.0, 4.0));
        assert_eq!(max_p, Point::new(2.0, 7.0));

        // Reverse order of points
        let data = vec![
            Point::new(0.0, 6.0),
            Point::new(1.0, 5.0),
            Point::new(2.0, 4.0),
            Point::new(3.0, 3.0),
        ];

        let (min_p, max_p) = minmax_partition(&data);
        assert_eq!(min_p, Point::new(3.0, 3.0));
        assert_eq!(max_p, Point::new(0.0, 6.0));

        let data = vec![
            Point::new(0.0, 4.0),
            Point::new(1.0, 4.0),
            Point::new(2.0, 4.0),
            Point::new(3.0, 4.0),
        ];

        let (min_p, max_p) = minmax_partition(&data);
        assert_eq!(min_p, Point::new(0.0, 4.0));
        assert_eq!(max_p, Point::new(1.0, 4.0));

        let data = vec![Point::new(0.0, 4.0)];

        let (min_p, max_p) = minmax_partition(&data);
        assert_eq!(min_p, Point::new(0.0, 4.0));
        assert_eq!(max_p, Point::new(0.0, 4.0));
    }

    #[test]
    fn bucket_boundaries_check() {
        let data_len = 6;
        let n_out = 4;

        assert_eq!(bucket_boundaries(data_len, n_out, 0), (0, 1));
        assert_eq!(bucket_boundaries(data_len, n_out, 1), (1, 3));
        assert_eq!(bucket_boundaries(data_len, n_out, 2), (3, 5));
        assert_eq!(bucket_boundaries(data_len, n_out, 3), (5, 6));
    }

    #[test]
    fn next_candidate_first_bucket() {
        let data = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
        ];
        let n_out = 3;
        let bucket_index = 0; // First bucket

        let result = third_vertex(&data, n_out, bucket_index).unwrap();
        assert_eq!(result, Point::new(1.5, 2.5)); // Should return the average of the second bucket
    }

    #[test]
    fn next_candidate_second_bucket() {
        let data = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
        ];
        let n_out = 3;
        let bucket_index = 1;

        let result = third_vertex(&data, n_out, bucket_index).unwrap();
        assert_eq!(result, Point::new(3.0, 4.0)); // Should return the last point
    }

    #[test]
    fn next_candidate_last_bucket() {
        let data = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
        ];
        let n_out = 3;
        let bucket_index = 2;

        let result = third_vertex(&data, n_out, bucket_index);
        assert_eq!(result, None);
    }

    #[test]
    fn next_candidate_middle_bucket() {
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

        let result = third_vertex(&data, n_out, bucket_index).unwrap();
        // Should return mean of bucket 2: (3.0+4.0)/2, (4.0+5.0)/2
        assert_eq!(result, Point::new(3.5, 4.5));
    }

    #[test]
    fn next_candidate_penultimate_bucket() {
        let data = vec![
            Point::new(0.0, 1.0), // bucket 0
            Point::new(1.0, 2.0), // bucket 1
            Point::new(2.0, 3.0), // bucket 2
            Point::new(3.0, 4.0), // bucket 3
        ];
        let n_out = 3;
        let bucket_index = n_out - 2; // Penultimate bucket (bucket 1)

        let result = third_vertex(&data, n_out, bucket_index).unwrap();
        assert_eq!(result, Point::new(3.0, 4.0)); // Should return the last point
    }

    #[test]
    fn best_candidate_middle_bucket() {
        let data = vec![
            Point::new(0.0, 0.0), // bucket 0
            Point::new(1.0, 1.0), // bucket 1 - candidate 1
            Point::new(1.0, 2.0), // bucket 1 - candidate 2 (higher area)
            Point::new(2.0, 0.0), // bucket 2
        ];
        let n_out = 3;
        let bucket_index = 1;
        let previous = Point::new(0.0, 0.0);
        let next = Point::new(2.0, 0.0);

        let result = max_area_vertex(&data, n_out, bucket_index, previous, next).unwrap();
        assert_eq!(result, Point::new(1.0, 2.0)); // Should pick the point with higher triangle area
    }

    #[test]
    fn best_candidate_last_bucket() {
        let data = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let n_out = 3;
        let bucket_index = 2; // Last bucket
        let previous = Point::new(1.0, 1.0);
        let dummy_next = Point::new(0.0, 0.0); // Not used for last bucket

        let result = max_area_vertex(&data, n_out, bucket_index, previous, dummy_next).unwrap();
        assert_eq!(result, Point::new(2.0, 2.0)); // Should return the last point
    }

    #[test]
    fn partition_boundaries_check() {
        // Invalid inputs
        assert_eq!(partition_boundaries(0, 3, 0), (0, 0));
        assert_eq!(partition_boundaries(4, 0, 0), (0, 0));

        // 10 points, 3 partitions
        // Should split as: 4, 3, 3
        assert_eq!(partition_boundaries(10, 3, 0), (0, 3));
        assert_eq!(partition_boundaries(10, 3, 1), (3, 6));
        assert_eq!(partition_boundaries(10, 3, 2), (6, 10));

        // 5 points, 2 partitions: 3, 2
        assert_eq!(partition_boundaries(5, 2, 0), (0, 2));
        assert_eq!(partition_boundaries(5, 2, 1), (2, 5));

        // 7 points, 7 partitions: all size 1
        for i in 0..7 {
            assert_eq!(partition_boundaries(7, 7, i), (i, i + 1));
        }
    }

    #[test]
    fn minmax_preselect_preserves_extrema() {
        // Data with clear peaks, valleys, and intermediate points
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
        // n_out = 5, ratio = 2 (so only a single partition per bucket)
        let selected = preselect_extrema(&data, 5, 2);
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
        let selected = preselect_extrema(&data, 3, 2);
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
        let selected = preselect_extrema(&data, 5, 2);
        // Should just return the original data
        assert_eq!(selected, data);
    }

    #[test]
    fn minmaxlttb_early_return() {
        let points = vec![Point::new(0.0, 1.0), Point::new(1.0, 2.0)];
        // n_out <= 2
        assert_eq!(minmaxlttb(&points, 2, 2), points);
        // n_out >= points.len()
        assert_eq!(minmaxlttb(&points, 3, 2), points);
        // ratio < 2
        assert_eq!(minmaxlttb(&points, 2, 1), points);
    }

    #[test]
    fn minmax_preselect_early_return() {
        let points = vec![Point::new(0.0, 1.0), Point::new(1.0, 2.0)];
        // n_out <= 2
        assert_eq!(preselect_extrema(&points, 2, 2), points);
        // n_out >= points.len()
        assert_eq!(preselect_extrema(&points, 3, 2), points);
        // ratio < 2
        assert_eq!(preselect_extrema(&points, 2, 1), points);
    }

    #[test]
    fn point_new() {
        let p = Point::new(1.0, 2.0);
        assert_eq!(p.x(), 1.0);
        assert_eq!(p.y(), 2.0);
    }

    #[test]
    fn downsample_standard_check() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 4.0),
            Point::new(4.0, 5.0),
        ];
        let result = lttb(&points, 3);
        assert_eq!(result.len(), 3);
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
        let result = minmaxlttb(&points, 3, 2);
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

        // Test builder pattern with standard LTTB
        let result_standard = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::Standard)
            .build();
        assert_eq!(result_standard.downsample(&points).len(), 3);

        // Test builder pattern with MinMaxLTTB and custom ratio
        let result_minmax = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::MinMax)
            .ratio(4)
            .build();
        assert_eq!(result_minmax.downsample(&points).len(), 3);

        // Test builder pattern with MinMaxLTTB and default ratio
        let result_minmax_default = LttbBuilder::new()
            .threshold(3)
            .method(LttbMethod::MinMax)
            .build();
        assert_eq!(result_minmax_default.downsample(&points).len(), 3);
    }
}
