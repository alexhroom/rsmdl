use std::cmp::min;
use std::iter::Iterator;
use std::time::{Duration, Instant};

use ndarray::{s, Array1, Array3, ArrayView1};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::data::Data;
use crate::filters::Weights;

/// Calculate histograms and output the result and time taken.
#[inline(always)]
pub fn calculate_histograms(
    dataset: Data,
    n_spec: usize,
    n_periods: usize,
    periods: Array1<usize>,
    weights: Weights,
) -> (HistogramResult, Duration) {
    let time = Instant::now();

    let min_time: f64 = 0.;
    let max_time: f64 = 32.768;
    let width: f64 = 0.016;
    let n_bins = ((max_time - min_time) / width).floor() as usize;

    // iterate over the data chunks, make histograms for each, then sum histograms at the end
    let results: HistogramResult = (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .map(|start| {
            let end = start + min(dataset.chunk_size, dataset.n_events - start);
            let array_slice = s![start..end];
            unsafe {
                make_histogram(
                    dataset
                        .times
                        .read_slice_1d(array_slice)
                        .expect("Failed to read times."),
                    dataset
                        .specs
                        .read_slice_1d(array_slice)
                        .expect("Failed to read specs."),
                    n_spec,
                    &periods.slice(array_slice),
                    n_periods,
                    weights.slice(start, end),
                    0.,
                    32.768,
                    0.016,
                    1e-3,
                )
            }
        })
        .reduce(
            || HistogramResult::new(n_periods, n_spec, n_bins),
            |mut acc, r| {
                acc.hist += &r.hist;
                acc.n += &r.n;
                acc
            },
        );

    (results, time.elapsed())
}

pub struct HistogramResult {
    pub hist: Array3<usize>,
    pub n: usize,
}

impl HistogramResult {
    fn new(n_periods: usize, n_spec: usize, n_bins: usize) -> HistogramResult {
        HistogramResult {
            hist: Array3::<usize>::zeros((n_periods, n_spec, n_bins)),
            n: 0,
        }
    }
}

/// Make a histogram for a set of data.
/// This function is unsafe because we do array indexing without bounds checks!
#[inline(always)]
unsafe fn make_histogram(
    times: Array1<u32>,
    specs: Array1<u32>,
    n_spec: usize,
    periods: &ArrayView1<usize>,
    n_periods: usize,
    weights: Weights,
    min_time: f32,
    max_time: f32,
    width: f32,
    conversion: f32,
) -> HistogramResult {
    let mut result = HistogramResult::new(
        n_periods,
        n_spec,
        ((max_time - min_time) / width).floor() as usize,
    );

    for (k, time) in times.into_iter().enumerate() {
        let t = time as f32 * conversion;
        let w_k = weights[k];

        if w_k && (t >= min_time) && (t <= max_time) {
            let bin = ((t - min_time) / width).floor() as usize;
            result.hist[[*periods.uget(k) as usize, *specs.uget(k) as usize, bin]] += 1;
            result.n += 1
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test a histogram with no filters is correctly constructed.
    #[test]
    fn test_hist_no_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let periods = Array1::zeros(6);
        let weights = Weights::ones(6);

        unsafe {
            let result = make_histogram(
                times,
                specs,
                2,
                &periods.slice(s![0..=5]),
                1,
                weights,
                0.,
                3.,
                1.,
                1e-3,
            );

            let expected =
                Array3::<usize>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();

            assert_eq!(result.hist, expected)
        }
    }

    /// Test a histogram with filters is correctly constructed.
    #[test]
    fn test_hist_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let periods = Array1::zeros(6);
        let weights: [bool; 6] = [false, true, true, false, false, true];

        unsafe {
            let result = make_histogram(
                times,
                specs,
                2,
                &periods.slice(s![0..=5]),
                1,
                weights.into_iter().into(),
                0.,
                3.,
                1.,
                1e-3,
            );

            let expected =
                Array3::<usize>::from_shape_vec((1, 2, 3), vec![0, 1, 0, 1, 0, 1]).unwrap();

            assert_eq!(result.hist, expected)
        }
    }
}
