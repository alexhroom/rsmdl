use ndarray::Array1;
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::*;

mod stats;
use stats::calculate_histograms;
mod data;
use data::Data;
mod filters;
use filters::{get_good_values, Weights};

// type for python-bound histogram
type PyHist<'py> = Bound<'py, PyArray3<usize>>;

#[pyfunction]
fn calc_histogram<'py>(
    py: Python<'py>,
    data: Data,
    n_filters: usize,
) -> PyResult<(PyHist<'py>, usize, u128)> {
    let n_specs = 960;

    // todo: this needs to become actual filtering code to get the weights
    let n_periods = 1;
    let periods = Array1::<usize>::zeros(data.n_events);

    // currently, filter out every other frame for testing
    let weights = match n_filters > 0 {
        true => {
            let f_starts = (0..).step_by(2).take(n_filters).collect();
            let f_ends = (1..).step_by(2).take(n_filters).collect();
            let frame_step = data.n_events / (2 * n_filters);
            let frame_starts = (0..=n_filters * 2).map(|n| n * frame_step).collect();
            get_good_values(f_starts, f_ends, frame_starts, data.n_events)
        }
        false => Weights::ones(data.n_events),
    };

    let (result, time) = calculate_histograms(data, n_specs, n_periods, periods, weights);
    Ok((result.hist.to_pyarray(py), result.n, time.as_millis()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsmdl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_histogram, m)?)?;
    m.add_class::<Data>()?;
    Ok(())
}
