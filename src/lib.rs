use std::path::Path;

use ndarray::Array1;
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::*;

mod stats;
use stats::calculate_histograms;
mod data;
use data::load_data;
mod filters;
use filters::Weights;

// type for python-bound histogram
type PyHist<'py> = Bound<'py, PyArray3<usize>>;

#[pyfunction]
fn calc_histogram<'py>(py: Python<'py>, file: String) -> PyResult<(PyHist<'py>, usize, u128)> {
    let chunk_size: usize = 1048576;

    // load file
    let path = Path::new(&file);
    let data = load_data(&path, chunk_size).expect("Failed to load data!");

    let n_specs = 960;

    // todo: this needs to become actual filtering code to get the weights
    let n_periods = 1;
    let periods = Array1::<usize>::zeros(data.n_events);
    let weights = Weights::ones(data.n_events);

    let (result, time) = calculate_histograms(data, n_specs, n_periods, periods, weights);
    Ok((result.hist.to_pyarray(py), result.n, time.as_millis()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsmdl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_histogram, m)?)?;
    Ok(())
}
