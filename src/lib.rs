use std::cmp::min;
use std::iter::Iterator;
use std::path::Path;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use numpy::ToPyArray;
use hdf5::{Dataset, Error, File};
use ndarray::{Array1, Array3, ArrayView1, s};
use numpy::PyArray3;
use rayon::prelude::{ParallelIterator, IntoParallelIterator, IndexedParallelIterator};

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
    let weight = Array1::<usize>::ones(data.n_events);

    let (result, time) = calculate_histograms(
        data,
        n_specs,
        n_periods,
        periods,
        weight,
    );
    Ok((result.hist.to_pyarray(py), result.n, time.as_millis()))
}

/// Calculate histograms and output the result and time taken.
fn calculate_histograms(
    dataset: Data,
    n_spec: usize,
    n_periods: usize,
    periods: Array1<usize>,
    weight: Array1<usize>,
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
                dataset.times
                    .read_slice_1d(array_slice)
                    .expect("Failed to read times."),
                dataset.specs
                    .read_slice_1d(array_slice)
                    .expect("Failed to read specs."),
                n_spec,
                &periods.slice(array_slice),
                n_periods,
                &weight.slice(array_slice),
                0.,
                32.768,
                0.016,
                1e-3,
            )
        }})
        .reduce(|| HistogramResult::new(n_periods, n_spec, n_bins),
                |mut acc, r| {
                    acc.hist += &r.hist;
                    acc.n += &r.n;
                    acc
                });

    (results, time.elapsed())
}

struct Data {
    pub times: Dataset,
    pub specs: Dataset,
    pub amps: Dataset,
    pub n_events: usize,   // the total number of events
    pub chunk_size: usize, // the size of the data chunks
}

fn load_data(filename: &Path, chunk_size: usize) -> Result<Data, Error> {
    let file = File::open(filename)?;
    let data = file.group("raw_data_1")?.group("detector_1_events")?;

    let times = data.dataset("event_time_offset")?;
    let specs = data.dataset("event_id")?;
    let amps = data.dataset("pulse_height")?;

    let n_events = specs.size();

    Ok(Data {
        times,
        specs,
        amps,
        n_events,
        chunk_size,
    })
}

pub struct HistogramResult {
    pub hist: Array3<usize>,
    pub n: usize,
}

impl HistogramResult {
    fn new(n_periods: usize, n_spec: usize, n_bins: usize) -> HistogramResult {
        HistogramResult {
            hist: Array3::<usize>::zeros((n_periods, n_spec, n_bins)),
            n: 0
        }
    }
}

/// Make a histogram for a set of data.
/// This function is unsafe because we do array indexing without bounds checks!
unsafe fn make_histogram(
    times: Array1<u32>,
    specs: Array1<u32>,
    n_spec: usize,
    periods: &ArrayView1<usize>,
    n_periods: usize,
    weight: &ArrayView1<usize>,
    min_time: f32,
    max_time: f32,
    width: f32,
    conversion: f32,
) -> HistogramResult {
    let mut result = HistogramResult::new(n_periods, n_spec, ((max_time - min_time)/width).floor() as usize);

    for (k, time) in times.into_iter().enumerate() {
        let t = time as f32 * conversion;
        let w_k = weight.uget(k);

        if (*w_k != 0) && (t >= min_time) && (t <= max_time) {
            let bin = ((t - min_time) / width).floor() as usize;
            result.hist[[*periods.uget(k) as usize, *specs.uget(k) as usize, bin]] += w_k;
            result.n += w_k
        }
    }
    result 
}

/// A Python module implemented in Rust.
#[pymodule]
fn stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_histogram, m)?)?;
    Ok(())
}
