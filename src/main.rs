use std::cmp::min;
use std::ops::Add;
use std::iter::{Iterator, Sum};
use std::path::Path;
use std::time::{Duration, Instant};

use hdf5::{Dataset, Error, File};
use ndarray::{Array1, Array3, ArrayView1, s};
use indicatif::ProgressIterator;
use tabled::builder::Builder;
use rayon::prelude::{ParallelIterator, IntoParallelIterator, IndexedParallelIterator};

fn main() {
    let files = ["HIFI00206202.nxs"];
    let mut results_builder = Builder::new();
    results_builder.push_record(["File", "Run time (ms)"]);

    for file in files {
        let avg_time = run_benchmark(file.to_string());
        results_builder.push_record([file, &avg_time.to_string()]) 
    }
    let table = results_builder.build();
    println!("{}", table)
}

fn run_benchmark(file: String) -> u128 {
    let stats: usize = 20;
    let chunk_size: usize = 1048576;

    // load file, measuring time taken...
    let path = Path::new(&file);
    let data = load_data(&path, chunk_size).expect("Failed to load data!");

    let n_specs = 960;

    // todo: this needs to become actual filtering code to get the weights
    let n_periods = 1;
    let periods = Array1::<usize>::zeros(data.n_events);
    let weight = Array1::<usize>::ones(data.n_events);

    let mut avg_time: u128 = 0;

    // calculate histograms `stats` times
    for _ in (0..stats).progress() {
        let (_, time) = calculate_histograms(
            data.clone(),
            n_specs,
            n_periods,
            periods.clone(),
            weight.clone(),
        );
        avg_time += time.as_millis();
    }
    avg_time / stats as u128
}

/// Calculate histograms and output the result and time taken.
fn calculate_histograms(
    dataset: Data,
    n_specs: usize,
    n_periods: usize,
    periods: Array1<usize>,
    weight: Array1<usize>,
) -> (HistogramResult, Duration) {
    let time = Instant::now();

    // iterate over the data chunks, make histograms for each, then sum histograms at the end
    let results: Vec<HistogramResult> = (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .map(|start| {
            let end = start + min(dataset.chunk_size, dataset.n_events - start);
            let array_slice = s![start..end];
            make_histogram(
            dataset.times
                .read_slice_1d(array_slice)
                .expect("Failed to read times."),
            dataset.specs
                .read_slice_1d(array_slice)
                .expect("Failed to read specs."),
                n_specs,
                &periods.slice(array_slice),
                n_periods,
                &weight.slice(array_slice),
                0.,
                32.768,
                0.016,
                1e-3,
            )
        })
        .collect();

    let final_result = results
        .into_iter()
        .reduce(|acc, r| HistogramResult{hist: acc.hist + r.hist, n: acc.n + r.n})
        .unwrap();

    (final_result, time.elapsed())
}

struct Data {
    pub times: Dataset,
    pub specs: Dataset,
    pub amps: Dataset,
    pub n_events: usize,   // the total number of events
    pub chunk_size: usize, // the size of the data chunks
}

impl Clone for Data {
    fn clone(&self) -> Self {
        Data {
            times: self.times.clone(),
            specs: self.specs.clone(),
            amps: self.amps.clone(),
            n_events: self.n_events,
            chunk_size: self.chunk_size,
        }
    }
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

struct HistogramResult {
    hist: Array3<usize>,
    n: usize,
}

/// Make a histogram for a set of data.
fn make_histogram(
    times: Array1<u32>,
    specs: Array1<u32>,
    n_spec: usize,
    periods: &ArrayView1<usize>,
    n_periods: usize,
    weight: &ArrayView1<usize>,
    min_time: f64,
    max_time: f64,
    width: f64,
    conversion: f64,
) -> HistogramResult {
    let mut n: usize = 0;
    let bins = Array1::<f64>::range(min_time, max_time, width);
    let mut hist = Array3::<usize>::zeros((n_periods, n_spec, bins.len() - 1));

    let float_times = Array1::<f64>::from_shape_fn(times.len(), |i| conversion * times[i] as f64);
    for (k, time) in float_times.into_iter().enumerate() {
        let w_k = weight[k];

        if (w_k != 0) && (time >= min_time) && (time <= max_time) {
            let bin = ((time - min_time) / width).floor() as usize;
            hist[[periods[k] as usize, specs[k] as usize, bin]] += w_k;
            n += w_k
        }
    }
    HistogramResult { hist, n }
}
