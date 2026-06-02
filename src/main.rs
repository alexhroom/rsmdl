use std::path::Path;
use std::iter::Iterator;

use hdf5::{File, Dataset, Error};
use ndarray::{Array1, Array3};

fn main() {
    let path = Path::new("");
    let data = load_data(&path, 1048576).expect("Failed to load data!");

    let n_specs = 960;
    let n_periods = 1;

    let periods = Array1::<u32>::zeros(data.n_events);
    let weight = Array1::<u32>::ones(data.n_events);

    let results: (Array3<u32>, u32) = data
        .map(|(times, specs, _, _, _)| 
        {make_histogram(
            times,
            specs,
            n_specs,
            &periods,
            n_periods,
            &weight,
            0.,
            32.768,
            0.016,
            1e-3
            )})
        .fold();
}

struct Data {
    pub times: Dataset,
    pub specs: Dataset,
    pub amps: Dataset,
    pub n_events: usize,  // the total number of events
    pub chunk_size: usize,  // the size of the data chunks

    current_start: usize  // the current start index for the chunks
}

impl Iterator for Data {
    type Item = (Array1<u32>, Array1<u32>, Array1<u32>, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_start;
        if start >= self.n_events {
            return None
        }

        let mut end = start + self.chunk_size;

        if end > self.n_events {
            end = self.n_events
        }

        // update start point for next iteration
        self.current_start = end;

        Some((
            self.times.read_slice_1d(start..end).expect("Failed to read times."),
            self.specs.read_slice_1d(start..end).expect("Failed to read specs."),
            self.amps.read_slice_1d(start..end).expect("Failed to read amps."),
            start,
            end 
            ))
    }
}

fn load_data(filename: &Path, chunk_size: usize) -> Result<Data, Error> {
    let file = File::open(filename)?;
    let data = file.group("raw_data_1")?.group("detector_1_events")?;

    let times = data.dataset("event_time_offset")?;
    let specs = data.dataset("event_id")?;
    let amps = data.dataset("pulse_height")?;

    let n_events = specs.size();

    Ok(Data{times, specs, amps, n_events, chunk_size, current_start: 0})
}

fn make_histogram(times: Array1<u32>,
                  specs: Array1<u32>,
                  n_spec: usize,
                  periods: &Array1<u32>,
                  n_periods: usize,
                  weight: &Array1<u32>,
                  min_time: f64,
                  max_time: f64,
                  width: f64,
                  conversion: f64) -> (Array3<u32>, u32) {
    let mut n: u32 = 0;
    let bins = Array1::<f64>::range(min_time, max_time, width);
    let mut hist = Array3::<u32>::zeros((n_periods, n_spec, bins.len()-1));

    let float_times = Array1::<f64>::from_shape_fn(n_spec, |i| {conversion * times[i] as f64});
    for (k, time) in float_times.into_iter().enumerate() {
        let w_k = weight[k];

        if (w_k != 0) && (min_time <= time) && (time <= max_time) {
            let bin = ((time - min_time) / width).floor() as usize;
            hist[[periods[k] as usize, specs[k] as usize, bin]] += w_k;
            n += w_k
        }
    }
    (hist, n) 
}
