use std::path::Path;

use hdf5::{Dataset, Error, File};
use pyo3::prelude::{pyclass, pymethods};

/// Class for storing a Nexus event file.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Data {
    pub times: Dataset,
    pub specs: Dataset,
    pub amps: Dataset,
    pub n_events: usize,   // the total number of events
    pub chunk_size: usize, // the size of the data chunks
}

#[pymethods]
impl Data {
    #[new]
    #[pyo3(signature = (filename, chunk_size=1048576))]
    fn new(filename: String, chunk_size: usize) -> Self {
        let path = Path::new(&filename);
        load_data(path, chunk_size).expect("Failed to load data!")
    }
}

pub fn load_data(filename: &Path, chunk_size: usize) -> Result<Data, Error> {
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
