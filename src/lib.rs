use pyo3::prelude::*;

mod stats;
use stats::Histogram;
mod data;
use data::Data;
mod filters;

/// A Python module implemented in Rust.
#[pymodule]
fn rsmdl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Data>()?;
    m.add_class::<Histogram>()?;
    Ok(())
}
