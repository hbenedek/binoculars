use ndarray_linalg::LeastSquaresSvd;
use pyo3::prelude::*;
use ndarray::{Array, Array1};

#[pyfunction]
fn _least_square(x: Vec<Vec<f64>>, y: Vec<f64>) ->  Vec<f64> {
    let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
    let y = Array1::from(y);
    x.least_squares(&y).unwrap().solution.to_vec()
}

#[pyfunction]
fn _predict(x: Vec<Vec<f64>>, w: Vec<f64>) -> Vec<f64> {
    let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
    let w = Array1::from(w);
    x.dot(&w).to_vec()
}

#[pyfunction]
fn test() -> String {
    "Hello, world!".to_string()
}

/// This module is implemented in Rust.
#[pymodule]
#[pyo3(name = "binoculars")]
fn my_extension(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_least_square, m)?)?;
    m.add_function(wrap_pyfunction!(test, m)?)?;
    m.add_function(wrap_pyfunction!(_predict, m)?)?;
    Ok(())
}