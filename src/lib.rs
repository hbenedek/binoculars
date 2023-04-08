use ndarray_linalg::LeastSquaresSvd;
use pyo3::prelude::*;
use ndarray::{Array, Array1, Array2, stack, Axis, array};

#[pyclass]
struct LinearModelRust {
    weights: Array1<f64>,
    schema: Vec<String>,
    with_bias: bool,
    method: String,
}

#[pymethods]
impl LinearModelRust {
    #[new]
    fn new() -> Self {
        LinearModelRust { weights: array![], schema: Vec::new(), with_bias: false, method: "".to_string() }
    }

    fn get_weights(&self) -> PyResult<Vec<f64>> {
        Ok(self.weights.to_vec())
    }

    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        let weights = Array1::from(weights);
        self.weights = weights;
        Ok(())
    }

    fn with_bias(&mut self, with_bias: bool) -> PyResult<()> {
        self.with_bias = with_bias;
        Ok(())
    }

    fn with_method(&mut self, method: String) -> PyResult<()> {
        self.method = method;
        Ok(())
    }

    fn with_schema(&mut self, schema: Vec<String>) -> PyResult<()> {
        self.schema = schema;
        Ok(())
    }

    fn fit(&mut self, mut x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        if self.with_bias {
            for p in x.iter_mut() {
                p.push(1.0)
            }
        }
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        let y = Array1::from(y);
        let weights = x.least_squares(&y).unwrap().solution.to_vec();
        self.set_weights(weights);
        Ok(())
    }

    fn predict(&self, mut x: Vec<Vec<f64>>) -> Vec<f64> {
        if self.with_bias {
            for p in x.iter_mut() {
                p.push(1.0)
            }
        }
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        x.dot(&self.weights).to_vec()
    }
}

/// This module is implemented in Rust.
#[pymodule]
#[pyo3(name = "binoculars")]
fn my_extension(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<LinearModelRust>()?;
    Ok(())
}