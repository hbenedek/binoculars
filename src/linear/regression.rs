use pyo3::prelude::*;
use ndarray::{Array, Array1, array};
use ndarray_linalg::LeastSquaresSvd;
use ndarray_linalg::Inverse;
use crate::linear::utils::{add_bias, sigmoid, compute_logistic_gradient, compute_logistic_loss, init_vector};


#[pyclass]
pub struct LinearRegressionRust {
    weights: Array1<f64>,
    schema: Vec<String>,
    with_bias: bool,
    method: String,
}

#[pyclass]
pub struct LogisticRegressionRust {
    weights: Array1<f64>,
    schema: Vec<String>,
    with_bias: bool,
    method: String,
    iter: usize,
    loss: f64,
}

#[pymethods]
impl LinearRegressionRust {
    #[new]
    fn new() -> Self {
        LinearRegressionRust { weights: array![], schema: Vec::new(), with_bias: false, method: "".to_string() }
    }

    fn get_weights(&self) -> PyResult<Vec<f64>> {
        Ok(self.weights.to_vec())
    }

    fn get_weights_dict(&self) -> PyResult<Vec<(String, f64)>> {
        let mut weights = Vec::new();
        if self.schema.len() == self.weights.len() {
            for (i, w) in self.weights.iter().enumerate() {
                weights.push((self.schema[i].clone(), w.to_owned()));
            }
        }
        Ok(weights)
        }

    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        let weights = Array1::from(weights);
        self.weights = weights;
        Ok(())
    }

    fn with_bias(&mut self, with_bias: bool) -> PyResult<()> {
        self.with_bias = with_bias;
        // self.schema.push("bias".to_string());
        Ok(())
    }

    fn with_method(&mut self, method: String) -> PyResult<()> {
        self.method = method;
        Ok(())
    }

    // fn with_schema(&mut self, schema: Vec<String>) -> PyResult<()> {
    //     self.schema = schema;
    //     Ok(())
    // }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let x = add_bias(x, self.with_bias);
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        let y = Array1::from(y);
        let mut weights = Vec::new();
        if self.method == String::from("ls") {
            weights = x.least_squares(&y).unwrap().solution.to_vec();
        } else if self.method == String::from("normal") {
            weights = (x.t().dot(&x)).inv().unwrap().dot(&x.t()).dot(&y).to_vec();
        }
        self.set_weights(weights).unwrap();
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let x = add_bias(x, self.with_bias);
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        x.dot(&self.weights).to_vec()
    }
}

#[pymethods]
impl LogisticRegressionRust {
    #[new]
    fn new() -> Self {
        LogisticRegressionRust { weights: array![], schema: Vec::new(), with_bias: false, method: "".to_string(), iter: 0, loss: 0.0}
    }

    fn get_loss(&self) -> PyResult<f64> {
        Ok(self.loss)
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

    fn with_iter(&mut self, iter: usize) -> PyResult<()> {
        self.iter = iter;
        Ok(())
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let x = add_bias(x, self.with_bias);
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        let y = Array1::from(y);
        let mut weights = init_vector(x.ncols());
        if self.method == String::from("gd") {
            for _ in 0..self.iter {
                let gradient = compute_logistic_gradient(&x, &y, &weights);
            weights = weights - gradient;
            }
        }
        self.set_weights(weights.to_vec()).unwrap();
        self.loss = compute_logistic_loss(&x, &y, &weights);
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let x = add_bias(x, self.with_bias);
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        let mu = x.dot(&self.weights);
        mu.mapv(sigmoid).to_vec()
    }

}
