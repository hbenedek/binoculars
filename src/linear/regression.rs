use crate::linear::utils::{
    add_bias, compute_logistic_gradient, compute_logistic_loss, init_vector, sigmoid,
};
use ndarray::{array, Array, Array1};
use ndarray_linalg::Inverse;
use ndarray_linalg::LeastSquaresSvd;
use pyo3::prelude::*;

#[pyclass]
pub struct LinearRegressionRust {
    method: String,
    with_bias: bool,
    weights: Array1<f64>,
}

#[pyclass]
pub struct LogisticRegressionRust {
    method: String,
    with_bias: bool,
    epoch: usize,
    batch: usize,
    learning_rate: f64,
    losses: Vec<f64>,
    weights: Array1<f64>,
}

#[pymethods]
impl LinearRegressionRust {
    #[new]
    fn new(method: String, with_bias: bool) -> Self {
        LinearRegressionRust {
            method,
            with_bias,
            weights: array![],
        }
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
        // self.schema.push("bias".to_string());
        Ok(())
    }

    fn with_method(&mut self, method: String) -> PyResult<()> {
        self.method = method;
        Ok(())
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let x = add_bias(x, self.with_bias);
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())
            .unwrap();
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
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())
            .unwrap();
        x.dot(&self.weights).to_vec()
    }
}

#[pymethods]
impl LogisticRegressionRust {
    #[new]
    fn new(method: String, with_bias: bool, epoch: usize, batch: usize) -> Self {
        LogisticRegressionRust {
            method,
            with_bias,
            epoch,
            batch,
            weights: array![],
            losses: Vec::new(),
            learning_rate: 0.0,
        }
    }

    fn get_losses(&self) -> PyResult<Vec<f64>> {
        Ok(self.losses.to_vec())
    }

    fn get_weights(&self) -> PyResult<Vec<f64>> {
        Ok(self.weights.to_vec())
    }

    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        let weights = Array1::from(weights);
        self.weights = weights;
        Ok(())
    }

    fn with_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        self.learning_rate = lr;
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

    fn with_epochs(&mut self, epoch: usize) -> PyResult<()> {
        self.epoch = epoch;
        Ok(())
    }

    fn with_batch_size(&mut self, batch: usize) -> PyResult<()> {
        self.batch = batch;
        Ok(())
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let x = add_bias(x, self.with_bias);
        let mut weights = init_vector(x[0].len());
        self.losses = Vec::new();
        if self.method == String::from("gd") {
            for _ in 0..self.epoch {
                let mut loss = 0.0;
                let x_chunks = x.chunks(self.batch).map(|chunk| chunk.to_vec());
                let y_chunks = y.chunks(self.batch).map(|chunk| chunk.to_vec());
                for (xb, yb) in x_chunks.zip(y_chunks) {
                    let xb = Array::from_shape_vec(
                        (xb.len(), xb[0].len()),
                        xb.into_iter().flatten().collect(),
                    )
                    .unwrap();
                    let yb = Array1::from(yb);
                    let gradient = compute_logistic_gradient(&xb, &yb, &weights);
                    weights = weights - self.learning_rate * gradient;
                    loss += compute_logistic_loss(&xb, &yb, &weights);
                }
                self.losses.push(loss);
            }
        }
        self.set_weights(weights.to_vec()).unwrap();
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let x = add_bias(x, self.with_bias);
        let x = Array::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())
            .unwrap();
        let mu = x.dot(&self.weights);
        mu.mapv(sigmoid).to_vec()
    }
}
