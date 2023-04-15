use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[pyclass]
pub struct KMeansRust {
    k: usize,
    num_iter: usize,
    centroids: Vec<f64>,
    rng: ChaCha8Rng,
}

#[pymethods]
impl KMeansRust {
    #[new]
    fn new() -> Self {
        KMeansRust {
            k: 0,
            num_iter: 0,
            centroids: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(0),
        }
    }

    fn with_k(&mut self, k: usize) -> PyResult<()> {
        self.k = k;
        Ok(())
    }

    fn with_num_iter(&mut self, num_iter: usize) -> PyResult<()> {
        self.num_iter = num_iter;
        Ok(())
    }

    fn with_random_state(&mut self, random_state: usize) -> PyResult<()> {
        self.rng = ChaCha8Rng::seed_from_u64(random_state as u64);
        Ok(())
    }

    fn get_centroids(&self) -> PyResult<Vec<f64>> {
        Ok(self.centroids.to_vec())
    }

    fn fit(&mut self, x: Vec<f64>) -> PyResult<()> {
    }
}