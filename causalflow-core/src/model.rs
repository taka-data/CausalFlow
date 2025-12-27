use crate::forest::InferenceResult;
use crate::errors::Result;
use ndarray::{ArrayView1, ArrayView2};

pub trait CausalModel: Send + Sync {
    fn fit(&mut self, x: ArrayView2<f64>, t: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<()>;
    fn predict(&self, x: ArrayView2<f64>) -> Result<InferenceResult>;
}
