use crate::model::CausalModel;
use crate::forest::InferenceResult;
use crate::errors::Result;
use ndarray::{Array1, ArrayView1, ArrayView2};

#[derive(Clone)]
pub struct LinearCausalModel {
    pub coef: f64,
}

impl LinearCausalModel {
    pub fn new() -> Self {
        Self { coef: 0.0 }
    }
}

impl CausalModel for LinearCausalModel {
    fn fit(&mut self, _x: ArrayView2<f64>, t: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<()> {
        // Dummy implementation: ATE estimation
        let mut y1_sum = 0.0;
        let mut y1_count = 0;
        let mut y0_sum = 0.0;
        let mut y0_count = 0;

        for (i, &val) in t.iter().enumerate() {
            if val > 0.5 {
                y1_sum += y[i];
                y1_count += 1;
            } else {
                y0_sum += y[i];
                y0_count += 1;
            }
        }

        if y1_count > 0 && y0_count > 0 {
            self.coef = (y1_sum / y1_count as f64) - (y0_sum / y0_count as f64);
        } else {
            self.coef = 0.0;
        }

        Ok(())
    }

    fn predict(&self, x: ArrayView2<f64>) -> Result<InferenceResult> {
        let n_samples = x.nrows();
        let predictions = Array1::from_elem(n_samples, self.coef);
        
        Ok(InferenceResult {
            predictions,
            mean_effect: self.coef,
            confidence_intervals: vec![(self.coef - 0.1, self.coef + 0.1); n_samples],
            feature_importance: vec![0.0; x.ncols()],
        })
    }
}
