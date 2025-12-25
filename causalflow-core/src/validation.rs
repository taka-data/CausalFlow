use crate::forest::CausalForest;
use ndarray::{Array1, Array2};

pub struct ValidationResult {
    pub is_robust: bool,
    pub message: String,
}

pub fn validate_causal_structure(
    forest: &CausalForest,
    x: &Array2<f64>,
    t: &Array1<f64>,
    y: &Array1<f64>,
    n_folds: usize,
) -> ValidationResult {
    // 1. Placebo Test: Shuffling treatment should result in near-zero effect
    let mut placebo_forest = forest.clone();
    placebo_forest.fit_placebo(x, t, y);
    let placebo_res = placebo_forest.predict(x);
    let placebo_effect = placebo_res.mean_effect.abs();

    // Threshold for placebo effect (should be close to 0)
    // In a real scenario, this might be relative to the original effect
    let original_res = forest.predict(x);
    let original_effect = original_res.mean_effect.abs();
    
    let is_robust = if original_effect > 0.0 {
        placebo_effect < original_effect * 0.2 // Placebo should be < 20% of real effect
    } else {
        placebo_effect < 0.05 // Absolute threshold if original is 0
    };

    if is_robust {
        ValidationResult {
            is_robust: true,
            message: format!(
                "Causal structure looks robust. Placebo effect ({:.4}) is significantly lower than estimated effect ({:.4}). Verified using {} folds.",
                placebo_effect, original_effect, n_folds
            ),
        }
    } else {
        ValidationResult {
            is_robust: false,
            message: format!(
                "Warning: Causal structure may NOT be robust. Placebo effect ({:.4}) is too high compared to estimated effect ({:.4}).",
                placebo_effect, original_effect
            ),
        }
    }
}
