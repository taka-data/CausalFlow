use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use causalflow_core::forest::CausalForest;
use causalflow_core::validation::validate_causal_structure;

#[pyfunction]
fn analyze_flow() -> PyResult<String> {
    Ok("Rust core analysis results".to_string())
}

#[pyclass]
struct InferenceResult {
    #[pyo3(get)]
    pub mean_effect: f64,
    #[pyo3(get)]
    pub predictions: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub confidence_intervals: Vec<(f64, f64)>,
}

#[pymethods]
impl InferenceResult {
    fn __repr__(&self, py: Python) -> String {
        let mut table = String::new();
        table.push_str("+----------------------------+----------------+\n");
        table.push_str("| Metric                     | Value          |\n");
        table.push_str("+----------------------------+----------------+\n");
        table.push_str(&format!("| Average Treatment Effect   | {:14.4} |\n", self.mean_effect));
        let num_obs = self.predictions.as_ref(py).len();
        table.push_str(&format!("| Number of Observations     | {:14} |\n", num_obs));
        table.push_str("+----------------------------+----------------+\n");
        table.push_str("\n(Sample Confidence Intervals)\n");
        for (i, ci) in self.confidence_intervals.iter().take(3).enumerate() {
            table.push_str(&format!("Sample {}: [{:.4}, {:.4}]\n", i, ci.0, ci.1));
        }
        if self.confidence_intervals.len() > 3 {
            table.push_str("...\n");
        }
        table
    }
}

#[pyclass]
struct ValidationResult {
    #[pyo3(get)]
    pub is_robust: bool,
    #[pyo3(get)]
    pub message: String,
}

#[pyclass]
struct Model {
    inner: CausalForest,
}

#[pymethods]
impl Model {
    fn estimate_effects(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<InferenceResult> {
        let x_node = x.as_array().to_owned();
        let core_res = self.inner.predict(&x_node);
        
        Ok(InferenceResult {
            mean_effect: core_res.mean_effect,
            predictions: core_res.predictions.to_pyarray(py).to_owned(),
            confidence_intervals: core_res.confidence_intervals,
        })
    }

    #[pyo3(signature = (n_folds = 5, is_time_series = false))]
    fn validate(&self, n_folds: usize, is_time_series: bool) -> PyResult<ValidationResult> {
        let res = validate_causal_structure(n_folds, is_time_series);
        Ok(ValidationResult {
            is_robust: res.is_robust,
            message: res.message,
        })
    }

    fn plot_importance(&self) {
        println!("Placeholder: Plotting feature importance for the Causal Forest...");
    }

    fn plot_effects(&self) {
        println!("Placeholder: Plotting individual treatment effects distribution...");
    }
}

#[pyfunction]
#[pyo3(signature = (features, treatment, outcome))]
fn create_model(
    features: PyReadonlyArray2<f64>,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>
) -> PyResult<Model> {
    let mut forest = CausalForest::new(10, 5, 5);
    forest.fit(&features.as_array().to_owned(), &treatment.as_array().to_owned(), &outcome.as_array().to_owned());
    Ok(Model { inner: forest })
}

#[pymodule]
fn causalflow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_flow, m)?)?;
    m.add_function(wrap_pyfunction!(create_model, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<InferenceResult>()?;
    m.add_class::<ValidationResult>()?;
    Ok(())
}
