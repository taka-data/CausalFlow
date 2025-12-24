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
