use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use causalflow_core::forest::CausalForest;
use causalflow_core::validation::validate_causal_structure;

#[pyfunction]
fn analyze_flow() -> PyResult<String> {
    Ok("Rust core analysis results".to_string())
}

#[pyclass]
pub struct InferenceResult {
    #[pyo3(get)]
    pub mean_effect: f64,
    #[pyo3(get)]
    pub predictions: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub confidence_intervals: Vec<(f64, f64)>,
    #[pyo3(get)]
    pub feature_importance: Vec<f64>,
    pub feature_names: Option<Vec<String>>,
}

use causalflow_core::visualization::{VisualOutput, NodeInfo, LinkInfo};

#[pymethods]
impl InferenceResult {
    fn to_visual_tag(&self) -> String {
        let labels = self.feature_names.clone().unwrap_or_else(|| {
            (0..self.feature_importance.len()).map(|i| format!("Feature {}", i)).collect()
        });
        let visual = VisualOutput::feature_importance(labels, self.feature_importance.clone());
        format!("```json:causal-plot\n{}\n```", visual.to_json())
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let labels = self.feature_names.clone().unwrap_or_else(|| {
            (0..self.feature_importance.len()).map(|i| format!("Feature {}", i)).collect()
        });
        let visual = VisualOutput::feature_importance(labels, self.feature_importance.clone());
        let json_str = visual.to_json();
        let json_module = py.import("json")?;
        let dict = json_module.call_method1("loads", (json_str,))?;
        Ok(dict.to_object(py))
    }

    fn __repr__(&self, py: Python) -> String {
        self.summary(py)
    }

    fn summary(&self, py: Python) -> String {
        let mut table = String::new();
        table.push_str("+----------------------------+----------------+\n");
        table.push_str("| Metric                     | Value          |\n");
        table.push_str("+----------------------------+----------------+\n");
        table.push_str(&format!("| Average Treatment Effect   | {:14.4} |\n", self.mean_effect));
        let num_obs = self.predictions.as_ref(py).len();
        table.push_str(&format!("| Number of Observations     | {:14} |\n", num_obs));
        table.push_str("+----------------------------+----------------+\n");
        
        table.push_str("\n[Feature Importance]\n");
        if let Some(names) = &self.feature_names {
            for (i, &imp) in self.feature_importance.iter().enumerate() {
                let name = names.get(i).cloned().unwrap_or_else(|| format!("Feature {}", i));
                table.push_str(&format!("{:<20}: {:.4}\n", name, imp));
            }
        } else {
            for (i, &imp) in self.feature_importance.iter().enumerate() {
                table.push_str(&format!("Feature {:<12}: {:.4}\n", i, imp));
            }
        }

        table.push_str("\n[Interpretation]\n");
        if self.mean_effect > 0.0 {
            table.push_str(&format!(
                "The treatment has a POSITIVE average effect of {:.4}.\n",
                self.mean_effect
            ));
        } else if self.mean_effect < 0.0 {
            table.push_str(&format!(
                "The treatment has a NEGATIVE average effect of {:.4}.\n",
                self.mean_effect
            ));
        } else {
            table.push_str("The treatment has NO average effect on the outcome.\n");
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
#[derive(Clone)]
struct Model {
    inner: CausalForest,
    feature_names: Option<Vec<String>>,
}

impl Model {
    fn get_visual(&self, plot_type: &str) -> VisualOutput {
        match plot_type {
            "graph" => {
                let mut nodes = Vec::new();
                let mut links = Vec::new();

                nodes.push(NodeInfo { 
                    id: "Treatment".to_string(), 
                    label: "Treatment".to_string(),
                    role: "treatment".to_string(),
                    value: 1.0 
                });
                nodes.push(NodeInfo { 
                    id: "Outcome".to_string(), 
                    label: "Outcome".to_string(),
                    role: "outcome".to_string(),
                    value: 1.0 
                });
                links.push(LinkInfo { source: "Treatment".to_string(), target: "Outcome".to_string(), weight: 1.0 });

                if let Some(names) = &self.feature_names {
                    for name in names {
                        if name != "Treatment" && name != "Outcome" {
                            nodes.push(NodeInfo { 
                                id: name.clone(), 
                                label: name.clone(),
                                role: "confounder".to_string(),
                                value: 0.5
                            });
                            links.push(LinkInfo { source: name.clone(), target: "Treatment".to_string(), weight: 0.5 });
                            links.push(LinkInfo { source: name.clone(), target: "Outcome".to_string(), weight: 0.5 });
                        }
                    }
                }
                VisualOutput::causal_graph(nodes, links)
            },
            "effect_dist" => {
                // Mock distribution data for now
                VisualOutput::effect_dist(
                    "Standard Causal Effect".to_string(),
                    "Frequency".to_string(),
                    vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    vec![10, 50, 100, 40, 5]
                )
            },
            _ => VisualOutput::feature_importance(vec![], vec![])
        }
    }
}

#[pymethods]
impl Model {
    fn to_visual_tag(&self, plot_type: &str) -> String {
        let visual = self.get_visual(plot_type);
        format!("```json:causal-plot\n{}\n```", visual.to_json())
    }

    fn estimate_effects(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<InferenceResult> {
        let x_node = x.as_array().to_owned();
        let core_res = self.inner.predict(&x_node);
        
        Ok(InferenceResult {
            mean_effect: core_res.mean_effect,
            predictions: core_res.predictions.to_pyarray(py).to_owned(),
            confidence_intervals: core_res.confidence_intervals,
            feature_importance: core_res.feature_importance,
            feature_names: self.feature_names.clone(),
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
#[pyo3(signature = (features, treatment, outcome, feature_names = None))]
fn create_model(
    features: PyReadonlyArray2<f64>,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
    feature_names: Option<Vec<String>>,
) -> PyResult<Model> {
    let mut forest = CausalForest::new(10, 5, 5);
    forest.fit(&features.as_array().to_owned(), &treatment.as_array().to_owned(), &outcome.as_array().to_owned());
    Ok(Model { 
        inner: forest,
        feature_names,
    })
}

#[pyfunction]
#[pyo3(signature = (model, plot = "graph"))]
fn plot_model(py: Python, model: Model, plot: &str) -> PyResult<PyObject> {
    let visual = model.get_visual(plot);
    let json_str = visual.to_json();
    let json_module = py.import("json")?;
    let dict = json_module.call_method1("loads", (json_str,))?;
    Ok(dict.to_object(py))
}

#[pymodule]
fn causalflow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_flow, m)?)?;
    m.add_function(wrap_pyfunction!(create_model, m)?)?;
    m.add_function(wrap_pyfunction!(plot_model, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<InferenceResult>()?;
    m.add_class::<ValidationResult>()?;
    Ok(())
}
