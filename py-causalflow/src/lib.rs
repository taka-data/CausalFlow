use causalflow_core::forest::CausalForest;
use causalflow_core::validation::validate_causal_structure;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

use causalflow_core::visualization::{LinkInfo, NodeInfo, VisualOutput};

#[pymethods]
impl InferenceResult {
    #[pyo3(signature = (plot_type = "importance"))]
    fn to_visual_tag(&self, py: Python, plot_type: &str) -> String {
        let visual = self.get_visual(py, plot_type);
        format!("```json:causal-plot\n{}\n```", visual.to_json())
    }

    #[pyo3(signature = (plot_type = "importance"))]
    fn to_dict(&self, py: Python, plot_type: &str) -> PyResult<PyObject> {
        let visual = self.get_visual(py, plot_type);
        let json_str = visual.to_json();
        let json_module = py.import("json")?;
        let dict = json_module.call_method1("loads", (json_str,))?;
        Ok(dict.to_object(py))
    }

    #[pyo3(signature = (plot_type = "importance"))]
    fn show(&self, py: Python, plot_type: &str) {
        println!("{}", self.to_visual_tag(py, plot_type));
    }

    #[pyo3(signature = (plot_type = "importance"))]
    fn preview(&self, py: Python, plot_type: &str) -> PyResult<()> {
        let visual = self.get_visual(py, plot_type);
        render_preview(py, &visual)
    }

    #[pyo3(signature = (plot_type = "importance"))]
    fn to_html(&self, py: Python, plot_type: &str) -> String {
        let visual = self.get_visual(py, plot_type);
        render_html_fragment(&visual)
    }

    fn __repr__(&self, py: Python) -> String {
        self.summary(py)
    }

    fn summary(&self, py: Python) -> String {
        let mut table = String::new();
        table.push_str("+----------------------------+----------------+\n");
        table.push_str("| Metric                     | Value          |\n");
        table.push_str("+----------------------------+----------------+\n");
        table.push_str(&format!(
            "| Average Treatment Effect   | {:14.4} |\n",
            self.mean_effect
        ));
        let num_obs = self.predictions.as_ref(py).len();
        table.push_str(&format!(
            "| Number of Observations     | {:14} |\n",
            num_obs
        ));
        table.push_str("+----------------------------+----------------+\n");

        table.push_str("\n[Feature Importance]\n");
        if let Some(names) = &self.feature_names {
            for (i, &imp) in self.feature_importance.iter().enumerate() {
                let name = names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature {}", i));
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

impl InferenceResult {
    fn get_visual(&self, py: Python, plot_type: &str) -> VisualOutput {
        match plot_type {
            "effect_dist" => {
                let preds_array = self.predictions.as_ref(py);
                let preds = preds_array.to_owned_array().to_vec();

                // Calculate real histogram
                let min = preds.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = preds.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let n_bins = 10;
                let bin_width = if (max - min).abs() < f64::EPSILON {
                    1.0
                } else {
                    (max - min) / n_bins as f64
                };

                let mut bins = Vec::new();
                for i in 0..n_bins {
                    bins.push(min + i as f64 * bin_width);
                }

                let mut counts = vec![0u64; n_bins];
                for &p in &preds {
                    let mut b = if bin_width == 0.0 {
                        0
                    } else {
                        ((p - min) / bin_width) as usize
                    };
                    if b >= n_bins {
                        b = n_bins - 1;
                    }
                    counts[b] += 1;
                }

                VisualOutput::effect_dist(
                    "Individual Treatment Effect Distribution".to_string(),
                    "Frequency".to_string(),
                    bins,
                    counts,
                )
            }
            _ => {
                let labels = self.feature_names.clone().unwrap_or_else(|| {
                    (0..self.feature_importance.len())
                        .map(|i| format!("Feature {}", i))
                        .collect()
                });
                VisualOutput::feature_importance(labels, self.feature_importance.clone())
            }
        }
    }
}

#[pyclass]
struct ValidationResult {
    #[pyo3(get)]
    pub is_robust: bool,
    #[pyo3(get)]
    pub message: String,
}

use causalflow_core::linear::LinearCausalModel;
use causalflow_core::model::CausalModel;

#[derive(Clone)]
enum CausalMethod {
    Forest(CausalForest),
    Linear(LinearCausalModel),
}

impl CausalMethod {
    fn as_trait(&self) -> &dyn CausalModel {
        match self {
            CausalMethod::Forest(f) => f,
            CausalMethod::Linear(l) => l,
        }
    }

    fn as_trait_mut(&mut self) -> &mut dyn CausalModel {
        match self {
            CausalMethod::Forest(f) => f,
            CausalMethod::Linear(l) => l,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct Model {
    method: CausalMethod,
    x: Py<PyArray2<f64>>,
    t: Py<PyArray1<f64>>,
    y: Py<PyArray1<f64>>,
    feature_names: Option<Vec<String>>,
}

impl Model {
    fn get_visual(&self, py: Python, plot_type: &str) -> VisualOutput {
        let x_view = unsafe { self.x.as_ref(py).as_array() };
        match plot_type {
            "graph" => {
                let mut nodes = Vec::new();
                let mut links = Vec::new();

                nodes.push(NodeInfo {
                    id: "Treatment".to_string(),
                    label: "Treatment".to_string(),
                    role: "treatment".to_string(),
                    value: 1.0,
                });
                nodes.push(NodeInfo {
                    id: "Outcome".to_string(),
                    label: "Outcome".to_string(),
                    role: "outcome".to_string(),
                    value: 1.0,
                });
                links.push(LinkInfo {
                    source: "Treatment".to_string(),
                    target: "Outcome".to_string(),
                    weight: 1.0,
                });

                let res = self.method.as_trait().predict(x_view).unwrap_or_else(|_| self.method.as_trait().predict(x_view).unwrap()); // Simplified for visual
                let importance = res.feature_importance;

                if let Some(names) = &self.feature_names {
                    for (i, name) in names.iter().enumerate() {
                        let val = importance.get(i).cloned().unwrap_or(0.1);
                        nodes.push(NodeInfo {
                            id: name.clone(),
                            label: name.clone(),
                            role: "confounder".to_string(),
                            value: val,
                        });
                        links.push(LinkInfo {
                            source: name.clone(),
                            target: "Treatment".to_string(),
                            weight: val,
                        });
                        links.push(LinkInfo {
                            source: name.clone(),
                            target: "Outcome".to_string(),
                            weight: val,
                        });
                    }
                }
                VisualOutput::causal_graph(nodes, links)
            }
            "effect_dist" => {
                let res = self.method.as_trait().predict(x_view).unwrap_or_else(|_| self.method.as_trait().predict(x_view).unwrap());
                let preds = res.predictions.to_vec();

                // Calculate real histogram
                let min = preds.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = preds.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let n_bins = 10;
                let bin_width = if (max - min).abs() < f64::EPSILON {
                    1.0
                } else {
                    (max - min) / n_bins as f64
                };

                let mut bins = Vec::new();
                for i in 0..n_bins {
                    bins.push(min + i as f64 * bin_width);
                }

                let mut counts = vec![0u64; n_bins];
                for &p in &preds {
                    let mut b = if bin_width == 0.0 {
                        0
                    } else {
                        ((p - min) / bin_width) as usize
                    };
                    if b >= n_bins {
                        b = n_bins - 1;
                    }
                    counts[b] += 1;
                }

                VisualOutput::effect_dist(
                    "Individual Treatment Effect Distribution".to_string(),
                    "Frequency".to_string(),
                    bins,
                    counts,
                )
            }
            _ => VisualOutput::feature_importance(vec![], vec![]),
        }
    }
}

#[pymethods]
impl Model {
    #[pyo3(signature = (plot_type = "graph"))]
    fn to_visual_tag(&self, py: Python, plot_type: &str) -> String {
        let visual = self.get_visual(py, plot_type);
        format!("```json:causal-plot\n{}\n```", visual.to_json())
    }

    #[pyo3(signature = (plot_type = "graph"))]
    fn show(&self, py: Python, plot_type: &str) {
        println!("{}", self.to_visual_tag(py, plot_type));
    }

    #[pyo3(signature = (plot_type = "graph"))]
    fn preview(&self, py: Python, plot_type: &str) -> PyResult<()> {
        let visual = self.get_visual(py, plot_type);
        render_preview(py, &visual)
    }

    #[pyo3(signature = (plot_type = "graph"))]
    fn to_html(&self, py: Python, plot_type: &str) -> String {
        let visual = self.get_visual(py, plot_type);
        render_html_fragment(&visual)
    }

    fn estimate_effects(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<InferenceResult> {
        let core_res = self.method.as_trait().predict(x.as_array())?;

        Ok(InferenceResult {
            mean_effect: core_res.mean_effect,
            predictions: core_res.predictions.to_pyarray(py).to_owned(),
            confidence_intervals: core_res.confidence_intervals,
            feature_importance: core_res.feature_importance,
            feature_names: self.feature_names.clone(),
        })
    }

    #[pyo3(signature = (n_folds = 5, is_time_series = false))]
    fn validate(&self, py: Python, n_folds: usize, is_time_series: bool) -> PyResult<ValidationResult> {
        let _ = is_time_series; // Suppress unused warning while keeping the name
        let (x_view, t_view, y_view) = unsafe {
            (
                self.x.as_ref(py).as_array(),
                self.t.as_ref(py).as_array(),
                self.y.as_ref(py).as_array(),
            )
        };
        
        if let CausalMethod::Forest(ref forest) = self.method {
            let res = validate_causal_structure(forest, x_view, t_view, y_view, n_folds);
            Ok(ValidationResult {
                is_robust: res.is_robust,
                message: res.message,
            })
        } else {
            Ok(ValidationResult {
                is_robust: true,
                message: "Validation not implemented for this model type yet.".to_string(),
            })
        }
    }

    fn plot_importance(&self, py: Python) {
        println!("{}", self.to_visual_tag(py, "importance"));
    }

    fn plot_effects(&self, py: Python) {
        println!("{}", self.to_visual_tag(py, "effect_dist"));
    }
}

#[pyfunction]
#[pyo3(signature = (features, treatment, outcome, method = "forest", feature_names = None))]
fn create_model(
    py: Python,
    features: Py<PyArray2<f64>>,
    treatment: Py<PyArray1<f64>>,
    outcome: Py<PyArray1<f64>>,
    method: &str,
    feature_names: Option<Vec<String>>,
) -> PyResult<Model> {
    let mut causal_method = match method {
        "forest" => CausalMethod::Forest(CausalForest::new(10, 5, 5)),
        "linear" => CausalMethod::Linear(LinearCausalModel::new()),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown method: {}. Supported methods are 'forest', 'linear'",
                method
            )))
        }
    };

    unsafe {
        causal_method.as_trait_mut().fit(
            features.as_ref(py).as_array(),
            treatment.as_ref(py).as_array(),
            outcome.as_ref(py).as_array(),
        )?;
    }

    Ok(Model {
        method: causal_method,
        x: features,
        t: treatment,
        y: outcome,
        feature_names,
    })
}

#[pyfunction]
#[pyo3(signature = (model, plot = "graph"))]
fn plot_model(py: Python, model: Model, plot: &str) -> PyResult<PyObject> {
    let visual = model.get_visual(py, plot);
    let json_str = visual.to_json();
    let json_module = py.import("json")?;
    let dict = json_module.call_method1("loads", (json_str,))?;
    Ok(dict.to_object(py))
}

fn render_preview(py: Python, visual: &VisualOutput) -> PyResult<()> {
    let json_data = visual.to_json();
    let html_template = format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CausalFlow Preview: {}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{ font-family: sans-serif; background: #1a1a2e; color: #fff; margin: 0; padding: 20px; }}
        #chart {{ width: 100%; height: 600px; }}
        h1 {{ color: #4fc3f7; text-align: center; }}
    </style>
</head>
<body>
    <h1>CausalFlow: {}</h1>
    <div id="chart"></div>
    <script>
        const rawData = {};
        const chart = echarts.init(document.getElementById('chart'), 'dark');
        
        let option = {{}};
        if (rawData.visual_type === 'causal_graph') {{
            option = {{
                title: {{ text: 'Causal structure', left: 'center' }},
                tooltip: {{}},
                series: [{{
                    type: 'graph', layout: 'force',
                    symbolSize: 50, roam: true, label: {{ show: true }},
                    edgeSymbol: ['none', 'arrow'],
                    data: rawData.data.nodes.map(n => ({{
                        name: n.label,
                        itemStyle: {{ color: n.role === 'treatment' ? '#ff7043' : (n.role === 'outcome' ? '#66bb6a' : '#4fc3f7') }}
                    }})),
                    links: rawData.data.links,
                    force: {{ repulsion: 1000 }}
                }}]
            }};
        }} else if (rawData.visual_type === 'effect_dist') {{
            option = {{
                xAxis: {{ type: 'category', data: rawData.data.bins, name: rawData.data.x_label }},
                yAxis: {{ type: 'value', name: rawData.data.y_label }},
                series: [{{ data: rawData.data.counts, type: 'bar', itemStyle: {{ color: '#4fc3f7' }} }}]
            }};
        }} else if (rawData.visual_type === 'feature_importance') {{
            option = {{
                yAxis: {{ type: 'category', data: rawData.data.labels }},
                xAxis: {{ type: 'value' }},
                series: [{{ data: rawData.data.values, type: 'bar', itemStyle: {{ color: '#81c784' }} }}]
            }};
        }}
        chart.setOption(option);
    </script>
</body>
</html>
"#,
        visual.title, visual.title, json_data
    );

    let tempfile = py.import("tempfile")?;
    let res = tempfile.call_method1("mkstemp", (".html",))?;
    let path = res.get_item(1)?.extract::<String>()?;

    let builtins = py.import("builtins")?;
    let f = builtins.call_method1("open", (&path, "w"))?;
    f.call_method1("write", (html_template,))?;
    f.call_method0("close")?;

    let webbrowser = py.import("webbrowser")?;
    let os_path = py.import("os.path")?;
    let abs_path = os_path.call_method1("abspath", (&path,))?;
    let url = format!("file://{}", abs_path.extract::<String>()?);
    webbrowser.call_method1("open", (url,))?;

    println!("\n[Preview] Opening visualization in browser: {}", path);
    Ok(())
}

fn render_html_fragment(visual: &VisualOutput) -> String {
    let json_data = visual.to_json();
    let div_id = format!("causal-plot-{}", uuid_gen());
    format!(r#"
<div id="{}" style="width: 100%; height: 500px; min-height: 400px; background: #1a1a2e; border-radius: 8px; padding: 10px;"></div>
<script>
(function() {{
    const render = () => {{
        const chartDom = document.getElementById('{}');
        if (!chartDom) return;
        
        // Ensure ECharts is loaded
        if (typeof echarts === 'undefined') {{
            if (!window._echartsLoading) {{
                window._echartsLoading = true;
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js';
                script.onload = () => {{
                    window._echartsLoaded = true;
                    document.dispatchEvent(new Event('echarts-ready'));
                }};
                document.head.appendChild(script);
            }}
            document.addEventListener('echarts-ready', render);
            return;
        }}

        const rawData = {};
        const chart = echarts.init(chartDom, 'dark');
        
        let option = {{}};
        if (rawData.visual_type === 'causal_graph') {{
            option = {{
                title: {{ text: rawData.title, left: 'center', textStyle: {{ color: '#4fc3f7' }} }},
                tooltip: {{}},
                series: [{{
                    type: 'graph', layout: 'force',
                    symbolSize: 40, roam: true, label: {{ show: true, fontSize: 10 }},
                    edgeSymbol: ['none', 'arrow'],
                    data: rawData.data.nodes.map(n => ({{
                        name: n.label,
                        itemStyle: {{ color: n.role === 'treatment' ? '#ff7043' : (n.role === 'outcome' ? '#66bb6a' : '#4fc3f7') }}
                    }})),
                    links: rawData.data.links,
                    force: {{ repulsion: 300, edgeLength: 100 }}
                }}]
            }};
        }} else if (rawData.visual_type === 'effect_dist') {{
            option = {{
                title: {{ text: rawData.title, left: 'center', textStyle: {{ color: '#4fc3f7' }} }},
                xAxis: {{ type: 'category', data: rawData.data.bins, name: rawData.data.x_label }},
                yAxis: {{ type: 'value', name: rawData.data.y_label }},
                series: [{{ data: rawData.data.counts, type: 'bar', itemStyle: {{ color: '#4fc3f7' }} }}]
            }};
        }} else if (rawData.visual_type === 'feature_importance') {{
            option = {{
                title: {{ text: rawData.title, left: 'center', textStyle: {{ color: '#4fc3f7' }} }},
                yAxis: {{ type: 'category', data: rawData.data.labels }},
                xAxis: {{ type: 'value' }},
                series: [{{ data: rawData.data.values, type: 'bar', itemStyle: {{ color: '#81c784' }} }}]
            }};
        }}
        chart.setOption(option);
        
        // Handle Resize
        window.addEventListener('resize', () => chart.resize());
    }};
    render();
}})();
</script>
"#, div_id, div_id, json_data)
}

fn uuid_gen() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;
    let mut s = DefaultHasher::new();
    SystemTime::now().hash(&mut s);
    format!("{:x}", s.finish())
}

#[pymodule]
fn _causalflow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_flow, m)?)?;
    m.add_function(wrap_pyfunction!(create_model, m)?)?;
    m.add_function(wrap_pyfunction!(plot_model, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<InferenceResult>()?;
    m.add_class::<ValidationResult>()?;
    Ok(())
}
