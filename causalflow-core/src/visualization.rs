use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct VisualOutput {
    pub visual_type: String,
    pub data: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureImportanceData {
    pub labels: Vec<String>,
    pub values: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CausalGraphData {
    pub nodes: Vec<NodeInfo>,
    pub edges: Vec<EdgeInfo>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NodeInfo {
    pub id: String,
    pub role: String, // "treatment", "outcome", "confounder", "feature"
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EdgeInfo {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

impl VisualOutput {
    pub fn feature_importance(labels: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            visual_type: "feature_importance".to_string(),
            data: serde_json::to_value(FeatureImportanceData { labels, values }).unwrap(),
        }
    }

    pub fn causal_graph(nodes: Vec<NodeInfo>, edges: Vec<EdgeInfo>) -> Self {
        Self {
            visual_type: "causal_graph".to_string(),
            data: serde_json::to_value(CausalGraphData { nodes, edges }).unwrap(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }
}
