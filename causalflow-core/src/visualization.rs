use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct VisualOutput {
    pub visual_type: String,
    pub title: String,
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
    pub links: Vec<LinkInfo>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NodeInfo {
    pub id: String,
    pub label: String,
    pub role: String, // "treatment", "outcome", "confounder", "feature"
    pub value: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LinkInfo {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EffectDistData {
    pub x_label: String,
    pub y_label: String,
    pub bins: Vec<f64>,
    pub counts: Vec<u64>,
}

impl VisualOutput {
    pub fn feature_importance(labels: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            visual_type: "feature_importance".to_string(),
            title: "Feature Importance Analysis".to_string(),
            data: serde_json::to_value(FeatureImportanceData { labels, values }).unwrap(),
        }
    }

    pub fn causal_graph(nodes: Vec<NodeInfo>, links: Vec<LinkInfo>) -> Self {
        Self {
            visual_type: "causal_graph".to_string(),
            title: "Causal Structure Graph".to_string(),
            data: serde_json::to_value(CausalGraphData { nodes, links }).unwrap(),
        }
    }

    pub fn effect_dist(x_label: String, y_label: String, bins: Vec<f64>, counts: Vec<u64>) -> Self {
        Self {
            visual_type: "effect_dist".to_string(),
            title: "Treatment Effect Distribution".to_string(),
            data: serde_json::to_value(EffectDistData { x_label, y_label, bins, counts }).unwrap(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }
}
