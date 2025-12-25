use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone)]
pub struct CausalForest {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_leaf_size: usize,
    pub trees: Vec<CausalTree>,
    pub n_features: usize,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct InferenceResult {
    pub predictions: Array1<f64>,
    pub mean_effect: f64,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub feature_importance: Vec<f64>,
}

#[derive(Clone)]
pub struct CausalTree {
    pub root: Option<Box<Node>>,
    pub feature_importance: Vec<f64>,
}

#[derive(Clone)]
pub enum Node {
    Leaf {
        treatment_effect: f64,
        size: usize,
    },
    Internal {
        feature_idx: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl CausalForest {
    pub fn new(n_estimators: usize, max_depth: usize, min_leaf_size: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
            min_leaf_size,
            trees: Vec::new(),
            n_features: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, t: &Array1<f64>, y: &Array1<f64>) {
        let n_features = x.ncols();
        self.n_features = n_features;
        self.trees = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| {
                let mut tree = CausalTree::new(n_features);
                tree.fit(
                    x.view(),
                    t.view(),
                    y.view(),
                    self.max_depth,
                    self.min_leaf_size,
                );
                tree
            })
            .collect();
    }

    pub fn predict(&self, x: &Array2<f64>) -> InferenceResult {
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for tree in &self.trees {
            predictions += &tree.predict(x);
        }

        if !self.trees.is_empty() {
            predictions /= self.trees.len() as f64;
        }

        let mean_effect = predictions.mean().unwrap_or(0.0);
        let confidence_intervals = predictions.iter().map(|&p| (p - 0.1, p + 0.1)).collect();

        // Aggregate feature importance
        let mut feature_importance = vec![0.0; self.n_features];
        if !self.trees.is_empty() {
            for tree in &self.trees {
                for (i, &imp) in tree.feature_importance.iter().enumerate() {
                    feature_importance[i] += imp;
                }
            }
            let sum: f64 = feature_importance.iter().sum();
            if sum > 0.0 {
                for imp in feature_importance.iter_mut() {
                    *imp /= sum;
                }
            }
        }

        InferenceResult {
            predictions,
            mean_effect,
            confidence_intervals,
            feature_importance,
        }
    }
}

impl CausalTree {
    pub fn new(n_features: usize) -> Self {
        Self {
            root: None,
            feature_importance: vec![0.0; n_features],
        }
    }

    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        t: ArrayView1<f64>,
        y: ArrayView1<f64>,
        max_depth: usize,
        min_leaf_size: usize,
    ) {
        let n_samples = x.nrows();
        let mut rng = thread_rng();

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let split_size = n_samples / 2;
        let split_indices = &indices[..split_size];
        let estimation_indices = &indices[split_size..];

        self.root = Some(self.build_tree(
            x,
            t,
            y,
            split_indices,
            estimation_indices,
            0,
            max_depth,
            min_leaf_size,
        ));
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        &mut self,
        x: ArrayView2<f64>,
        t: ArrayView1<f64>,
        y: ArrayView1<f64>,
        split_idx: &[usize],
        est_idx: &[usize],
        depth: usize,
        max_depth: usize,
        min_leaf_size: usize,
    ) -> Box<Node> {
        if depth >= max_depth
            || split_idx.len() < min_leaf_size * 2
            || est_idx.len() < min_leaf_size
        {
            return Box::new(Node::Leaf {
                treatment_effect: self.estimate_effect(t, y, est_idx),
                size: est_idx.len(),
            });
        }

        let n_features = x.ncols();
        let mut rng = thread_rng();

        let n_sub_features = (n_features as f64).sqrt() as usize;
        let mut sampled_features: Vec<usize> = (0..n_features).collect();
        sampled_features.shuffle(&mut rng);
        let sampled_features = &sampled_features[..n_sub_features];

        let best_split = sampled_features
            .par_iter()
            .map(|&f_idx| {
                let mut local_rng = thread_rng();
                let mut local_best_gain = -1.0;
                let mut local_best_split = None;

                let values: Vec<f64> = split_idx.iter().map(|&i| x[[i, f_idx]]).collect();
                for _ in 0..10 {
                    let threshold = values[local_rng.gen_range(0..values.len())];

                    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                        split_idx.iter().partition(|&&i| x[[i, f_idx]] <= threshold);

                    if left_idx.len() < min_leaf_size || right_idx.len() < min_leaf_size {
                        continue;
                    }

                    let gain = self.calculate_causal_gain(t, y, &left_idx, &right_idx);
                    if gain > local_best_gain {
                        local_best_gain = gain;
                        local_best_split = Some((f_idx, threshold, left_idx, right_idx));
                    }
                }
                (local_best_gain, local_best_split)
            })
            .reduce(|| (-1.0, None), |a, b| if a.0 > b.0 { a } else { b });

        if let Some((gain, Some((f_idx, threshold, left_split, right_split)))) = Some(best_split) {
            // Track importance
            self.feature_importance[f_idx] += gain;

            let (left_est, right_est): (Vec<usize>, Vec<usize>) =
                est_idx.iter().partition(|&&i| x[[i, f_idx]] <= threshold);

            Box::new(Node::Internal {
                feature_idx: f_idx,
                threshold,
                left: self.build_tree(
                    x,
                    t,
                    y,
                    &left_split,
                    &left_est,
                    depth + 1,
                    max_depth,
                    min_leaf_size,
                ),
                right: self.build_tree(
                    x,
                    t,
                    y,
                    &right_split,
                    &right_est,
                    depth + 1,
                    max_depth,
                    min_leaf_size,
                ),
            })
        } else {
            Box::new(Node::Leaf {
                treatment_effect: self.estimate_effect(t, y, est_idx),
                size: est_idx.len(),
            })
        }
    }

    fn calculate_causal_gain(
        &self,
        t: ArrayView1<f64>,
        y: ArrayView1<f64>,
        left: &[usize],
        right: &[usize],
    ) -> f64 {
        let tau_l = self.estimate_effect(t, y, left);
        let tau_r = self.estimate_effect(t, y, right);
        let nl = left.len() as f64;
        let nr = right.len() as f64;
        let n = nl + nr;

        (nl * nr / (n * n)) * (tau_l - tau_r).powi(2)
    }

    fn estimate_effect(&self, t: ArrayView1<f64>, y: ArrayView1<f64>, indices: &[usize]) -> f64 {
        let mut y1_sum = 0.0;
        let mut y1_count = 0;
        let mut y0_sum = 0.0;
        let mut y0_count = 0;

        for &i in indices {
            if t[i] > 0.5 {
                y1_sum += y[i];
                y1_count += 1;
            } else {
                y0_sum += y[i];
                y0_count += 1;
            }
        }

        if y1_count > 0 && y0_count > 0 {
            (y1_sum / y1_count as f64) - (y0_sum / y0_count as f64)
        } else {
            0.0
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut preds = Array1::zeros(n_samples);
        if let Some(ref root) = self.root {
            for i in 0..n_samples {
                let row = x.row(i);
                if let Some(slice) = row.as_slice() {
                    preds[i] = root.predict(slice);
                } else {
                    let row_vec: Vec<f64> = row.iter().cloned().collect();
                    preds[i] = root.predict(&row_vec);
                }
            }
        }
        preds
    }
}

impl Node {
    pub fn predict(&self, x: &[f64]) -> f64 {
        match self {
            Node::Leaf {
                treatment_effect, ..
            } => *treatment_effect,
            Node::Internal {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if x[*feature_idx] <= *threshold {
                    left.predict(x)
                } else {
                    right.predict(x)
                }
            }
        }
    }
}
