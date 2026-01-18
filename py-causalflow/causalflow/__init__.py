from . import _causalflow
from ._causalflow import *
from .preprocessing import DataProcessor
import pandas as pd
import numpy as np

class CausalModelWrapper:
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor
        self.feature_names_out_ = processor.feature_names_out_

    def estimate_effects(self, x):
        x_proc = self._processor.transform(x)
        return self._model.estimate_effects(x_proc)
    
    def validate(self, n_folds=5, is_time_series=False):
        return self._model.validate(n_folds, is_time_series)
    
    def show(self, plot_type="graph"):
        return self._model.show(plot_type)
    
    def preview(self, plot_type="graph"):
        return self._model.preview(plot_type)
    
    def to_visual_tag(self, plot_type="graph"):
        return self._model.to_visual_tag(plot_type)

    def __getattr__(self, name):
        # Fallback to the internal Rust model
        return getattr(self._model, name)

def create_model(features, treatment, outcome, method="forest", feature_names=None, use_mice=True):
    """
    High-level factory function with automated preprocessing and unified API.
    """
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    
    if isinstance(treatment, (pd.Series, pd.DataFrame)):
        treatment_df = pd.DataFrame(treatment)
    else:
        treatment_df = pd.DataFrame(treatment, columns=["treatment"])
        
    if isinstance(outcome, (pd.Series, pd.DataFrame)):
        outcome_df = pd.DataFrame(outcome)
    else:
        outcome_df = pd.DataFrame(outcome, columns=["outcome"])

    # Align indices and check for NaNs in treatment and outcome
    # We join them to ensure indices are aligned before dropping
    combined = pd.concat([features, treatment_df, outcome_df], axis=1)
    
    # Identify indices where treatment or outcome are NaN
    treatment_cols = [f"t_{i}" if i < treatment_df.shape[1] else col for i, col in enumerate(treatment_df.columns)]
    outcome_cols = [f"y_{i}" if i < outcome_df.shape[1] else col for i, col in enumerate(outcome_df.columns)]
    
    # Simpler way to get the mask: use the original dataframes to check for NaNs
    valid_mask = treatment_df.notna().all(axis=1) & outcome_df.notna().all(axis=1)
    
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"Warning: Dropping {n_dropped} rows due to missing values in treatment or outcome.")
        features = features[valid_mask]
        treatment_df = treatment_df[valid_mask]
        outcome_df = outcome_df[valid_mask]

    processor = DataProcessor(use_mice=use_mice)
    
    # 1. Preprocess features
    x_processed = processor.fit_transform(features)
    
    # 2. Preprocess treatment
    t_numeric = treatment_df.values.flatten()
        
    # 3. Preprocess outcome
    y_numeric = outcome_df.values.flatten()

    # Create the internal Rust model
    rust_model = _causalflow.create_model(
        x_processed, 
        t_numeric.astype(np.float64), 
        y_numeric.astype(np.float64), 
        method, 
        processor.feature_names_out_
    )
    
    return CausalModelWrapper(rust_model, processor)

__all__ = ["create_model", "DataProcessor", "CausalModelWrapper"]
