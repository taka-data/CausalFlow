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

def create_model(features, treatment, outcome, method="forest", feature_names=None):
    """
    High-level factory function with automated preprocessing and unified API.
    """
    processor = DataProcessor()
    
    # 1. Preprocess features
    x_processed = processor.fit_transform(features)
    
    # 2. Preprocess treatment
    if isinstance(treatment, (pd.Series, pd.DataFrame)):
        t_numeric = treatment.values.flatten()
    else:
        t_numeric = np.array(treatment).flatten()
        
    # 3. Preprocess outcome
    if isinstance(outcome, (pd.Series, pd.DataFrame)):
        y_numeric = outcome.values.flatten()
    else:
        y_numeric = np.array(outcome).flatten()

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
