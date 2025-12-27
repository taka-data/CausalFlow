from typing import List, Optional, Tuple, Any
import numpy as np

class InferenceResult:
    mean_effect: float
    predictions: np.ndarray
    confidence_intervals: List[Tuple[float, float]]
    feature_importance: List[float]
    feature_names: Optional[List[str]]
    def to_visual_tag(self, plot_type: str = "importance") -> str: ...
    def show(self, plot_type: str = "importance") -> None: ...

class Model:
    def estimate_effects(self, x: Any) -> InferenceResult: ...
    def validate(self, n_folds: int = 5, is_time_series: bool = False) -> Any: ...

def create_model(
    features: Any,
    treatment: Any,
    outcome: Any,
    method: str = "forest",
    feature_names: Optional[List[str]] = None
) -> Model: ...
