import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional

class InferenceResult:
    mean_effect: float
    predictions: npt.NDArray[np.float64]
    confidence_intervals: List[Tuple[float, float]]
    feature_importance: List[float]
    def summary(self) -> str: ...

class ValidationResult:
    is_robust: bool
    message: str

class Model:
    def estimate_effects(self, x: npt.NDArray[np.float64]) -> InferenceResult: ...
    def validate(self, n_folds: int = 5, is_time_series: bool = False) -> ValidationResult: ...
    def plot_importance(self) -> None: ...
    def plot_effects(self) -> None: ...

def create_model(
    features: npt.NDArray[np.float64],
    treatment: npt.NDArray[np.float64],
    outcome: npt.NDArray[np.float64],
    feature_names: Optional[List[str]] = None
) -> Model: ...

def analyze_flow() -> str: ...
