# CausalFlow

High-performance Causal Inference engine powered by Rust and LLM-friendly Python interfaces.

## Overview

CausalFlow provides a state-of-the-art **Causal Forest (Generalized Random Forest)** implementation designed for speed, accuracy, and interpretability. By leveraging Rust for core computations and PyO3 for seamless Python integration, CausalFlow enables researchers and data scientists to estimate Heterogeneous Treatment Effects (ITE) on large-scale datasets with minimal overhead.

## Key Features

- 🦀 **High-Performance Core**: Implemented in Rust with **zero-copy data access** and **Rayon-powered multi-threading**.
- 🌲 **Generalized Random Forest (GRF)**: Robust splitting criteria maximizing treatment effect variance.
- ⚖️ **Honesty Property**: Built-in data subsampling to ensure unbiased effect estimation and prevent overfitting.
- 🤖 **LLM-Friendly**: 
    - **Structured Summaries**: Automatic natural language interpretation of results.
    - **ASCII Tables**: Clean terminal outputs for easy digestion by AI agents.
    - **Type Hints**: Full `.pyi` support for IDEs and LLM tool calling.
- 🛡️ **Validation Guardrails**: Built-in methods for causal robustness checks (cross-validation, time-series support).

## Installation

### Prerequisites

- Rust toolchain (cargo, rustc)
- Python 3.8+
- `maturin` (for building from source)

### From Source

```bash
git clone https://github.com/taka-data/CausalFlow.git
cd CausalFlow/py-causalflow
maturin develop
```

## Quick Start

```python
import numpy as np
import causalflow as cf
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X, Y = data.data, data.target
T = (X[:, 0] > X[:, 0].mean()).astype(float) # Dummy treatment

# 1. Create Model
model = cf.create_model(
    features=X, 
    treatment=T, 
    outcome=Y, 
    feature_names=list(data.feature_names)
)

# 2. Estimate Effects
results = model.estimate_effects(X)
print(results)  # Displays a structured summary table & interpretation

# 3. Validate Robustness
validation = model.validate(n_folds=5)
print(validation.message)
```

## Project Structure

- `causalflow-core`: Core Rust library for causal algorithms.
- `py-causalflow`: Python bindings using PyO3.
- `causalflow-macros`: Procedural macros for metadata generation.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our development workflow.

## License

MIT License.
