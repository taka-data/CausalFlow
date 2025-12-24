import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import causalflow as cf

def main():
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    print("Preparing data for causal analysis...")
    # Let's say we want to see the effect of 'MedInc' (Median Income) on 'target' (House Value)
    # This is a bit simplistic for causal inference but works for demonstration.
    X = df.drop(columns=['target']).values
    T = df['MedInc'].values
    Y = df['target'].values

    print(f"Features shape: {X.shape}")
    print(f"Treatment shape: {T.shape}")
    print(f"Outcome shape: {Y.shape}")

    # 1. Create a causal model (Causal Forest)
    print("\n[Step 1] Creating Causal Model (Rust-backed)...")
    # Using named arguments
    model = cf.create_model(features=X, treatment=T, outcome=Y)
    
    # 2. Estimate effects
    print("[Step 2] Estimating Heterogeneous Treatment Effects (ITE)...")
    results = model.estimate_effects(X)

    print(f"\nEstimated Effects (first 5 samples):")
    print(results.predictions[:5])

    print(f"Average Estimated Effect (ATE): {results.mean_effect:.4f}")
    print(f"Confidence Interval (first sample): {results.confidence_intervals[0]}")

    # 3. Validation (Guardrail)
    print("\n[Step 3] Validating Causal Robustness...")
    # New: specifying n_folds and is_time_series
    validation = model.validate(n_folds=10, is_time_series=False)
    if validation.is_robust:
        print(f"Success: {validation.message}")
    else:
        print(f"Warning: {validation.message}")
    
    print("\nCausalFlow Analysis Complete.")

if __name__ == "__main__":
    main()
