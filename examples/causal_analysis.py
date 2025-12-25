import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import causalflow as cf
import json

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
    # Using named arguments and passing feature names
    model = cf.create_model(
        features=X, 
        treatment=T, 
        outcome=Y, 
        feature_names=list(data.feature_names)
    )
    
    # 2. Estimate effects
    print("[Step 2] Estimating Heterogeneous Treatment Effects (ITE)...")
    results = model.estimate_effects(X)

    print("\n[Causal Inference Summary]")
    print(results)  # This calls the new __repr__

    # 3. Validation (Guardrail)
    print("\n[Step 3] Validating Causal Robustness...")
    validation = model.validate(n_folds=10, is_time_series=False)
    if validation.is_robust:
        print(f"Success: {validation.message}")
    else:
        print(f"Warning: {validation.message}")

    # 4. Visualization (Headless / LLM-Friendly)
    print("\n[Step 4] Generating Visualization Data (Headless UI Concept)...")
    
    # Pattern A: Causal Graph
    print("\n[Causal Graph Tag]")
    model.show(plot_type='graph')
    
    # Pattern B: Effect Distribution
    print("\n[Effect Distribution Tag]")
    model.show(plot_type='effect_dist')

    # Feature Importance (from InferenceResult)
    print("\n[Feature Importance Tag]")
    results.show()

    # --- Local Preview (Optional, for CLI users) ---
    # Uncomment the lines below to open graphs in your browser locally.
    print("\n[Local Preview] Opening graphs in your browser...")
    model.preview(plot_type='graph')
    results.preview()
    
    print("\nCausalFlow Analysis Complete.")

if __name__ == "__main__":
    main()
