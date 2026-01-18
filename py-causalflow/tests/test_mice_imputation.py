import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path so we can import causalflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import causalflow

def test_mice_imputation():
    print("Testing MICE imputation...")
    # 1. Create a dataset with missing values in features
    df = pd.DataFrame({
        'x1': [1, 2, np.nan, 4, 5],
        'x2': [5, np.nan, 3, 2, 1],
        'treatment': [0, 1, 0, 1, 0],
        'outcome': [10, 12, 11, 13, 14]
    })
    
    features = df[['x1', 'x2']]
    treatment = df['treatment']
    outcome = df['outcome']
    
    # create_model will preprocess and call the Rust implementation
    # We want to check if the preprocessing works without errors
    model = causalflow.create_model(features, treatment, outcome, method="forest", use_mice=True)
    
    print("MICE imputation test: SUCCESS")

def test_treatment_outcome_nan():
    print("Testing treatment/outcome NaN handling...")
    df = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'treatment': [0, 1, np.nan, 1, 0],
        'outcome': [10, 12, 11, np.nan, 14]
    })
    
    features = df[['x1']]
    treatment = df['treatment']
    outcome = df['outcome']
    
    # Should drop 2 rows (index 2 and 3)
    model = causalflow.create_model(features, treatment, outcome, method="forest", use_mice=True)
    
    # We can check if the internal processor has the correct number of samples if we had access,
    # but the print statement "Warning: Dropping 2 rows" should appear.
    print("Treatment/Outcome NaN handling test: SUCCESS (check warning output)")

def test_mice_toggle():
    print("Testing MICE toggle (use_mice=False)...")
    df = pd.DataFrame({
        'x1': [1, 2, np.nan, 4, 5],
        'treatment': [0, 1, 0, 1, 0],
        'outcome': [10, 12, 11, 13, 14]
    })
    
    features = df[['x1']]
    treatment = df['treatment']
    outcome = df['outcome']
    
    model = causalflow.create_model(features, treatment, outcome, method="forest", use_mice=False)
    print("MICE toggle test: SUCCESS")

if __name__ == "__main__":
    try:
        test_mice_imputation()
        test_treatment_outcome_nan()
        test_mice_toggle()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)
