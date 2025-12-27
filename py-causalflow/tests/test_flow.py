import pandas as pd
import numpy as np
import pytest
import causalflow

def test_preprocessing_categorical():
    # Test that categorical data is automatically handled
    df = pd.DataFrame({
        'feature1': [10, 20, 30],
        'category1': ['A', 'B', 'A']
    })
    t = [0, 1, 0]
    y = [100, 200, 100]
    
    model = causalflow.create_model(df, t, y, method='linear')
    
    # Check if features are one-hot encoded
    assert 'category1_A' in model.feature_names_out_
    assert 'category1_B' in model.feature_names_out_
    assert len(model.feature_names_out_) == 3 # feature1, category1_A, category1_B

def test_preprocessing_nans():
    # Test that NaNs are automatically imputed
    df = pd.DataFrame({
        'feature1': [1, np.nan, 3],
        'feature2': [10, 20, 30]
    })
    t = [0, 1, 0]
    y = [5, 15, 5]
    
    model = causalflow.create_model(df, t, y, method='linear')
    
    # Should not raise error during creation
    assert model is not None
    
    # Check prediction with NaN
    new_df = pd.DataFrame({'feature1': [np.nan], 'feature2': [25]})
    res = model.estimate_effects(new_df)
    assert not np.isnan(res.mean_effect)

def test_algorithm_switching():
    # Verify that different methods work
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    t = np.array([0, 1], dtype=np.float64)
    y = np.array([1, 10], dtype=np.float64)
    
    # Forest
    m_forest = causalflow.create_model(x, t, y, method='forest')
    res_forest = m_forest.estimate_effects(x)
    assert res_forest is not None
    
    # Linear
    m_linear = causalflow.create_model(x, t, y, method='linear')
    res_linear = m_linear.estimate_effects(x)
    assert res_linear is not None
    assert abs(res_linear.mean_effect - 9.0) < 1e-5

def test_invalid_method():
    # Verify that unknown method raises ValueError
    x = np.array([[1, 2]])
    t = [0]
    y = [1]
    with pytest.raises(ValueError, match="Unknown method"):
        causalflow.create_model(x, t, y, method='unknown_algo')
