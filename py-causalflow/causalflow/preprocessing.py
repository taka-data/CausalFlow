import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

class DataProcessor:
    def __init__(self, use_mice=True):
        self.use_mice = use_mice
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.categorical_columns_ = []
        self.imputers_ = {}
        self.mice_imputer_ = None
        self.one_hot_encodings_ = {}

    def fit_transform(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        self.feature_names_in_ = df.columns.tolist()
        processed_df = df.copy()

        # 1. Handle Categorical Columns (Simple One-Hot)
        self.categorical_columns_ = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 2. Imputation
        if self.use_mice:
            # First, simple imputation for categorical columns as MICE needs numeric input
            for col in self.categorical_columns_:
                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "unknown"
                processed_df[col] = processed_df[col].fillna(mode_val)
                self.imputers_[col] = mode_val
            
            # One-Hot Encoding BEFORE MICE to ensure all inputs are numeric
            if self.categorical_columns_:
                processed_df = pd.get_dummies(processed_df, columns=self.categorical_columns_)
            
            # Now apply MICE to all columns (which are now numeric)
            self.mice_imputer_ = IterativeImputer(random_state=42)
            processed_df_values = self.mice_imputer_.fit_transform(processed_df)
            processed_df = pd.DataFrame(processed_df_values, columns=processed_df.columns)
        else:
            # Fallback to Simple Imputation (as before)
            for col in processed_df.columns:
                if processed_df[col].dtype.kind in 'iuf':
                    mean_val = processed_df[col].mean()
                    processed_df[col] = processed_df[col].fillna(mean_val)
                    self.imputers_[col] = mean_val
                else:
                    mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "unknown"
                    processed_df[col] = processed_df[col].fillna(mode_val)
                    self.imputers_[col] = mode_val

            # One-Hot Encoding AFTER simple imputation
            if self.categorical_columns_:
                processed_df = pd.get_dummies(processed_df, columns=self.categorical_columns_)
        
        self.feature_names_out_ = [str(c) for c in processed_df.columns.tolist()]
        return processed_df.values.astype(np.float64)

    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=self.feature_names_in_)
        
        processed_df = df.copy()
        
        if self.use_mice:
            # Apply simple imputation for categorical columns first (as in fit)
            for col in self.categorical_columns_:
                val = self.imputers_.get(col, "unknown")
                processed_df[col] = processed_df[col].fillna(val)
            
            # One-Hot Encoding BEFORE MICE
            if self.categorical_columns_:
                processed_df = pd.get_dummies(processed_df, columns=self.categorical_columns_)
                
                # Align columns (important for dummy variables)
                missing_cols = set(self.mice_imputer_.feature_names_in_) - set(processed_df.columns)
                for col in missing_cols:
                    processed_df[col] = 0
                processed_df = processed_df[list(self.mice_imputer_.feature_names_in_)]

            # Apply MICE
            processed_df_values = self.mice_imputer_.transform(processed_df)
            processed_df = pd.DataFrame(processed_df_values, columns=processed_df.columns)
        else:
            # Apply simple imputation
            for col, val in self.imputers_.items():
                processed_df[col] = processed_df[col].fillna(val)

            # One-Hot Encoding
            if self.categorical_columns_:
                processed_df = pd.get_dummies(processed_df, columns=self.categorical_columns_)
                
                # Align columns
                for col in self.feature_names_out_:
                    if col not in processed_df.columns:
                        processed_df[col] = 0
                processed_df = processed_df[self.feature_names_out_]

        return processed_df.values.astype(np.float64)
