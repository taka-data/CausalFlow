import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.categorical_columns_ = []
        self.imputers_ = {}
        self.one_hot_encodings_ = {}

    def fit_transform(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        self.feature_names_in_ = df.columns.tolist()
        processed_df = df.copy()

        # 1. Handle Categorical Columns (Simple One-Hot)
        self.categorical_columns_ = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 2. Imputation (Mean for numeric, Mode for categorical)
        for col in processed_df.columns:
            if processed_df[col].dtype.kind in 'iuf':
                mean_val = processed_df[col].mean()
                processed_df[col] = processed_df[col].fillna(mean_val)
                self.imputers_[col] = mean_val
            else:
                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "unknown"
                processed_df[col] = processed_df[col].fillna(mode_val)
                self.imputers_[col] = mode_val

        # 3. One-Hot Encoding
        if self.categorical_columns_:
            processed_df = pd.get_dummies(processed_df, columns=self.categorical_columns_)
        
        self.feature_names_out_ = processed_df.columns.tolist()
        return processed_df.values.astype(np.float64)

    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=self.feature_names_in_)
        
        processed_df = df.copy()
        
        # Apply imputation
        for col, val in self.imputers_.items():
            processed_df[col] = processed_df[col].fillna(val)

        # One-Hot Encoding (must align with fit columns)
        if self.categorical_columns_:
            processed_df = pd.get_dummies(processed_df, columns=self.categorical_columns_)
            
            # Align columns
            for col in self.feature_names_out_:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            processed_df = processed_df[self.feature_names_out_]

        return processed_df.values.astype(np.float64)
