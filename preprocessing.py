"""
Stage 2: Data Preprocessing
Handles feature scaling, encoding, and train-test split.    
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.feature_names = []
        
    def preprocess(self, df, target_col="target"):
        """Full preprocessing pipeline."""
        logger.info("Starting preprocessing...")
        
        # Drop non-numeric/non-feature columns
        drop_cols = [c for c in df.columns if c in ["target_name"]]
        df = df.drop(columns=drop_cols, errors="ignore")
        
        X = df.drop(columns=[target_col])
        Y = df[target_col]
        
        self.feature_names = list(X.columns)
        
        #Imputed missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=self.test_size, random_state=self.random_state, stratify=Y
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Save preprocessors
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, "models/scaler.pkl")
        joblib.dump(self.imputer, "models/imputer.pkl")
        joblib.dump(self.feature_names, "models/feature_names.pkl")
        
        stats = {
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "num_features": int(X_train.shape[1]),
            "feature_names": self.feature_names,
            "test_ratio": self.test_size,
            "scaling": "StandardScaler",
            "imputation": "Mean Imputation"
        }
        
        return X_train, X_test, Y_train, Y_test, stats
    
    def run(self, df):
        return self.preprocess(df)    