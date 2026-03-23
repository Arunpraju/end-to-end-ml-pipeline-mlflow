"""
Stage 1: Data Ingestion
Loads and validates the dataset (Wine Quality).
- If a CSV exists at data_path, loads it directly.
- Otherwise generates the Wine dataset from sklearn and saves it as CSV.
"""

import os
import logging

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, data_path=r"C:\Users\USER\Desktop\End 2 End ML\data\wine_quality.csv"):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file, or generate from sklearn if absent."""
        if os.path.exists(self.data_path):
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            logger.info(f"'{self.data_path}' not found — generating Wine dataset from sklearn...")
            wine = load_wine()
            df = pd.DataFrame(wine.data, columns=wine.feature_names)
            df["target"] = wine.target
            df["target_name"] = [wine.target_names[i] for i in wine.target]
            data_dir = os.path.dirname(self.data_path)
            if data_dir:
                os.makedirs(data_dir, exist_ok=True)
            df.to_csv(self.data_path, index=False)
            logger.info(f"Dataset saved to {self.data_path}")
        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """Run basic data-quality checks and return a summary report."""
        report = {
            "total_rows":        int(len(df)),
            "total_columns":     int(len(df.columns)),
            "missing_values":    int(df.isnull().sum().sum()),
            "duplicate_rows":    int(df.duplicated().sum()),
            "columns":           list(df.columns),
            "dtypes":            {col: str(dtype) for col, dtype in df.dtypes.items()},
            "class_distribution": (
                df["target"].value_counts().to_dict()
                if "target" in df.columns else {}
            ),
        }
        logger.info(
            f"Validation complete — {report['total_rows']} rows, "
            f"{report['total_columns']} columns, "
            f"{report['missing_values']} missing, "
            f"{report['duplicate_rows']} duplicates"
        )
        return report

   
    def run(self):
        """Load the data, validate it, and return (df, data_report)."""
        df = self.load_data()
        data_report = self.validate_data(df)
        return df, data_report