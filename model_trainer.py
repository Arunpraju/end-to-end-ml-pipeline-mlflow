"""
stages 3: Model Training with MLflow Tracking
Trains multiple models and logs everything to MLflow.
"""
import mlflow 
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib
import os 
import json
import logging

logger = logging.getLogger(__name__)

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, probability=True, random_state=42
    ),
}

class ModelTrainer:
    def __init__(self, experiment_name="Wine_Quality_Pipeline"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)
        
    def train_and_log(self, model_name, model, X_train, X_test, y_train, y_test, extra_params=None):
        """Train a single model and log to MLflow."""
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            logger.info(f"Training {model_name} (run_id={run_id[:8]}...)")
            
            #Log parameters
            params = model.get_params()
            if extra_params:
                params.update(extra_params)
            mlflow.log_params({k: str(v) for k, v in list(params.items())[:15]})
            
            #Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True)
            
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            model_path = f"models/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_path)
            
            result = {
                "run_id": run_id,
                "model_name": model_name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm,
                "classification_report": report,
                "model_path": model_path
            }
            
            logger.info(f"{model_name} → Acc: {acc:.4f} | F1: {f1:.4f}")
            return result
    def train_all(self, X_train, X_test, y_train, y_test, selected_models=None):
        """Train all or selected models."""
        results = []
        models_to_train = selected_models if selected_models else list(MODELS.keys())
        
        for name in models_to_train:
            if name in MODELS:
                result = self.train_and_log(
                    name, MODELS[name], X_train, X_test, y_train, y_test
                )
                results.append(result)
                
        # Save results summary
        os.makedirs("models", exist_ok=True)
        with open("models/results.json", "w") as f:
            json.dump(results, f, indent=2)
                
        # Find best model
        best = max(results, key=lambda x: x["f1_score"])
        logger.info(f"\n Best Model: {best['model_name']} with F1={best['f1_score']}")
            
        # Save best model reference
        with open("models/best_model.json", "w") as f:
            json.dump({"model_name":  best["model_name"], "run_id": best["run_id"]}, f)
                
        return results, best
        
    def get_available_models(self):
        return list(MODELS.keys())