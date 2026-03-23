"""
app.py - Main Flask Application
End-to-End ML Pipeline with ML flow Tracking
"""
 
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import os
import joblib
import numpy as np
import pandas as pd
import threading
import time
import logging

from pipeline.data_ingestion import DataIngestion
from pipeline.preprocessing import DataPreprocessor
from pipeline.model_trainer import ModelTrainer, MODELS
from pipeline.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")   
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "mlpipeline-secret-2024"

DATA_PATH = r"C:\Users\USER\Desktop\End 2 End ML\data\wine_quality.csv"

pipeline_state = {
    "status" : "idle",
    "stage" : "",
    "progress" : 0,
    "logs" : [],
    "results" : [],
    "best_model" : None,
    "data_report" : None,
    "preprocess_stats" : None,
    "charts" : None,
    "error" : ""
}

def add_log(message, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    pipeline_state["logs"].append({"time": timestamp, "level": level, "msg": message})
    logger.info(message)
    
def run_pipeline(selected_models):
    """Background thread: runs the full ML pipeline."""
    global pipeline_state
    pipeline_state.update({"status": "running", "logs": [], "progress": 0, "error": ""})
    
    try:
        # Stage 1: Data Ingestion
        pipeline_state["stage"] = "Data Ingestion"
        pipeline_state["progress"] = 10
        add_log(" Stage 1: Data ingestion started...")
        ingestion = DataIngestion(data_path=DATA_PATH)
        df, data_report = ingestion.run()
        pipeline_state["data_report"] = data_report
        add_log(f"Loaded {data_report['total_rows']} rows, {data_report['total_columns']} columns")
        add_log(f"Missing values: {data_report['missing_values']} | Duplicates: {data_report['duplicate_rows']}")
        
        # Stage 2: Preprocessing
        pipeline_state["stage"] = "Preprocessing"
        pipeline_state["progress"] = 30
        add_log(" Stage 2: Preprocessing started...")
        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, Y_train, Y_test, stats = preprocessor.run(df)
        pipeline_state["preprocess_stats"] = stats
        add_log(f" Train set: {stats['train_size']} samples | Test set: {stats['test_size']} sample")
        add_log(f" Features: {stats['num_features']} | Scaling: {stats['scaling']}")
        
        # Stage 3: Model Training
        pipeline_state["stage"] = "Model Training"
        pipeline_state["progress"] = 50
        add_log(f" Stage 3: Training {len(selected_models)} model(s) with MLflow tracking...")
        trainer = ModelTrainer(experiment_name="Wine_Quality_Pipeline")
        results, best = trainer.train_all(X_train, X_test, Y_train, Y_test, selected_models)
        pipeline_state["results"] = results
        pipeline_state["best_model"] = best
        add_log(f" All models trained and logged to MLfow!")
        for r in results:
            add_log(f" → {r['model_name']}: Acc={r['accuracy']:.4f} | F1={r['f1_score']:.4f}")
            
        # Stage 4: Evaluation
        pipeline_state["stage"] = "Evaluation"
        pipeline_state["progress"] = 85
        add_log(" Stage 4: Generating evaluation charts...")
        evaluation = ModelEvaluator(results)
        charts = evaluation.generate_all(best["model_name"])
        pipeline_state["charts"] = charts
        add_log(f" Best model :{best['model_name']} (F1={best['f1_score']:.4f})")
        
        # Done
        pipeline_state["stage"] = "Complete"
        pipeline_state["progress"] = 100
        pipeline_state["status"] = "done"
        add_log(" pipeline completed successfully!", "SUCCESS")
        
    except Exception as e:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = str(e)
        add_log(f"❌ Pipeline failed: {str(e)}", "ERROR")
        logger.exception("Pipeline error")
        
# Routes

@app.route("/")
def index():
    return render_template("index.html", models=list(MODELS.keys()))

@app.route("/run_pipeline", methods=['POST'])
def start_pipeline():
    if pipeline_state["status"] == "running":
        return jsonify({"error": "Pipeline already running"}), 400
    selected = request.form.getlist("models")
    if not selected:
        selected = list(MODELS.keys())
    # Reset state
    pipeline_state.update({"results": [], "best_model": None, "charts":None, "data_report": None, "preprocess_stats": None})
    thread = threading.Thread(target=run_pipeline, args=(selected, ), daemon=True)
    thread.start()
    return redirect(url_for("monitor"))

@app.route("/monitor")
def monitor():
    return render_template("monitor.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": pipeline_state["status"],
        "stage": pipeline_state["stage"],
        "progress": pipeline_state["progress"],
        "logs": pipeline_state["logs"][-30:],
        "error": pipeline_state["error"]
    })

@app.route("/results")
def results():
    if not pipeline_state["results"]:
        return redirect(url_for("index"))
    return render_template(
        "results.html",
        results=pipeline_state["results"],
        best=pipeline_state["best_model"],
        data_report=pipeline_state["data_report"],
        preprocess_stats=pipeline_state["preprocess_stats"]
    )
    
@app.route("/metrics")
def metrics():
    if not pipeline_state["charts"]:
        return redirect(url_for("index"))
    
    print(type(pipeline_state["results"]))
    print(pipeline_state["results"])
    
    return render_template(
        "metrics.html",
        charts=pipeline_state["charts"],
        results=pipeline_state["results"],
        best=pipeline_state["best_model"]
    )
    
@app.route("/predict", methods=["GET","POST"])
def predict():
    feature_names = []
    prediction = None
    error = None
    
    #load feature names
    fn_path = "models/feature_names.pkl"
    if os.path.exists(fn_path):
        feature_names = joblib.load(fn_path)
        
    best_model_path = None
    best_info_path = "models/best_model.json"
    if os.path.exists(best_info_path):
        with open(best_info_path) as f:
            info = json.load(f)
        best_model_path = f"models/{info['model_name'].replace(' ', '_').lower()}.pkl"
        
    if request.method == "POST":
        try:
            values = [float(request.form.get(f, 0)) for f in feature_names]
            X = np.array(values).reshape(1, -1)
            
            #Apply saved preprocessing
            imputer = joblib.load("models/imputer.pkl")
            scaler = joblib.load("models/scaler.pkl")
            model = joblib.load(best_model_path)
            
            X = imputer.transform(X)
            X = scaler.transform(X)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            
            class_map = {0: "Class 0 (Wine Type 0)", 1: "Class 1 (Wine Type 1)", 2: "Class 2 (Wine Type 2)"}
            prediction = {
                "class": int(pred),
                "label": class_map.get(int(pred), f"Class {pred}"),
                "confidence": round(float(max(proba)) * 100, 2),
                "probabilities": {f"Class {i}": round(float(p) * 100,2) for i, p in enumerate(proba)}
                
            }
        except Exception as e:
            error = str(e)
            
    return render_template(
        "predict.html",
        feature_names=feature_names,
        prediction = prediction,
        error = error,
        model_ready=os.path.exists(best_model_path) if best_model_path else False
    )
    
@app.route("/experiments")
def experiments():
    """List MLflow experiments and runs."""
    import mlflow
    mlflow.set_tracking_uri("mlruns")
    try:
        client = mlflow.tracking.MLflowClient()
        experiments = client.search_experiments()
        exp_data = []
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id, order_by=["metrics.f1_score DESC"])
            run_list = []
            for run in runs[:10]:
                run_list.append({
                    "run_id": run.info.run_id[:8],
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "accuracy": round(run.data.metrics.get("accuracy", 0), 4),
                    "f1_score": round(run.data.metrics.get("f1_score", 0), 4),
                    "precision": round(run.data.metrics.get("precision", 0), 4),
                    "recall": round(run.data.metrics.get("recall", 0), 4),
                    
                })
            exp_data.append({
            "name": exp.name,
            "id": exp.experiment_id,
            "runs": run_list
        })
    except Exception as e:
        exp_data = []
        error = str(e)
    return render_template("experiment.html",experiments=exp_data)
    
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/charts", exist_ok=True)
    print("\n" + "="*55)
    print(" 🚀 ML Pipeline with MLflow - Starting...")
    print(" 📍 Open: http://127.0.0.1:5000")
    print(" 📊 MLflow UI: run `mlflow ui` in terminal")
    print("="*55 + "\n")
    app.run(debug=True, use_reloader=False, port=5000)