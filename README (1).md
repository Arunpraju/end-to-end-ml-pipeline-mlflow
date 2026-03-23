# 🚀 End-to-End ML Pipeline with MLflow
**Built with:** Python · Flask · MLflow · scikit-learn · Bootstrap 5 · Plotly

---

## 📁 Project Structure

```
ml_pipeline_project/
│
├── app.py                        # Flask app (main entry point)
│
├── pipeline/
│   ├── __init__.py
│   ├── data_ingestion.py         # Stage 1: Load & validate data
│   ├── preprocessing.py          # Stage 2: Scale, impute, split
│   ├── model_trainer.py          # Stage 3: Train + MLflow logging
│   └── evaluator.py              # Stage 4: Charts & evaluation
│
├── templates/
│   ├── base.html                 # Base layout + Bootstrap navbar
│   ├── index.html                # Home (pipeline launcher)
│   ├── monitor.html              # Live pipeline monitor
│   ├── results.html              # Model comparison table
│   ├── metrics.html              # Plotly charts & reports
│   ├── predict.html              # Live prediction form
│   └── experiments.html          # MLflow experiment viewer
│
├── static/
│   ├── css/style.css             # Custom dark theme styles
│   └── js/main.js                # UI enhancements
│
├── data/                         # Auto-generated dataset
├── models/                       # Saved models & scalers
├── mlruns/                       # MLflow tracking store
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```
Open: **http://127.0.0.1:5000**

### 4. (Optional) Launch MLflow UI
In a **separate terminal**, from the project folder:
```bash
mlflow ui --port 5001
```
Open: **http://127.0.0.1:5001**

---

## 🔄 Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_ingestion.py` | Loads Wine Quality dataset, runs quality checks |
| 2 | `preprocessing.py` | StandardScaler, mean imputation, 80/20 split |
| 3 | `model_trainer.py` | Trains 4 models, logs all runs to MLflow |
| 4 | `evaluator.py` | Generates Plotly charts, confusion matrix |

---

## 🤖 Models Trained

- **Random Forest** — Ensemble of 100 decision trees
- **Logistic Regression** — Multinomial with L2 regularization
- **Gradient Boosting** — Sequential boosting, lr=0.1
- **SVM** — RBF kernel, C=1.0

---

## 📊 Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Launch pipeline, view model info |
| Monitor | `/monitor` | Live progress + log console |
| Results | `/results` | Comparison table with best model |
| Metrics | `/metrics` | Interactive Plotly charts |
| Predict | `/predict` | Enter features, get prediction |
| Experiments | `/experiments` | MLflow runs browser |

---

## 🎯 Dataset

**Wine Quality (UCI / sklearn)**
- 178 samples
- 13 chemical features (alcohol, ash, flavanoids, etc.)
- 3 wine classes (class_0, class_1, class_2)
- Task: Multiclass classification

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 2.3 |
| ML | scikit-learn, XGBoost |
| Tracking | MLflow 2.8 |
| Frontend | Bootstrap 5, Plotly.js |
| Charts | Plotly Python, Plotly.js |
| Serialization | joblib |

---

## 💡 Tips

- Click **"Fill Sample"** on the Predict page to auto-populate typical wine values
- Re-run the pipeline anytime to generate new MLflow runs
- All models are saved in `models/` as `.pkl` files
- Feature names, scaler, and imputer are also saved for consistent inference
