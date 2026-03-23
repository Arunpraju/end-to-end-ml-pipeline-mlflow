"""
stage 4: Model Evaluation & Report Generation
Generates charts and comparison tables.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import os
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, results):
        self.results = results
        os.makedirs("static/charts", exist_ok=True)
        
    def metrics_comparison_chart(self):
        """Bar chart comparing all models across metrics."""
        models = [r["model_name"] for r in self.results]
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        colors = ["#4361ee", "#3a0ca3", "#7209b7", "#f72585"]
        
        fig = go.Figure()
        for i, metric in enumerate(metrics):
            values = [r[metric] for r in self.results]
            fig.add_trace(go.Bar(
                name=metric.replace("_", " ").title(),
                x=models,
                y=values,
                marker_color=colors[i],
                text=[f"{v:.3f}" for v in values],
                textposition="outside"
            ))
        fig.update_layout(
            barmode="group",
            title="Model Performance Comparison",
            yaxis=dict(range=[0,1.1], title="score"),
            xaxis_title="Model",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
            font=dict(family="Segoe UI", size=12),
            height=420
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def confusion_matrix_chart(self, model_name):
        """Confusion matrix heatmap for a given model."""
        result = next((r for r in self.results if r["model_name"] == model_name), None)
        if not result:
            return None
        
        cm = np.array(result["confusion_matrix"])
        labels = [f"Class {i}" for i in range(len(cm))]
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            title=f"confusion Matrix - {model_name}",
            text_auto=True
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Segoe UI", size=12),
            height=300,
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    def radar_chart(self):
        """Radar chart for multi-metric model comparison."""
        categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        
        colors = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7"]
        for i, r in enumerate(self.results):
            values = [r["accuracy"],r["precision"],r["recall"], r["f1_score"]]
            values += values[:1]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=r["model_name"],
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Metrics Radar Chart",
            template="plotly_white",
            height=420,
            font=dict(family="Segoe UI", size=12)
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def f1_ranking_chart(self):
        """Horizontal bar chart ranking models by F1 score."""
        sorted_results = sorted(self.results, key=lambda x:x["f1_score"])
        models = [r["model_name"] for r in sorted_results]
        f1_scores = [r["f1_score"] for r in sorted_results]
        
        fig = go.Figure(go.Bar(
            x=f1_scores,
            y=models,
            orientation="h",
            marker=dict(
                color=f1_scores,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="F1 Score")
            ),
            text=[f"{v:.4f}" for v in f1_scores],
            textposition="outside"
        ))
        fig.update_layout(
            title="Model Ranking by F1 Score",
            xaxis=dict(range=[0,1.1], title="F1 Scores"),
            yaxis_title="Model",
            template="plotly_white",
            height=320,
            font=dict(family="Segoe UI", size=12)
        
        )
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def generate_all(self, best_model_name):
        return {
            "metrics_chart": self.metrics_comparison_chart(),
            "radar_chart": self.radar_chart(),
            "f1_ranking_chart": self.f1_ranking_chart(),
            "confusion_matrix": self.confusion_matrix_chart(best_model_name)
        }