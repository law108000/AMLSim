#!/usr/bin/env python3
"""Compare all trained models."""

import json
from pathlib import Path
import pandas as pd

def main():
    model_dir = Path("training/models")
    metrics_files = list(model_dir.glob("*_metrics.json"))
    
    results = []
    for f in metrics_files:
        if "falcon" in f.name: continue # Skip falcon if it failed or wasn't part of this batch
        
        name = f.name.replace("sample_", "").replace("_metrics.json", "")
        if name == "metrics": name = "logistic_regression"
        
        data = json.loads(f.read_text())
        results.append({
            "Model": name,
            "ROC AUC": data.get("roc_auc", 0),
            "Avg Precision": data.get("average_precision", 0),
            "Recall (Class 1)": data.get("classification_report", {}).get("1", {}).get("recall", 0) if "classification_report" in data else 0,
            "Precision (Class 1)": data.get("classification_report", {}).get("1", {}).get("precision", 0) if "classification_report" in data else 0
        })
        
    df = pd.DataFrame(results).sort_values("Avg Precision", ascending=False)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
