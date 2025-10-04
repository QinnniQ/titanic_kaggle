# scripts/tune_threshold.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    proc = proj / "data" / "processed"
    out  = proj / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    df = pd.read_csv(proc / "train_processed.csv")
    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    # ---- split (same seed/stratify as before) ----
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- pick model: tuned GB > tuned RF > baseline RF ----
    candidates = [
        out / "best_GradientBoosting.pkl",
        out / "best_RandomForest.pkl",
        out / "random_forest_model.pkl",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No model found in outputs/. Train or tune a model first.")

    model = joblib.load(model_path)
    print(f"Using model for threshold tuning: {model_path.name}")

    # ---- need predict_proba ----
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Model {model_path.name} has no predict_proba(). Use GB/RF or a probabilistic model.")

    # ---- get probabilities on validation ----
    p_va = model.predict_proba(X_va)[:, 1]

    # ---- scan thresholds ----
    thresholds = np.linspace(0.25, 0.75, 51)
    best_acc, best_acc_t = -1.0, 0.5
    best_f1,  best_f1_t  = -1.0, 0.5

    for t in thresholds:
        preds = (p_va >= t).astype(int)
        acc = accuracy_score(y_va, preds)
        f1  = f1_score(y_va, preds)
        if acc > best_acc:
            best_acc, best_acc_t = acc, t
        if f1 > best_f1:
            best_f1, best_f1_t = f1, t

    result = {
        "model_file": model_path.name,
        "best_accuracy": round(float(best_acc), 4),
        "best_accuracy_threshold": round(float(best_acc_t), 4),
        "best_f1": round(float(best_f1), 4),
        "best_f1_threshold": round(float(best_f1_t), 4),
        "metric_used": "accuracy"   # change to "f1" if you prefer that
    }

    with open(out / "threshold.json", "w") as f:
        json.dump(result, f, indent=2)

    print("✅ Threshold search complete.")
    print(result)
    print(f"Saved -> { (out / 'threshold.json').resolve() }")
