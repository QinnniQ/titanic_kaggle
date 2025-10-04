# scripts/submit.py
from pathlib import Path
import pandas as pd
import joblib
import json

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    out_dir = proj / "outputs"
    sub_dir = out_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # --- pick tuned model if available, else fallback
    tuned_rf  = out_dir / "best_RandomForest.pkl"
    tuned_gb  = out_dir / "best_GradientBoosting.pkl"
    fallback  = out_dir / "random_forest_model.pkl"

    if tuned_gb.exists():
        model_path = tuned_gb
    elif tuned_rf.exists():
        model_path = tuned_rf
    else:
        model_path = fallback

    print(f"Using model: {model_path.name}")
    model = joblib.load(model_path)

    # --- load data ---
    test_df = pd.read_csv(proj / "data/processed/test_processed.csv")
    raw_test = pd.read_csv(proj / "data/raw/test.csv")

    # --- try to load threshold (optional) ---
    threshold_path = out_dir / "threshold.json"
    use_threshold = False
    thr = 0.5
    if threshold_path.exists() and hasattr(model, "predict_proba"):
        try:
            cfg = json.loads(threshold_path.read_text())
            # If you prefer F1 threshold, change the key here:
            thr = float(cfg.get("best_accuracy_threshold", 0.5))
            use_threshold = True
            print(f"Applying tuned threshold: {thr:.3f}")
        except Exception as e:
            print("Warning: could not parse threshold.json; falling back to default 0.5. Error:", e)

    # --- predict ---
    if use_threshold:
        probs = model.predict_proba(test_df)[:, 1]
        preds = (probs >= thr).astype(int)
    else:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(test_df)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = model.predict(test_df).astype(int)

    submission = pd.DataFrame({
        "PassengerId": raw_test["PassengerId"],
        "Survived": preds
    })

    out_path = sub_dir / "titanic_submission.csv"
    submission.to_csv(out_path, index=False)

    print("Pred counts:", submission["Survived"].value_counts().to_dict())
    print("🚀 Wrote:", out_path.resolve())
