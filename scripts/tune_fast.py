# scripts/tune_fast.py
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    df = pd.read_csv(proj / "data/processed/train_processed.csv")
    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # --- Random Forest search (balanced & parallel)
    rf = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    rf_dist = {
        "n_estimators": np.linspace(300, 900, 7, dtype=int),
        "max_depth": [None, 6, 8, 10, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_dist,
        n_iter=20,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )
    rf_search.fit(X, y)
    print("\nRF best:", round(rf_search.best_score_, 4), rf_search.best_params_)

    # --- Gradient Boosting search
    gb = GradientBoostingClassifier(random_state=42)
    gb_dist = {
        "n_estimators": np.linspace(150, 400, 6, dtype=int),
        "learning_rate": [0.02, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [1.0, 0.9, 0.8],
    }
    gb_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=gb_dist,
        n_iter=20,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )
    gb_search.fit(X, y)
    print("\nGB best:", round(gb_search.best_score_, 4), gb_search.best_params_)

    # --- Pick the winner & save
    if gb_search.best_score_ >= rf_search.best_score_:
        best_name, best_est = "GradientBoosting", gb_search.best_estimator_
    else:
        best_name, best_est = "RandomForest", rf_search.best_estimator_

    out = proj / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"best_{best_name}.pkl"
    joblib.dump(best_est, path)
    print(f"\n🏆 Saved best model → {path}")
