from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    df = pd.read_csv(proj / "data/processed/train_processed.csv")
    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grids = [
        ("RandomForest",
         RandomForestClassifier(random_state=42, n_jobs=-1),
         {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 6, 10, 14],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
         }),
        ("GradientBoosting",
         GradientBoostingClassifier(random_state=42),
         {
            "n_estimators": [150, 250, 400],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_depth": [2, 3, 4]
         }),
    ]

    best_score, best_name, best_est = -1, None, None
    for name, est, param_grid in grids:
        grid = GridSearchCV(est, param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=0)
        grid.fit(X, y)
        print(f"\n{name} best: {round(grid.best_score_,4)}  params: {grid.best_params_}")
        if grid.best_score_ > best_score:
            best_score, best_name, best_est = grid.best_score_, name, grid.best_estimator_

    outdir = proj / "outputs"; outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"best_{best_name}.pkl"
    joblib.dump(best_est, path)
    print(f"\n🏆 Saved best model -> {path}")
