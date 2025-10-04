from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def split_xy(train_df: pd.DataFrame):
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)
    return X, y

def train_baseline(train_df: pd.DataFrame, outdir: Path) -> Path:
    X, y = split_xy(train_df)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xva)
    acc = accuracy_score(yva, preds)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(yva, preds, digits=3))

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Saved model -> {model_path}")
    return model_path

def cv_score(train_df: pd.DataFrame, model=None, n_splits=5):
    X, y = split_xy(train_df)
    if model is None:
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"CV mean={scores.mean():.4f}  std={scores.std():.4f}")
    return scores
