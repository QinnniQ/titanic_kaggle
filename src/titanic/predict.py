from pathlib import Path
import pandas as pd
import joblib

def make_submission(model_path: Path, test_df: pd.DataFrame, raw_test_path: Path, out_csv: Path):
    model = joblib.load(model_path)
    preds = model.predict(test_df).astype(int)
    raw_test = pd.read_csv(raw_test_path)

    sub = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": preds})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print("Pred counts:", sub["Survived"].value_counts().to_dict())
    print(f"🚀 Wrote submission -> {out_csv.resolve()}")
