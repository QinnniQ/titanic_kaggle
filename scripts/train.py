from pathlib import Path
import pandas as pd
from src.titanic.models import train_baseline

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    train_df = pd.read_csv(proj / "data/processed/train_processed.csv")
    model_path = train_baseline(train_df, proj / "outputs")
    print("✅ Training finished:", model_path)
