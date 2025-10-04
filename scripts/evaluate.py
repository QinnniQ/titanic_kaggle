from pathlib import Path
import pandas as pd
from src.titanic.models import cv_score

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    train_df = pd.read_csv(proj / "data/processed/train_processed.csv")
    cv_score(train_df)
