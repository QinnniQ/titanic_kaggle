from pathlib import Path
import pandas as pd

def load_raw(raw_dir: Path):
    train = pd.read_csv(raw_dir / "train.csv")
    test  = pd.read_csv(raw_dir / "test.csv")
    return train, test

def save_processed(train_df, test_df, proc_dir: Path):
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(proc_dir / "train_processed.csv", index=False)
    test_df.to_csv(proc_dir / "test_processed.csv", index=False)

def load_processed(proc_dir: Path):
    train = pd.read_csv(proc_dir / "train_processed.csv")
    test  = pd.read_csv(proc_dir / "test_processed.csv")
    return train, test
