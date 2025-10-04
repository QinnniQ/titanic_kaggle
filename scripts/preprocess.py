from pathlib import Path
from src.titanic.preprocess import run_preprocessing

if __name__ == "__main__":
    proj = Path(__file__).resolve().parents[1]
    raw = proj / "data" / "raw"
    proc = proj / "data" / "processed"
    run_preprocessing(raw, proc)
