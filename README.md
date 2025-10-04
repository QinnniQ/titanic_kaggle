# Titanic ML Pipeline (Scripts-First)

A simple, reproducible pipeline for the Kaggle **Titanic** challenge.  
Scripts > notebooks: the pipeline runs end-to-end from the command line, while notebooks are kept for EDA.

## Project Structure
```
titanic/
├─ data/
│  ├─ raw/           # put Kaggle CSVs here: train.csv, test.csv
│  └─ processed/     # created by preprocess step
├─ outputs/
│  ├─ figures/
│  └─ submissions/   # final CSV for Kaggle
├─ notebooks/        # optional EDA
├─ src/
│  └─ titanic/
│     ├─ __init__.py
│     ├─ data.py
│     ├─ preprocess.py
│     ├─ models.py
│     └─ predict.py
├─ scripts/
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ submit.py
│  └─ tune_fast.py   # fast RandomizedSearchCV tuner
├─ requirements.txt
├─ pyproject.toml    # enables `pip install -e .`
└─ .gitignore
```

## Setup

```bash
# from repo root
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate

pip install -U pip
pip install -r requirements.txt

# (optional, recommended) install package in editable mode
pip install -e .
```

**Data:** download from Kaggle and place in `data/raw/`:
- `train.csv`
- `test.csv`
- (`gender_submission.csv` optional)

## Run Pipeline

```bash
# 1) Preprocess (raw -> processed)
python -m scripts.preprocess

# 2) Train baseline model (saves outputs/random_forest_model.pkl)
python -m scripts.train

# 3) Evaluate (5-fold CV mean ± std, confusion, report)
python -m scripts.evaluate

# 4) (Optional) Fast tuning (RF + GB, RandomizedSearchCV)
python -m scripts.tune_fast

# 5) (Optional) Tune probability threshold on a val split
python -m scripts.tune_threshold

# 6) Create submission CSV (auto-uses tuned model/threshold if present)
python -m scripts.submit
```

**Outputs**
- Processed data → `data/processed/`
- Models → `outputs/`
- Submission CSV → `outputs/submissions/titanic_submission.csv`

## What’s Inside
- Clean-slate preprocessing (no chained ops), with:
  - Basic imputations (`Age`, `Fare`, `Embarked`)
  - Features: `FamilySize`, `IsAlone`, `Title`, `FarePerPerson`, `TicketGroupSize`, `CabinDeck`
  - One-hot for `Pclass`, `Embarked`, `CabinDeck`, `Title`
- Baseline + tuned models (RF / GradientBoosting)
- Threshold tuning for better class-1 recall
- Reproducible scripts and clear prints at each step

## Submit to Kaggle
Upload `outputs/submissions/titanic_submission.csv` on the competition page (Submit Predictions).

## Notes
- Keep data files out of Git; folders are tracked via `.gitkeep`.
- If Python can’t import `src/`, run scripts with `python -m scripts.<name>` from repo root or `pip install -e .`.

## License
MIT — feel free to use and adapt.
