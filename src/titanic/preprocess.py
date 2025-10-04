from pathlib import Path
import pandas as pd
from .data import load_raw, save_processed

def _extract_title(name: str) -> str:
    # e.g., "Braund, Mr. Owen Harris" -> "Mr"
    try:
        title = name.split(",")[1].split(".")[0].strip()
    except Exception:
        return "Rare"
    # normalize common variants
    mapping = {
        "Mlle": "Miss", "Ms": "Miss",
        "Mme": "Mrs", "Lady": "Rare", "Countess": "Rare", "Dona": "Rare",
        "Sir": "Rare", "Jonkheer": "Rare", "Don": "Rare", "Rev": "Rare",
        "Col": "Rare", "Major": "Rare", "Capt": "Rare", "Dr": "Rare"
    }
    return mapping.get(title, title if title in {"Mr","Mrs","Miss","Master"} else "Rare")

def run_preprocessing(raw_dir: Path, proc_dir: Path) -> None:
    train_df, test_df = load_raw(raw_dir)
    train_df["__is_train__"] = 1
    test_df["__is_train__"]  = 0

    # ---------------- Impute basic missing ----------------
    age_median   = train_df["Age"].median()
    fare_median  = (train_df["Fare"].median()
                    if "Fare" in train_df.columns else test_df["Fare"].median())
    embarked_mode = train_df["Embarked"].mode(dropna=True)[0]

    for df in (train_df, test_df):
        df["Age"]      = df["Age"].fillna(age_median)
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    if "Fare" in train_df.columns:
        train_df["Fare"] = train_df["Fare"].fillna(fare_median)
    test_df["Fare"] = test_df["Fare"].fillna(fare_median)

    # ---------------- Feature engineering ----------------
    for df in (train_df, test_df):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"]    = (df["FamilySize"] == 1).astype("Int64")
        df["Title"]      = df["Name"].apply(_extract_title)
        df["FarePerPerson"] = (df["Fare"] / df["FamilySize"]).replace([float("inf")], 0).fillna(0)
        # Ticket group size
        # (compute on concatenated to align across splits)
    full = pd.concat([train_df, test_df], ignore_index=True)

    # Ticket group size on full
    ticket_counts = full["Ticket"].value_counts()
    full["TicketGroupSize"] = full["Ticket"].map(ticket_counts).astype("Int64")

    # CabinDeck (first letter), U = unknown
    full["CabinDeck"] = full["Cabin"].astype(str).str.strip().str[0].where(full["Cabin"].notna(), "U")
    full.loc[~full["CabinDeck"].isin(list("ABCDEFGT")), "CabinDeck"] = "U"  # collapse odd letters

    # Encode Sex
    sex_map = {"male": 0, "female": 1}
    if full["Sex"].dtype == "object":
        full["Sex"] = (full["Sex"].astype(str).str.lower().map(sex_map)).astype("Int64")
    else:
        full["Sex"] = full["Sex"].astype("Int64")

    # One-hot for Embarked, Pclass, CabinDeck, Title
    cat_cols = ["Embarked", "Pclass", "CabinDeck", "Title"]
    full = pd.get_dummies(full, columns=cat_cols, drop_first=True)

    # Safe drop high-cardinality originals
    drop_cols = ["Name", "Ticket", "Cabin"]
    full = full.drop(columns=[c for c in drop_cols if c in full.columns], errors="ignore")

    # Split back
    train_p = full[full["__is_train__"] == 1].drop(columns="__is_train__").copy()
    test_p  = full[full["__is_train__"] == 0].drop(columns=["__is_train__", "Survived"], errors="ignore").copy()

    save_processed(train_p, test_p, proc_dir)
    print("✅ Preprocessing complete (with Title, TicketGroupSize, FarePerPerson, CabinDeck, OHE).")
