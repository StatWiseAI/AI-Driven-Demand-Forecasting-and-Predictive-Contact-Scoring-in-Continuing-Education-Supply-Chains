# pipeline/scorer.py
import pickle
from pathlib import Path
import pandas as pd
from config import MODEL_PATH, ABC_THRESHOLDS
from pipeline.feature_engineer import build_features

def score_contacts(df: pd.DataFrame, event: dict, model_path: str = MODEL_PATH) -> pd.DataFrame:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No model at '{model_path}'. Train first.")
    model = pickle.load(open(model_path, "rb"))
    X = build_features(df, event)
    df = df.copy()
    df["score"] = model.predict_proba(X)[:, 1]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["abc_tier"] = _abc(df["score"])
    return df

def _abc(scores: pd.Series) -> pd.Series:
    n = len(scores)
    ac = int(n * ABC_THRESHOLDS["A"]); bc = int(n * ABC_THRESHOLDS["B"])
    t = pd.Series("C", index=scores.index)
    t.iloc[:ac] = "A"; t.iloc[ac:bc] = "B"
    return t
