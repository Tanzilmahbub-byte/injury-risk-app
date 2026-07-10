"""
injury_insights.py
===================
Parses the REAL player_injuries_impact.csv file (fixing the messy '(S)' /
'N.A.' formatting the same way the notebook's preprocess_injuries() did)
and computes the numbers the "Injury Insights" page needs.

Note: this file has no shared player ID with data.csv, so nothing here is
merged into the risk-prediction models — it's shown as its own honest
source of insight about injury cases, separate from the risk predictions.
"""
import os
import re

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INJ_PATH = os.path.join(BASE_DIR, "data", "player_injuries_impact.csv")


def _clean_numeric(series):
    """Strip trailing markers like '(S)' and turn 'N.A.' into NaN floats."""
    cleaned = series.astype(str).str.replace(r"\(S\)", "", regex=True)
    cleaned = cleaned.replace("N.A.", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def load_injury_data():
    df = pd.read_csv(INJ_PATH)

    for col in ["Date of Injury", "Date of return"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    rating_cols = [c for c in df.columns if "Player_rating" in c]
    gd_cols = [c for c in df.columns if c.endswith("_GD")]
    for col in rating_cols + gd_cols:
        df[col] = _clean_numeric(df[col])

    df["days_out"] = (df["Date of return"] - df["Date of Injury"]).dt.days
    return df


def get_injury_insights(top_n=8):
    df = load_injury_data()

    # normalize casing so "Hamstring injury" and "hamstring injury" merge
    normalized = df["Injury"].astype(str).str.strip().str.lower().str.capitalize()
    counts = normalized.value_counts()
    top = counts.head(top_n)
    other_total = int(counts.iloc[top_n:].sum())
    injury_type_counts = top.to_dict()
    if other_total > 0:
        injury_type_counts["Other"] = other_total

    before_rating_cols = [f"Match{i}_before_injury_Player_rating" for i in (1, 2, 3)]
    after_rating_cols = [f"Match{i}_after_injury_Player_rating" for i in (1, 2, 3)]
    before_gd_cols = [f"Match{i}_before_injury_GD" for i in (1, 2, 3)]
    after_gd_cols = [f"Match{i}_after_injury_GD" for i in (1, 2, 3)]

    avg_rating_before = float(np.nanmean(df[before_rating_cols].values))
    avg_rating_after = float(np.nanmean(df[after_rating_cols].values))
    avg_gd_before = float(np.nanmean(df[before_gd_cols].values))
    avg_gd_after = float(np.nanmean(df[after_gd_cols].values))

    valid_days = df.dropna(subset=["days_out"])
    valid_days = valid_days[(valid_days["days_out"] >= 0) & (valid_days["days_out"] <= 730)]
    top_cases = (
        valid_days.sort_values("days_out", ascending=False)
        .head(10)[["Name", "Injury", "days_out"]]
        .to_dict(orient="records")
    )

    return {
        "total_cases": int(len(df)),
        "injury_type_counts": injury_type_counts,
        "avg_rating_before": round(avg_rating_before, 2),
        "avg_rating_after": round(avg_rating_after, 2),
        "avg_gd_before": round(avg_gd_before, 2),
        "avg_gd_after": round(avg_gd_after, 2),
        "avg_days_out": round(float(valid_days["days_out"].mean()), 1),
        "longest_cases": top_cases,
    }
