from __future__ import annotations

import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

DEFAULT_FEATURE_COLS = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
DEFAULT_TARGET_COL_CLASSIF = "Drug"
DEFAULT_TARGET_COL_REG = "Na_to_K"

def load_dataset(
    path: str,
    feature_cols: Optional[list] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if target_col is None:
        # If the target is present among defaults, choose classification target
        target_col = DEFAULT_TARGET_COL_CLASSIF if DEFAULT_TARGET_COL_CLASSIF in df.columns else DEFAULT_TARGET_COL_REG

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True
):
    strat = y if stratify and y.dtype == 'O' else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
