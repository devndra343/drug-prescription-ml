from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .data import load_dataset, split, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL_REG
from .plots import ensure_outdir, save_scatter

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = [c for c in X.columns if X[c].dtype == 'O']
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = LinearRegression()
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe

def main():
    ap = argparse.ArgumentParser(description="Na/K Regression (Linear Regression)")
    ap.add_argument("--data-path", default="data/drug200.csv")
    ap.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURE_COLS)
    ap.add_argument("--target-col", default=DEFAULT_TARGET_COL_REG)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    X, y = load_dataset(args.data_path, args.feature_cols, args.target_col)
    Xtr, Xte, ytr, yte = split(X, y, test_size=args.test_size, seed=args.seed, stratify=False)

    pipe = build_pipeline(X)
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    r2 = r2_score(yte, yhat)
    mae = mean_absolute_error(yte, yhat)
    rmse = mean_squared_error(yte, yhat, squared=False)

    ensure_outdir(args.outdir)
    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }
    with open(os.path.join(args.outdir, "reg_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_scatter(yte, yhat, os.path.join(args.outdir, "reg_pred_vs_true.png"),
                 title="Regression: Predicted vs True", xlabel="True Na/K", ylabel="Predicted Na/K")

    print(f"[OK] Saved regression metrics and plot in: {args.outdir}")

if __name__ == "__main__":
    main()
