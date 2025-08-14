from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from .data import load_dataset, DEFAULT_FEATURE_COLS
from .plots import ensure_outdir, save_cluster_scatter

def main():
    ap = argparse.ArgumentParser(description="DBSCAN Clustering for Patients")
    ap.add_argument("--data-path", default="data/drug200.csv")
    ap.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURE_COLS)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    X, _ = load_dataset(args.data_path, args.feature_cols, target_col=None)

    cat_cols = [c for c in X.columns if X[c].dtype == 'O']
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pipe = Pipeline(steps=[("pre", pre)])
    Xproc = pipe.fit_transform(X)

    db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    labels = db.fit_predict(Xproc)

    # 2D projection for visualization
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xproc.toarray() if hasattr(Xproc, "toarray") else Xproc)

    ensure_outdir(args.outdir)
    save_cluster_scatter(X2, labels, os.path.join(args.outdir, "cluster_scatter.png"))

    # summary
    unique, counts = np.unique(labels, return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique, counts)}
    with open(os.path.join(args.outdir, "cluster_summary.json"), "w") as f:
        json.dump({"eps": args.eps, "min_samples": args.min_samples, "cluster_counts": summary}, f, indent=2)

    print(f"[OK] Saved clustering summary and plot in: {args.outdir}")
    print(f"Clusters found (label:-1 is noise): {summary}")

if __name__ == "__main__":
    main()
