from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .data import load_dataset, split, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL_CLASSIF
from .plots import ensure_outdir, save_confusion_matrix

def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == 'O']
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    if model_name == "svm":
        clf = SVC(kernel="rbf", probability=False, random_state=42)
    elif model_name == "nb":
        # GaussianNB needs dense arrays (handled after preprocessor)
        clf = GaussianNB()
    else:
        raise ValueError("model must be one of: svm, nb")

    # For NB, we need to convert the sparse matrix to dense before NB
    if model_name == "nb":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer
        to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
        pipe = Pipeline(steps=[("pre", pre), ("to_dense", to_dense), ("clf", clf)])
    else:
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    return pipe

def main():
    ap = argparse.ArgumentParser(description="Drug Prescription Classification")
    ap.add_argument("--data-path", default="data/drug200.csv")
    ap.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURE_COLS)
    ap.add_argument("--target-col", default=DEFAULT_TARGET_COL_CLASSIF)
    ap.add_argument("--model", choices=["svm", "nb"], default="svm")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    X, y = load_dataset(args.data_path, args.feature_cols, args.target_col)
    Xtr, Xte, ytr, yte = split(X, y, test_size=args.test_size, seed=args.seed, stratify=True)

    pipe = build_pipeline(args.model, X)
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    acc = accuracy_score(yte, yhat)
    f1 = f1_score(yte, yhat, average="weighted")
    report = classification_report(yte, yhat, output_dict=True)

    ensure_outdir(args.outdir)
    metrics = {
        "model": args.model,
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
    }
    with open(os.path.join(args.outdir, f"clf_metrics_{args.model}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    labels_sorted = sorted(pd.unique(y))
    save_confusion_matrix(yte, yhat, labels_sorted, os.path.join(args.outdir, f"confusion_matrix_{args.model}.png"),
                          title=f"Confusion Matrix ({args.model.upper()})")

    print(f"[OK] Saved metrics and confusion matrix in: {args.outdir}")

if __name__ == "__main__":
    main()
