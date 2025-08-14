import os
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def save_confusion_matrix(y_true, y_pred, labels, outpath: str, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def save_scatter(x: np.ndarray, y: np.ndarray, outpath: str, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def save_cluster_scatter(X_2d: np.ndarray, labels: np.ndarray, outpath: str, title: str = "DBSCAN Clusters"):
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for ul in unique_labels:
        mask = labels == ul
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {ul}" if ul!=-1 else "Noise", alpha=0.8)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
