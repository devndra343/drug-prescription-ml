# Drug Prescription ML

Reproducible ML mini-project that:
1. **Classifies** the prescribed drug from patient attributes.
2. **Regresses** the Sodium-to-Potassium (Na/K) ratio.
3. **Clusters** patients with DBSCAN to explore natural groups.

> This repository is a clean-room reimplementation adapted from the ideas in *Noorjahan-aktar/DrugPredictionML*.  
> Please **add the original dataset** (`drug200.csv`) to `data/` before running.

## Dataset
- Expected file: `data/drug200.csv`
- Columns (typical): `Age, Sex, BP, Cholesterol, Na_to_K, Drug`
- Target (classification): `Drug`
- Target (regression): `Na_to_K`

## Quickstart

### Environment Setup (Recommended)

Create the environment **inside the project folder** so it stays self-contained.

```bash
# Go to the project folder
cd path/to/drug-prescription-ml

# Create virtual environment in .venv folder
python -m venv .venv

# Activate environment
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```


### 1) Create environment & install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Put the dataset in place
Place `drug200.csv` into `data/`. If your file uses different column names,
you can pass `--feature-cols` / `--target-col` to the CLI (see below).

### 3) Run experiments
**Classification (SVM or Naive Bayes):**
```bash
python -m src.classification --model svm --test-size 0.2 --seed 42 --outdir results
python -m src.classification --model nb  --test-size 0.2 --seed 42 --outdir results
```

**Regression (Linear Regression):**
```bash
python -m src.regression --test-size 0.2 --seed 42 --outdir results
```

**Clustering (DBSCAN):**
```bash
python -m src.clustering --eps 1.0 --min-samples 5 --outdir results
```

### 4) Outputs
All metrics and plots are saved under `results/`:
- `clf_metrics_<model>.json`, `confusion_matrix_<model>.png`
- `reg_metrics.json`, `reg_pred_vs_true.png`
- `cluster_summary.json`, `cluster_scatter.png`

## Reproducibility
- Deterministic splits via `--seed`
- Pipelines with proper **scaling** and **one-hot encoding**
- Pinned core dependencies in `requirements.txt`

## Attribution
This work is inspired by: `Noorjahan-aktar/DrugPredictionML`.
Please cite or link the original notebook if you use their dataset/notebook directly.
