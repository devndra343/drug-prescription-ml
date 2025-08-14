# data/


Expected columns (default):
- Age (numeric)
- Sex (categorical; e.g., M/F)
- BP (categorical; e.g., HIGH, NORMAL, LOW)
- Cholesterol (categorical; e.g., HIGH, NORMAL)
- Na_to_K (numeric)  # sometimes written as 'Na/K' or 'Na_to_K'
- Drug (categorical; e.g., DrugA..DrugE)

If your column names differ, pass them via CLI flags, e.g.:
```
python -m src.classification --feature-cols Age Sex BP Cholesterol Na_to_K --target-col Drug
```
