# City-Level 80/20 Holdout Evaluation

Self-contained experiment in `tests/city_holdout_80_20/`.

This folder **only reads** the master CSV from the project root and writes results to `./output/`. It does **not** change any existing notebooks or CSVs.

---

## Overview

| Item | Detail |
|------|--------|
| Purpose | Evaluate geographic generalization when train/test are split by **whole cities** |
| Data | `../../Master_Hexagon_Gun_Violence_Summary_Table.csv` |
| Target | `Total_Gun_Violence > 0` (binary) |
| Split | ~80% / ~20% hexes, `seed=42`, no city in both sets |
| Notebook | `notebook.ipynb` (executed end-to-end) |

---

## Method

### 1. Preprocessing (same as main pipeline)

1. Drop leakage / outcome fields (`Accidental_*`, `Children_*`, `Teens_*`, `Mass_Shootings`, `Bars`)
2. Drop geometry and admin fields (`Shape_*`, `QA_Flag`, `aggregationMethod`, etc.)
3. Create percentage features (`Black_Percentage`, `White_Percentage`, `Poverty_Rate_Pct`, education `%`) and remove matching raw counts
4. Use `City_name` for the split only; exclude it from model features

### 2. City holdout split (`seed=42`)

1. Shuffle cities with `numpy.RandomState(42)`
2. Cut at the prefix whose cumulative hex count is closest to 20%
3. Save assignments to `output/city_assignment.csv`

### 3. Scaling and feature selection (train-only fit)

1. Fit `MinMaxScaler` on train; transform train and test
2. Iterative VIF pruning on train until all VIF ≤ 5
3. Drop one feature from each train pair with `|r| ≥ 0.75` (keep the one more correlated with the continuous target)
4. Apply the final feature list to test cities (**21 features**)

### 4. Models (same hyperparameters as main pipeline)

| Model | Key settings |
|-------|----------------|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `max_depth=15`, balanced |
| Gradient Boosting | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1` |
| XGBoost | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `scale_pos_weight` |
| LightGBM | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, balanced |

### 5. Evaluation

- Report Accuracy, Precision, Recall, F1, and ROC-AUC at threshold **0.50**
- Select best model by F1 @ 0.50
- Tune that model’s threshold on `[0.10, 0.90]` (step `0.01`) to maximize F1

---

## Split summary

| Set | Cities | Hexes | Share |
|-----|-------:|------:|------:|
| Train | 28 | 9,528 | 83.2% |
| Test | 7 | 1,920 | 16.8% |
| **Total** | **35** | **11,448** | **100%** |

**Test cities:** Indianapolis, Kansas City, Milwaukee, New Orleans, Oakland, Philadelphia, Portland

**Train cities:** Atlanta, Baltimore, Birmingham, Boston, Charlotte, Chicago, Cincinnati, Cleveland, Columbus, Dallas, Denver, Detroit, Houston, Jacksonville, Las Vegas, Los Angeles, Memphis, Nashville, Norfolk, North Las Vegas, Oklahoma City, Phoenix, Pittsburgh, Richmond, Rochester, Saint Louis, San Antonio, Seattle

---

## Results

### Metrics at threshold 0.50

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| **Random Forest** | 0.8953 | 0.6578 | 0.6092 | **0.6325** | 0.9051 |
| XGBoost | 0.8547 | 0.5055 | 0.8028 | 0.6204 | 0.9141 |
| LightGBM | 0.8484 | 0.4925 | 0.8134 | 0.6135 | 0.9141 |
| Logistic Regression | 0.8464 | 0.4882 | 0.8028 | 0.6072 | 0.9101 |
| Gradient Boosting | 0.8932 | 0.7182 | 0.4577 | 0.5591 | 0.9141 |

### Best model and threshold tuning

| | Value |
|--|------:|
| Best model (F1 @ 0.50) | **Random Forest** |
| F1 @ 0.50 | **0.6325** |
| Tuned threshold | **0.42** |
| F1 @ tuned threshold | **0.6555** |
| Precision / Recall @ tuned | 0.6242 / 0.6901 |
| Accuracy @ tuned | 0.8927 |

---

## Outputs

All files are under `output/`:

| File | Description |
|------|-------------|
| `city_assignment.csv` | City → train/test + hex count |
| `metrics.csv` | Per-model metrics at threshold 0.50 |
| `predictions.csv` | Test-hex probabilities and predictions |
| `results.json` | Full machine-readable summary |

---

## How to re-run

From this folder (project `venv` recommended):

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace --ExecutePreprocessor.timeout=600
```

Or open `notebook.ipynb` and run all cells.

---

## Isolation

- **Read-only** access to `../../Master_Hexagon_Gun_Violence_Summary_Table.csv`
- **No writes** to project-root CSVs or notebooks
- All artifacts stay inside `tests/city_holdout_80_20/`
