import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('Gun_Violence_VIF_Clean.csv', index_col=None)
if 'GRID_ID' in df.columns:
    df = df.drop(columns=['GRID_ID'])

X = df.drop(columns=['Total_Gun_Violence'])
X = X.loc[:, X.apply(lambda s: pd.api.types.is_numeric_dtype(s))]
y = (df['Total_Gun_Violence'] > 0).astype(int)

print(f"Dataset: {df.shape[0]} rows | Class balance: {np.bincount(y)} (no violence, any violence)\n")

# ── Models & metrics ───────────────────────────────────────────────────────────
models = {
    'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost':              XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM':             LGBMClassifier(random_state=42, verbose=-1),
}
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# ── K-Fold CV ──────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_results = {m: {met: [] for met in metrics} for m in models}

for model_name, model in models.items():
    print(f"K-Fold: {model_name} ...", flush=True)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        yp  = model.predict(X.iloc[te])
        ypr = model.predict_proba(X.iloc[te])[:, 1]
        kfold_results[model_name]['accuracy'].append(accuracy_score(y.iloc[te], yp))
        kfold_results[model_name]['precision'].append(precision_score(y.iloc[te], yp, zero_division=0))
        kfold_results[model_name]['recall'].append(recall_score(y.iloc[te], yp, zero_division=0))
        kfold_results[model_name]['f1'].append(f1_score(y.iloc[te], yp, zero_division=0))
        kfold_results[model_name]['roc_auc'].append(roc_auc_score(y.iloc[te], ypr))

kfold_summary = {}
for m in models:
    kfold_summary[m] = {f'{met}_mean': np.mean(kfold_results[m][met]) for met in metrics}
    kfold_summary[m].update({f'{met}_std': np.std(kfold_results[m][met]) for met in metrics})

# ── Train-test split ───────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
split_results = {}
for model_name, model in models.items():
    model.fit(X_tr, y_tr)
    yp  = model.predict(X_te)
    ypr = model.predict_proba(X_te)[:, 1]
    split_results[model_name] = {
        'accuracy':  accuracy_score(y_te, yp),
        'precision': precision_score(y_te, yp, zero_division=0),
        'recall':    recall_score(y_te, yp, zero_division=0),
        'f1':        f1_score(y_te, yp, zero_division=0),
        'roc_auc':   roc_auc_score(y_te, ypr),
    }

# ── Print final comparison ─────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'FINAL RESULTS: Train-Test Split vs 5-Fold Stratified CV':^80}")
print("="*80)

header = f"{'Model':<22} {'Metric':<12} {'Split':>8}   {'K-Fold Mean':>12} {'±Std':>8}"
print(header)
print("-"*80)

for model_name in models:
    first = True
    for met in metrics:
        split_val  = split_results[model_name][met]
        kfold_mean = kfold_summary[model_name][f'{met}_mean']
        kfold_std  = kfold_summary[model_name][f'{met}_std']
        label = model_name if first else ""
        print(f"{label:<22} {met:<12} {split_val:>8.4f}   {kfold_mean:>12.4f} {kfold_std:>8.4f}")
        first = False
    print()

# ── Compact table ──────────────────────────────────────────────────────────────
print("="*80)
print("ACCURACY SUMMARY")
print("="*80)
print(f"{'Model':<22} {'Split Acc':>10} {'KFold Acc Mean':>16} {'±Std':>8}")
print("-"*60)
for model_name in models:
    print(f"{model_name:<22} {split_results[model_name]['accuracy']:>10.4f} "
          f"{kfold_summary[model_name]['accuracy_mean']:>16.4f} "
          f"{kfold_summary[model_name]['accuracy_std']:>8.4f}")
