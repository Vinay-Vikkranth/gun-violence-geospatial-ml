"""Train the five models used by the original project and write twin artifacts only."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'digital_twin' / 'artifacts'
SEED = 42

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    data = pd.read_csv(ROOT / 'Gun_Violence_VIF_Clean.csv')
    features = [c for c in data.columns if c not in ('GRID_ID', 'Total_Gun_Violence')]
    X = data[features].astype(float)
    y = (data['Total_Gun_Violence'] > 0).astype(int)
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=.20, random_state=SEED, stratify=y)
    positive_weight = float((y.iloc[train_idx] == 0).sum() / max((y.iloc[train_idx] == 1).sum(), 1))
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, class_weight='balanced', n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=.1, random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=.1, random_state=SEED, scale_pos_weight=positive_weight, eval_metric='logloss', n_jobs=1),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=.1, random_state=SEED, class_weight='balanced', verbose=-1, n_jobs=1),
    }
    registry = {'target': 'Total_Gun_Violence > 0', 'features': features, 'seed': SEED, 'models': {}, 'sources': {}}
    scores = pd.DataFrame({'GRID_ID': data.GRID_ID, 'Total_Gun_Violence': data.Total_Gun_Violence})
    for name, model in models.items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = model.predict_proba(X)[:, 1]
        holdout = probability[test_idx]
        prediction = (holdout >= .5).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y.iloc[test_idx], prediction)),
            'precision': float(precision_score(y.iloc[test_idx], prediction, zero_division=0)),
            'recall': float(recall_score(y.iloc[test_idx], prediction, zero_division=0)),
            'f1': float(f1_score(y.iloc[test_idx], prediction, zero_division=0)),
            'roc_auc': float(roc_auc_score(y.iloc[test_idx], holdout)),
            'average_precision': float(average_precision_score(y.iloc[test_idx], holdout)),
            'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
        }
        key = name.lower().replace(' ', '_')
        model_path = ART / f'model_{key}.joblib'
        joblib.dump(model, model_path)
        registry['models'][key] = {'name': name, 'artifact': model_path.name, 'metrics': metrics}
        scores[key] = probability
    scores.to_csv(ART / 'model_scores.csv', index=False)
    registry['sources'] = {
        'vif_clean': {'file': 'Gun_Violence_VIF_Clean.csv', 'sha256': sha(ROOT / 'Gun_Violence_VIF_Clean.csv')},
        'model_scores': {'file': 'model_scores.csv', 'sha256': sha(ART / 'model_scores.csv')},
    }
    (ART / 'model_registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in registry['models'].items()}, indent=2))

if __name__ == '__main__':
    main()
