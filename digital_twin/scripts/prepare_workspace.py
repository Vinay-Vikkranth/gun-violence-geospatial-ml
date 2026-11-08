"""Read project data, verify original-unit transforms, write only twin artifacts."""
from pathlib import Path
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, precision_score, recall_score, average_precision_score
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'digital_twin' / 'artifacts'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    schema = json.loads((ART / 'feature_schema.json').read_text())
    cols = schema['feature_columns']
    master = pd.read_csv(ROOT / 'Master_Hexagon_Gun_Violence_Summary_Table.csv').set_index('GRID_ID')
    vif = pd.read_csv(ROOT / 'Gun_Violence_VIF_Clean.csv').set_index('GRID_ID')
    saved = pd.read_csv(ART / 'hex_features.csv').set_index('GRID_ID')
    if not master.index.is_unique or not vif.index.is_unique or not saved.index.is_unique:
        raise ValueError('Source and artifact GRID_ID values must be unique')
    if set(master.index) != set(vif.index) or set(saved.index) != set(vif.index):
        raise ValueError('Source and artifact GRID_ID sets differ')
    raw = master.copy()
    derived = {
        'Black_Percentage': ('Black_Population', 'Total_Population'),
        'Poverty_Rate_Pct': ('Population_Below_Poverty_Level', 'Total_Population'),
        'Education_Less_Than_9th_Pct': ('Education_Less_Than_Ninth_Grade', 'Total_Educational_Attainment_Base'),
        'Education_HS_Diploma_Pct': ('Education_Max_Have_High_School_Diploma', 'Total_Educational_Attainment_Base'),
        'Education_Bachelors_Pct': ('Education_Have_Bachelors_Degree', 'Total_Educational_Attainment_Base'),
    }
    for name, (num, den) in derived.items():
        raw[name] = (master[num] / master[den].replace(0, np.nan) * 100).fillna(0)
    labels = {
        'Black_Segregation_Index': 'Black segregation index', 'Food_Stamp_SNAP': 'SNAP source measure',
        'Park': 'Park · source field', 'Parks': 'Parks · source field', 'University': 'Universities',
        'Police_Building': 'Police buildings', 'Place_of_Worship': 'Places of worship',
        'Liquor_Stores': 'Liquor stores', 'Hospital': 'Hospitals', 'Fire_Station': 'Fire stations',
        'Community_Center': 'Community centers', 'Schools': 'Schools',
        'Unemployment_Rate': 'Unemployment rate', 'Vacant_Housing_Units': 'Vacant housing units',
        'Average_Commute_To_Work': 'Commute · source measure',
        'Working_Age_Economic_Dependency_Ratio': 'Working-age dependency ratio',
        'food_X2001_X': 'Food · source measure', 'Black_Percentage': 'Black population share',
        'Poverty_Rate_Pct': 'Derived poverty percentage', 'Education_Less_Than_9th_Pct': 'Education: below ninth grade',
        'Education_HS_Diploma_Pct': 'Education: high-school diploma',
        'Education_Bachelors_Pct': 'Education: bachelor’s degree',
    }
    facility_features = {'Park','University','Police_Building','Place_of_Worship','Parks','Liquor_Stores','Hospital','Fire_Station','Community_Center','Schools'}
    groups = {f:g['name'] for g in schema['feature_groups'] for f in g['features']}
    metadata = {}
    for col in cols:
        s = raw[col].astype(float)
        lo, hi = float(s.min()), float(s.max())
        transformed = (s - lo) / (hi - lo) if hi > lo else s * 0
        error = float(np.max(np.abs(transformed.loc[vif.index] - vif[col])))
        if not np.isfinite(s).all() or error > 1e-9:
            raise ValueError(f'Original-unit mapping failed for {col}: {error}')
        if not np.allclose(saved.loc[vif.index, col], vif[col], atol=1e-12, rtol=0):
            raise ValueError(f'Twin feature differs from source: {col}')
        unit = 'recorded count' if col in facility_features else ('%' if col in derived or col == 'Unemployment_Rate' else 'source units')
        note = 'Original master-table value; its precise unit is not documented in the project.'
        if col in facility_features:
            note = 'Integer facility count recorded in the master table. Scenario changes test model sensitivity; they do not estimate intervention effects.'
        if col in derived:
            num, den = derived[col]
            note = f'100 × {num} / {den}. The original pipeline encodes a zero denominator as 0; these rows are flagged separately.'
        if col in {'Park', 'Parks'}:
            note = 'Park and Parks are separate source fields. Their relationship is undocumented, so both remain read-only.'
        q95 = float(s.quantile(.95))
        q05 = float(s.quantile(.05))
        positive = s[s > 0]
        positive_floor = float(positive.quantile(.05)) if len(positive) else 0.0
        # Scenario boundaries use the central 5th–95th percentile range. When
        # the 5th percentile is zero but the variable has a positive support,
        # use the positive-support 5th percentile to avoid a degenerate
        # all-zero optimizer result. The original baseline remains viewable.
        boundary_min = positive_floor if q05 <= 0 and q95 > 0 else q05
        boundary_max = max(1.0, float(np.ceil(q95))) if col in facility_features else q95
        scenario_min = boundary_min
        scenario_max = boundary_max
        metadata[col] = dict(label=labels[col], unit=unit, min=lo, max=hi, q05=q05, q95=q95,
                             boundary_min=boundary_min, boundary_max=boundary_max,
                             boundary_method='positive-support 5th percentile to study-wide 95th percentile',
                             scenario_min=scenario_min, scenario_max=scenario_max, step=1 if col in facility_features else max((scenario_max-scenario_min)/100, 1e-4),
                             editable=True, facility=col in facility_features, group=groups[col], note=note, transform_max_error=error)
    np.testing.assert_allclose(saved.loc[vif.index,'Total_Gun_Violence'], vif.Total_Gun_Violence)
    np.testing.assert_allclose(master.loc[vif.index,'Total_Gun_Violence'], vif.Total_Gun_Violence)
    if not (saved.loc[vif.index,'City_name'] == master.loc[vif.index,'City_name']).all():
        raise ValueError('City assignments differ')
    model = joblib.load(ART / 'model.joblib')
    model.set_params(n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        probs = model.predict_proba(vif[cols])[:,1]
    np.testing.assert_allclose(probs, saved.loc[vif.index,'twin_probability'], atol=1e-10, rtol=0)
    y = (vif.Total_Gun_Violence > 0).astype(int)
    train, test = train_test_split(np.arange(len(vif)), test_size=.2, random_state=42, stratify=y)
    yp, yt = probs[test], y.iloc[test]
    frac, mean = calibration_curve(yt, yp, n_bins=8, strategy='quantile')
    evaluation = dict(schema['metrics'], precision=float(precision_score(yt, yp >= .5)), recall=float(recall_score(yt, yp >= .5)),
                      average_precision=float(average_precision_score(yt, yp)), brier=float(brier_score_loss(yt, yp)), prevalence=float(yt.mean()),
                      calibration=[dict(mean_score=float(a), observed_fraction=float(b)) for a,b in zip(mean,frac)],
                      note='Reconstructed random-hex holdout from the existing builder (seed 42). Preprocessing was fitted before splitting in the original project. This is an inherited evaluation, not independent prospective validation. Scores are uncalibrated.')
    raw_out = raw.loc[vif.index, cols + ['City_name','Total_Gun_Violence','Total_Population','Total_Educational_Attainment_Base']].copy()
    raw_out['evaluation_split'] = 'train'
    raw_out.iloc[test, raw_out.columns.get_loc('evaluation_split')] = 'holdout'
    raw_out.to_csv(ART / 'original_values.csv')
    sources = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in [ROOT/'Master_Hexagon_Gun_Violence_Summary_Table.csv',ROOT/'Gun_Violence_VIF_Clean.csv',ART/'model.joblib',ART/'hex_features.csv',ART/'original_values.csv',ART/'hex_centers.csv']}
    registry_path = ART / 'model_registry.json'
    if registry_path.exists():
        sources['model_registry.json'] = dict(sha256=sha(registry_path), bytes=registry_path.stat().st_size)
    scores_path = ART / 'model_scores.csv'
    if scores_path.exists():
        sources['model_scores.csv'] = dict(sha256=sha(scores_path), bytes=scores_path.stat().st_size)
    manifest = dict(features=metadata, evaluation=evaluation, sources=sources, model_version=sha(ART/'model.joblib')[:12],
                    data_version=sha(ROOT/'Master_Hexagon_Gun_Violence_Summary_Table.csv')[:12],
                    period='Observation dates are not recorded in the supplied tables; this is not a future forecast.',
                    n_hexes=len(vif), n_cities=int(master.City_name.nunique()),
                    total_incidents=int(master.Total_Gun_Violence.sum()),
                    quality=dict(zero_population=int((raw.Total_Population==0).sum()), poverty_above_100=int((raw.Poverty_Rate_Pct>100).sum())),
                    regional_evidence=[dict(split='Random hex holdout',model='LightGBM',f1=.5247524752475248),dict(split='City holdout',model='LightGBM',f1=.6135),dict(split='East → West',model='LightGBM',f1=.3982)],
                    model_registry='model_registry.json' if registry_path.exists() else None)
    (ART/'workspace.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['n_hexes','n_cities','total_incidents','quality','model_version']}))

if __name__ == '__main__':
    main()
