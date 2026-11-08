"""
Build digital-twin artifacts from existing project CSVs (read-only).

Writes only under digital_twin/artifacts/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]  # Final_Gun_Violence
DT_ROOT = Path(__file__).resolve().parents[1]
ART = DT_ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURE_GROUPS = {
    "POI / facilities": {
        "tag": "actionable",
        "features": [
            "Park",
            "University",
            "Police_Building",
            "Place_of_Worship",
            "Parks",
            "Liquor_Stores",
            "Hospital",
            "Fire_Station",
            "Community_Center",
            "Schools",
        ],
    },
    "Socioeconomic": {
        "tag": "correlational",
        "features": [
            "Food_Stamp_SNAP",
            "Unemployment_Rate",
            "Vacant_Housing_Units",
            "Average_Commute_To_Work",
            "Working_Age_Economic_Dependency_Ratio",
            "food_X2001_X",
            "Poverty_Rate_Pct",
        ],
    },
    "Demographic / education": {
        "tag": "correlational",
        "features": [
            "Black_Segregation_Index",
            "Black_Percentage",
            "Education_Less_Than_9th_Pct",
            "Education_HS_Diploma_Pct",
            "Education_Bachelors_Pct",
        ],
    },
}


def main() -> None:
    vif_path = ROOT / "Gun_Violence_VIF_Clean.csv"
    master_path = ROOT / "Master_Hexagon_Gun_Violence_Summary_Table.csv"
    pred_path = ROOT / "Gun_Violence_Predictions.csv"

    vif = pd.read_csv(vif_path)
    cities = pd.read_csv(master_path, usecols=["GRID_ID", "City_name"])
    df = vif.merge(cities, on="GRID_ID", how="left")

    if pred_path.exists():
        preds = pd.read_csv(
            pred_path,
            usecols=["GRID_ID", "Predicted_Probability_Best_Model", "Hotspot_Flag", "Risk_Score"],
        )
        df = df.merge(preds, on="GRID_ID", how="left")

    feature_cols = [c for c in vif.columns if c not in ("GRID_ID", "Total_Gun_Violence")]
    X = df[feature_cols]
    y = (df["Total_Gun_Violence"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model": "LightGBM",
        "test_f1_at_0_50": float(f1_score(y_test, pred)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_cols),
    }

    # Full-table twin probabilities (this app's model)
    df["twin_probability"] = model.predict_proba(X)[:, 1]

    # Slider metadata
    slider_meta = {}
    for col in feature_cols:
        series = df[col].astype(float)
        lo = float(series.min())
        hi = float(series.max())
        # Ensure non-zero range for UI
        if hi <= lo:
            hi = lo + 1e-6
        slider_meta[col] = {
            "min": lo,
            "max": hi,
            "step": float(max((hi - lo) / 200.0, 1e-4)),
            "mean": float(series.mean()),
        }

    # Flatten groups with only existing features
    groups_out = []
    used = set()
    for name, spec in FEATURE_GROUPS.items():
        feats = [f for f in spec["features"] if f in feature_cols]
        used.update(feats)
        groups_out.append({"name": name, "tag": spec["tag"], "features": feats})
    leftover = [f for f in feature_cols if f not in used]
    if leftover:
        groups_out.append({"name": "Other", "tag": "correlational", "features": leftover})

    actionable = []
    for g in groups_out:
        if g["tag"] == "actionable":
            actionable.extend(g["features"])

    hex_out = df[
        [
            "GRID_ID",
            "City_name",
            "Total_Gun_Violence",
            "twin_probability",
            *feature_cols,
        ]
        + ([c for c in ["Predicted_Probability_Best_Model", "Hotspot_Flag", "Risk_Score"] if c in df.columns])
    ].copy()

    joblib.dump(model, ART / "model.joblib")
    hex_out.to_csv(ART / "hex_features.csv", index=False)

    schema = {
        "feature_columns": feature_cols,
        "feature_groups": groups_out,
        "actionable_features": actionable,
        "slider_meta": slider_meta,
        "target": "Total_Gun_Violence > 0",
        "metrics": metrics,
        "source_files": {
            "vif_clean": str(vif_path.name),
            "master": str(master_path.name),
        },
    }
    (ART / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Wrote artifacts to", ART)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
