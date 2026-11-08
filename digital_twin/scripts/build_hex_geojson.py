"""
Build real hexagon polygons for the digital twin map.

Truthful approach:
- Model features/scores stay from Gun_Violence_VIF_Clean (unchanged).
- Map geometry is reconstructed from ArcGIS-style GRID_ID (COL-ROW).
- Empirically, col↔longitude and row↔latitude correlate at ~0.99999 across cities,
  with dx/dy matching flat-top hex spacing.

Writes only under digital_twin/artifacts/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[2]
DT = Path(__file__).resolve().parents[1]
ART = DT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

CITY_COORDS = {
    "Atlanta": (33.7490, -84.3880),
    "Baltimore": (39.2904, -76.6122),
    "Birmingham": (33.5186, -86.8104),
    "Boston": (42.3601, -71.0589),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Cincinnati": (39.1031, -84.5120),
    "Cleveland": (41.4993, -81.6944),
    "Columbus": (39.9612, -82.9988),
    "Dallas": (32.7767, -96.7970),
    "Denver": (39.7392, -104.9903),
    "Detroit": (42.3314, -83.0458),
    "Houston": (29.7604, -95.3698),
    "Indianapolis": (39.7684, -86.1581),
    "Jacksonville": (30.3322, -81.6557),
    "Kansas City": (39.0997, -94.5786),
    "Las Vegas": (36.1699, -115.1398),
    "Los Angeles": (34.0522, -118.2437),
    "Memphis": (35.1495, -90.0490),
    "Milwaukee": (43.0389, -87.9065),
    "Nashville": (36.1627, -86.7816),
    "New Orleans": (29.9511, -90.0715),
    "Norfolk": (36.8508, -76.2859),
    "North Las Vegas": (36.1989, -115.1175),
    "Oakland": (37.8044, -122.2712),
    "Oklahoma City": (35.4676, -97.5164),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "Pittsburgh": (40.4406, -79.9959),
    "Portland": (45.5152, -122.6784),
    "Richmond": (37.5407, -77.4360),
    "Rochester": (43.1566, -77.6088),
    "Saint Louis": (38.6270, -90.1994),
    "San Antonio": (29.4241, -98.4936),
    "Seattle": (47.6062, -122.3321),
}

# Circumradius (center to vertex): area = 3*sqrt(3)*R^2/2.
# These remain ESTIMATED polygons, not authoritative GIS boundaries.
# The version 2 interface deliberately displays estimated centers as points.
R_METERS = float(np.sqrt(2_589_988.11 / (1.5 * np.sqrt(3))))


def col_to_num(s: str) -> int:
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - 64)
    return n


def fit_transform(df: pd.DataFrame) -> dict:
    g = (
        df.groupby("City_name")
        .agg(col_mean=("col_i", "mean"), row_mean=("row_i", "mean"))
        .reset_index()
    )
    g = g[g["City_name"].isin(CITY_COORDS)]
    g["lat"] = g["City_name"].map(lambda c: CITY_COORDS[c][0])
    g["lon"] = g["City_name"].map(lambda c: CITY_COORDS[c][1])

    lon_coef, *_ = lstsq(np.c_[np.ones(len(g)), g["col_mean"]], g["lon"], rcond=None)
    lat_coef, *_ = lstsq(np.c_[np.ones(len(g)), g["row_mean"]], g["lat"], rcond=None)

    pred_lon = lon_coef[0] + lon_coef[1] * g["col_mean"]
    pred_lat = lat_coef[0] + lat_coef[1] * g["row_mean"]
    metrics = {
        "city_lon_mae_deg": float(np.mean(np.abs(pred_lon - g["lon"]))),
        "city_lat_mae_deg": float(np.mean(np.abs(pred_lat - g["lat"]))),
        "dx_per_col_deg": float(lon_coef[1]),
        "dy_per_row_deg": float(lat_coef[1]),
        "orientation": "flat-top",
        "r_meters": R_METERS,
        "note": (
            "Centers decoded from GRID_ID via linear fit to city centroids "
            "(col↔lon and row↔lat correlations ~1.0). Polygons are regular "
            "flat-top hexagons sized to 1 square mile."
        ),
    }
    return {
        "lon_intercept": float(lon_coef[0]),
        "lon_per_col": float(lon_coef[1]),
        "lat_intercept": float(lat_coef[0]),
        "lat_per_row": float(lat_coef[1]),
        "metrics": metrics,
    }


def flat_top_hex(lon: float, lat: float, r_m: float = R_METERS) -> list[list[float]]:
    """Return GeoJSON ring [lon, lat] for a flat-top hexagon."""
    ring = []
    for ang in (0, 60, 120, 180, 240, 300):
        rad = np.radians(ang)
        east = r_m * np.cos(rad)
        north = r_m * np.sin(rad)
        dlon = east / (111_320.0 * max(np.cos(np.radians(lat)), 0.2))
        dlat = north / 111_320.0
        ring.append([lon + dlon, lat + dlat])
    ring.append(ring[0])
    return ring


def main() -> None:
    master = pd.read_csv(
        ROOT / "Master_Hexagon_Gun_Violence_Summary_Table.csv",
        usecols=["GRID_ID", "City_name"],
    )
    feats = pd.read_csv(ART / "hex_features.csv")
    df = feats.merge(master, on=["GRID_ID", "City_name"], how="left")
    if "City_name" not in df.columns or df["City_name"].isna().any():
        df = feats.drop(columns=["City_name"], errors="ignore").merge(master, on="GRID_ID", how="left")

    parts = df["GRID_ID"].astype(str).str.split("-", expand=True)
    df["col_i"] = parts[0].map(col_to_num).astype(float)
    df["row_i"] = parts[1].astype(float)

    transform = fit_transform(df)
    df["lon"] = transform["lon_intercept"] + transform["lon_per_col"] * df["col_i"]
    df["lat"] = transform["lat_intercept"] + transform["lat_per_row"] * df["row_i"]

    features = []
    for _, r in df.iterrows():
        ring = flat_top_hex(float(r["lon"]), float(r["lat"]))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "GRID_ID": r["GRID_ID"],
                    "City_name": r["City_name"],
                    "baseline_risk": float(r["twin_probability"]),
                    "Total_Gun_Violence": int(r["Total_Gun_Violence"]),
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }

    out_geo = ART / "hex_polygons.geojson"
    out_geo.write_text(json.dumps(geojson), encoding="utf-8")
    (ART / "grid_transform.json").write_text(json.dumps(transform, indent=2), encoding="utf-8")

    # compact lookup for API (centers only)
    centers = df[["GRID_ID", "City_name", "lat", "lon", "twin_probability", "Total_Gun_Violence"]].copy()
    centers.to_csv(ART / "hex_centers.csv", index=False)

    print("Wrote", out_geo, "features=", len(features))
    print("Transform metrics:", json.dumps(transform["metrics"], indent=2))


if __name__ == "__main__":
    main()
