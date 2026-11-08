# Study Atlas — Gun Violence Digital Twin

A local research workspace using the actual 35-city project dataset and the existing saved LightGBM model. It includes 11,448 hexagons, 22 verified model features, and 2,618 recorded incidents.

## Start

Double-click **Start Digital Twin.cmd** in this folder. Open http://127.0.0.1:8765. Keep the launch window open; Ctrl+C stops the service. If the port is already running, use that existing browser page.

From the project directory, the equivalent command is:

```powershell
.\venv\Scripts\python.exe -B -m uvicorn digital_twin.app.main:app --host 127.0.0.1 --port 8765
```

No installation is needed with the current project runtime. The launcher reads the existing environment without changing it and disables bytecode writes. It prefers `digital_twin/.venv` if a dedicated twin environment exists.

For a fresh setup, create a dedicated environment **inside digital_twin**, then install `requirements-lock.txt` into that environment. Do not install into or alter the original project venv. The lock file records the versions used for validation; the existing saved model was verified with this runtime.

## Workflow

1. Choose any study city. Its entire set of locations loads, including all 1,136 Houston hexagons. The location list uses 60-row pages and literal GRID_ID search.
2. Select a hexagon. Review its recorded incidents, population, baseline score, and training/holdout membership.
3. Change any of the 22 inputs in original units. Facility counts are whole numbers; continuous measures accept decimals within realistic study ranges. Undo and Reset restore assumptions without editing source data.
4. Compare the baseline with the scenario. The scenario map layer changes only the selected hexagon; other locations retain baseline scores.
5. Explore a one-feature response curve, save a named scenario, or export its JSON snapshot.
6. Use Scenario library to compare saved assumptions, reopen a copy, export, or delete a snapshot. Saved snapshots contain model/data versions; incompatible versions cannot silently reopen.
7. Original data lists all 22 original values with definitions and quality flags. Optimize tests bounded low/high values and reports the lowest model score found. Model drivers shows baseline TreeSHAP contributions in log-odds.
8. Data & evidence documents the target, evaluation, calibration, regional holdouts, quality issues, and source fingerprints.

## Isolation

All application writes stay under **digital_twin/**. The original notebooks, root CSVs, tests folder, and project environment are read-only inputs. The application uses an internal SQLite database at `artifacts/scenarios.sqlite3`. No external database or account is required. Back up that file when the server is stopped to preserve scenarios.

The browser requests Leaflet assets from unpkg.com and background map tiles from OpenStreetMap. Data scoring and scenario storage run locally. If external map assets cannot load, the location list, original data, scoring, and saved scenarios remain usable. There are no external fonts, analytics, or model APIs.

## Data fidelity

`scripts/prepare_workspace.py` reconstructs the five derived percentage features using the original notebook formulas, verifies all 22 original-to-scaled transformations (tolerance 1e-9), checks unique and matching identifiers, confirms city/outcome agreement, and checks all 11,448 saved baseline predictions against the existing model. It does not retrain the model.

`scripts/build_model_registry.py` reproduces the original seed-42 split and trains Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM from the project VIF-clean feature table. It writes model-specific artifacts and a five-model score table under `artifacts/`. The browser lets you select a model, compare all five baseline outputs, and rescore scenarios with the selected model. LightGBM remains the default because it has the strongest random-hex F1 in the supplied comparison; that is a model-selection convenience, not a claim that it is universally best.

```powershell
.\venv\Scripts\python.exe -B digital_twin\scripts\prepare_workspace.py
```

Restart after preparing artifacts. Startup fingerprints reject mismatched model, feature, original-value, or center artifacts. Original baseline vectors are preserved exactly; re-dividing floating-point values at a tree split can otherwise introduce tiny prediction changes. Facility values already observed in the source use their exact serialized model representation.

`build_artifacts.py` remains an explicit retraining utility. Do not run it just to launch the app. If intentionally rebuilding, regenerate centers and run `prepare_workspace.py` afterward; old saved scenarios remain historical snapshots.

## Geometry

The supplied project directory contains no authoritative GIS boundaries. The previous generated hexagon polygons are not displayed. The map shows **estimated point locations** and labels that limitation; neither neighborhood placement nor spatial adjacency is verified.

To add original geometry, place a GeoJSON FeatureCollection at:

`digital_twin/artifacts/authoritative_hexes.geojson`

Each feature must have a unique `properties.GRID_ID` matching the study and a Polygon or MultiPolygon in WGS84 longitude/latitude (EPSG:4326). Do not merely relabel coordinates from another projection; export/reproject them correctly. Restart the server. Supplied polygons replace estimated points for matching identifiers; missing IDs retain an explicit approximation label. This file is never written by the application.

The legacy geometry builder's circumradius formula is corrected, but its fitted locations are still estimates. Its historical `hex_polygons.geojson` output is unused by this interface.

## Scientific interpretation

- The target is `Total_Gun_Violence > 0`, a binary indicator over the supplied observations.
- Observation dates are absent from these tables. A folder name is not treated as an established observation period.
- Displayed scores equal 100 times the uncalibrated model output; they are not future probabilities, incident counts, or individual risk estimates.
- Facility changes are model-sensitivity experiments, not causal intervention effects. No cost, implementation time, or neighborhood spillover is modeled.
- Demographic and socioeconomic variables are now testable scenario inputs with explicit boundaries, but they remain contextual measures; changing them is not a causal recommendation. Park/Parks and unclear source measures keep their source labels.
- The original pipeline encodes undefined percentage denominators as zero. There are 2,792 zero-population rows and 17 derived poverty percentages above 100%; source values are preserved and flagged.
- Baseline maps include training rows as well as held-out rows. The inspector labels this distinction.
- The existing random-hex evaluation is reconstructed, with preprocessing inherited from the original full-data pipeline. It is not an independent prospective evaluation. The calibration plot is diagnostic; it does not claim to calibrate the saved model.
- City and East-to-West holdout results are read from the existing project comparison and are context for generalization, not evaluations of this deployed artifact.

## Verification

With the app running on port 8765:

```powershell
.\venv\Scripts\python.exe -B digital_twin\tests\test_workspace.py
```

The suite checks original-unit mappings, all baseline predictions, explanation additivity, full-city pagination, data flags, invalid inputs, local-origin protection, scenario persistence, and protected research-file fingerprints. Temporary test database files stay under `digital_twin/tests`.

`artifacts/protected_files.json` records the 23 research-file SHA-256 hashes captured before implementation. The runtime environment was not modified. `VALIDATION.md` records the implementation checks and limitations.

## Structure

- `app/`: FastAPI service, input validation, inference, geometry loading, and SQLite storage.
- `frontend/`: responsive browser workspace; no frontend build/install step.
- `artifacts/`: model/data snapshots, provenance, and local scenario database.
- `scripts/prepare_workspace.py`: original-unit recovery and fidelity checks.
- `scripts/build_model_registry.py`: five-model registry and baseline score generation.
- `tests/test_workspace.py`: integration and scientific-data regression tests.

This is a local single-user research application. Keep the server bound to 127.0.0.1. It is not configured as a public multi-user service.
