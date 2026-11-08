# Reproduction and application setup

## Public sample

requirements.txt contains lightweight dependencies for scripts/demo.py and tests. No data download or model serialization is required. Three models are freshly fitted to synthetic data. Their metrics are not research performance claims.

## Complete research

Use scripts/prepare_research.py with the ignored local data/private copies, then work from outputs/research_workspace. GitHub-only clones need independently authorized source inputs first. The preparation script copies the source notebooks and required CSVs into a separate writable workspace.

The research stack additionally needs seaborn, XGBoost, LightGBM, SHAP, joblib, Jupyter and compatible versions of scikit-learn. See requirements-research.txt. The core table is Master_Hexagon_Gun_Violence_Summary_Table.csv; run the core notebook before cross-validation if rebuilding derived tables. Historical full-data preprocessing and threshold-selection choices are retained for provenance, not endorsed as leakage-free evaluation.

## Study Atlas

The public repository retains the complete app source, frontend, builders and original documentation. To reconstruct artifacts, place the authorized project CSVs beside digital_twin in an isolated workspace, install digital_twin/requirements-lock.txt, then run its build_artifacts.py, build_hex_geojson.py, build_model_registry.py and prepare_workspace.py as documented in digital_twin/README.md. These builders can be expensive and need all required source inputs; do not deserialize untrusted model files. The packaging checks do not claim a fresh full model-registry build or a functioning public-hosted application.

After artifact preparation, launch from the workspace root with:

```bash
python -m uvicorn digital_twin.app.main:app --host 127.0.0.1 --port 8765
```

Run digital_twin/tests/test_workspace.py against that local server only after preparing the matching artifacts. It is not the public synthetic test suite. Geometry is estimated unless authoritative GRID_ID-matched polygons are supplied. No cloud service is provisioned by this repository.

## Local verification copy

The Desktop workspace at outputs/research_workspace additionally contains private local copies of the existing five model artifacts. These are not on GitHub. They were loaded using the original matching read-only Python runtime with bytecode writes disabled, and health plus baseline scoring passed. Preparing that workspace again copies source but preserves its local artifacts. This confirms existing-model loading, not fresh artifact-builder reproduction.
