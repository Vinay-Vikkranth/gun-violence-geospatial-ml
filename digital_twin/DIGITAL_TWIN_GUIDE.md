# Digital Twin guide

## 1. What the map shows
- The map covers the GIS study area: 35 cities and 11,448 study hexagons.
- City records are shown as cards until original GIS polygons are available. Card order and spacing have no spatial meaning. The old inferred positions could place records offshore; they are no longer plotted.
- Score accents are colored from yellow/orange (lower score) to dark red (higher score).
- List and overview colors use LightGBM, explicitly labeled. The inspector uses the selected model. Each score is model output multiplied by 100. It is a comparison score, not a confirmed probability of a future shooting.

## 2. Baseline score
- Baseline means the model receives the original recorded values for that hexagon.
- Five trained models are available: LightGBM, XGBoost, Random Forest, Gradient Boosting, and Logistic Regression.
- The model selector recalculates the baseline and scenario score using the selected saved model.

## 3. Scenario controls
- All 22 project inputs can be tested.
- Facility counts use whole numbers. Context, demographic, and socioeconomic inputs use decimal values when the source measure is continuous.
- Every variable has an explicit boundary_min and boundary_max in the workspace manifest. The lower bound is the study-wide 5th percentile; when zero inflation makes that zero but positive values exist, it uses the positive-support 5th percentile. The upper bound is the study-wide 95th percentile (rounded up for facility counts). Manual controls extend the interval to include an original baseline outside these bounds; the displayed Allowed interval reflects this. The optimizer tests the narrower empirical boundaries. These are study-distribution limits, not expert-validated feasibility limits.
- Changing a value runs the selected model again with only that input changed.

## 4. Optimize
- Open a hexagon, choose a model, and select **Optimize**.
- The optimizer tests the allowed low and high value for each input, keeps changes that lower the score, and repeats until no remaining single input lowers it.
- Search starts from the original baseline, replacing any current edits; Undo restores those edits. It tests endpoints greedily, not every combination or interior value, so it does not guarantee the global minimum.
- The result lists every change, the before/after values, and the lowest score found.
- This is bounded model minimization. It is useful for sensitivity analysis, but it is not a causal intervention plan and does not include cost, timing, feasibility, or spillover effects.

## 5. Model drivers
- Model drivers are LightGBM-only TreeSHAP contributions for the baseline row.
- A positive contribution pushes the model output higher; a negative contribution pushes it lower.
- They explain the prediction, not what would happen if a policy changed a variable.

## 6. Original data and quality
- Original data shows the preserved source values and field notes.
- Warnings identify source conditions such as zero population denominators or poverty percentages above 100%.
- No original project files are edited. Local scenarios are stored in `digital_twin/artifacts/scenarios.sqlite3`. On the hosted site they are stored in the current browser. Export JSON to share or back up them.

## 7. Reproducibility
- The app uses the project’s reconstructed original-unit table, exact model artifacts, and model registry inside `digital_twin/artifacts`.
- No API key is required. The app runs locally at `http://127.0.0.1:8765`.

## 8. Professor review fixes — October 1
- Choosing a city opens a record and its Edit variables tab automatically. Expand topic headings to find all 22 inputs.
- Reset, response plots, and reopening saved scenarios preserve the selected scoring model.
- Invalid inputs disable scoring, saving, exporting, and optimization until corrected.
- Changing inputs or the model invalidates an in-progress optimizer result. Optimized exports include the actual optimized values.
- Study colors use yellow/orange to dark red on a neutral background.
- The screenshot does not identify the exact scenario-planning bug; the reproducible issues above were corrected. The request to finish the paper is separate from the website.
