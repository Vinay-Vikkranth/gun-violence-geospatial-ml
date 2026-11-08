# Three-Way Holdout Comparison

This document compares **three evaluation designs** for the same gun-violence classification task.

All three use:

- the same master data (`Master_Hexagon_Gun_Violence_Summary_Table.csv`)
- the same target: **`Total_Gun_Violence > 0`**
- the same model family: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- the same style of preprocessing (feature drops, percentage features, Min-Max scaling, VIF / correlation cleanup)

They differ **only in how train and test hexes are chosen**.

---

## 1. How the three splits differ

| | **1. Random hexagon 80/20** | **2. City holdout 80/20** | **3. East → West coast ~80/20** |
|--|---------------------------|---------------------------|--------------------------------|
| **Location** | Main pipeline (`../../Gun_Violence_ML_Pipeline.executed.ipynb`) | `../city_holdout_80_20/` | `../coast_holdout_east_west/` |
| **Split unit** | Individual hexagons | Whole cities | Whole cities by region |
| **Same city in train & test?** | Yes | **No** | **No** |
| **What it measures** | Hex-level generalization (easier) | City-level generalization | Cross-coast / regional generalization (hardest) |

### Split sizes

| Experiment | Train hexes | Test hexes | Train / test cities |
|------------|------------:|-----------:|---------------------|
| 1. Random hexagon | 9,158 (80.0%) | 2,290 (20.0%) | Hexes mixed across all 35 cities |
| 2. City holdout | 9,528 (83.2%) | 1,920 (16.8%) | 28 train / 7 test |
| 3. East → West | 9,133 (79.8%) | 2,315 (20.2%) | 27 train / 8 test |

---

## 2. How hexes / cities were selected (method)

Important shared rule for experiments **2** and **3**:  
assignment is by **city**, not by individual hex. Once a city is assigned to train or test, **all of that city’s hexes** go with it. No city appears in both sets.

### Experiment 1 — Random hexagon 80/20

| Step | Method |
|------|--------|
| 1 | After preprocessing, treat each hexagon as one sample |
| 2 | Use scikit-learn `train_test_split` with `test_size=0.20`, `random_state=42`, `stratify=y` |
| 3 | Hexes are drawn at random across **all cities**, so the same city can contribute hexes to both train and test |

**Selection unit:** individual hexagons (not cities).

### Experiment 2 — City holdout 80/20

| Step | Method |
|------|--------|
| 1 | Count hexes per city |
| 2 | Shuffle the list of cities with `numpy.RandomState(42)` |
| 3 | Walk that shuffled order and choose the prefix cut whose **cumulative hex count** is closest to **20%** of all hexes |
| 4 | Cities in that prefix → **test**; remaining cities → **train** |
| 5 | Assign **every hex** in a city to that city’s split |

**Selection unit:** whole cities. Hexes are not sampled inside a city; the city is all-in for train or all-in for test.

### Experiment 3 — East → West coast ~80/20

| Step | Method |
|------|--------|
| 1 | Define a fixed **West test** city list (Pacific West Coast + nearby western cities) |
| 2 | Assign all remaining cities to **East / non-West train** |
| 3 | Choose the West list so total West hexes are about **20%** of the dataset (Pacific-only cities are only ~10%, so Phoenix, Denver, Las Vegas, and North Las Vegas were added) |
| 4 | Assign **every hex** in each city to that city’s regional split |

**Selection unit:** whole cities by region (not a random shuffle).

---

## 3. Train / test city tables

### Experiment 2 — City holdout 80/20

#### Test cities (7 cities · 1,920 hexes · 16.8%)

| City | Hexes |
|------|------:|
| Indianapolis | 443 |
| Kansas City | 428 |
| New Orleans | 381 |
| Portland | 205 |
| Philadelphia | 190 |
| Milwaukee | 153 |
| Oakland | 120 |
| **Total** | **1,920** |

#### Train cities (28 cities · 9,528 hexes · 83.2%)

| City | Hexes | City | Hexes |
|------|------:|------|------:|
| Houston | 1,136 | Jacksonville | 882 |
| Oklahoma City | 729 | San Antonio | 622 |
| Los Angeles | 612 | Phoenix | 606 |
| Nashville | 574 | Dallas | 489 |
| Memphis | 395 | Charlotte | 372 |
| Columbus | 364 | Chicago | 320 |
| Birmingham | 285 | Denver | 234 |
| Seattle | 203 | Detroit | 195 |
| Las Vegas | 188 | Atlanta | 176 |
| North Las Vegas | 147 | Boston | 141 |
| Cleveland | 136 | Cincinnati | 135 |
| Norfolk | 123 | Baltimore | 114 |
| Pittsburgh | 94 | Richmond | 92 |
| Saint Louis | 90 | Rochester | 74 |
| **Total** | | | **9,528** |

Full assignment file: `../city_holdout_80_20/output/city_assignment.csv`

---

### Experiment 3 — East → West coast ~80/20

#### Test cities — West (8 cities · 2,315 hexes · 20.2%)

| City | Hexes | Why included |
|------|------:|--------------|
| Los Angeles | 612 | Pacific West Coast |
| Phoenix | 606 | Western US (to reach ~20%) |
| Denver | 234 | Western US (to reach ~20%) |
| Portland | 205 | Pacific West Coast |
| Seattle | 203 | Pacific West Coast |
| Las Vegas | 188 | Western US (to reach ~20%) |
| North Las Vegas | 147 | Western US (to reach ~20%) |
| Oakland | 120 | Pacific West Coast |
| **Total** | **2,315** | |

#### Train cities — East / non-West (27 cities · 9,133 hexes · 79.8%)

| City | Hexes | City | Hexes |
|------|------:|------|------:|
| Houston | 1,136 | Jacksonville | 882 |
| Oklahoma City | 729 | San Antonio | 622 |
| Nashville | 574 | Dallas | 489 |
| Indianapolis | 443 | Kansas City | 428 |
| Memphis | 395 | New Orleans | 381 |
| Charlotte | 372 | Columbus | 364 |
| Chicago | 320 | Birmingham | 285 |
| Detroit | 195 | Philadelphia | 190 |
| Atlanta | 176 | Milwaukee | 153 |
| Boston | 141 | Cleveland | 136 |
| Cincinnati | 135 | Norfolk | 123 |
| Baltimore | 114 | Pittsburgh | 94 |
| Richmond | 92 | Saint Louis | 90 |
| Rochester | 74 | | |
| **Total** | | | **9,133** |

Full assignment file: `../coast_holdout_east_west/output/city_assignment.csv`

> **Note:** Pure Pacific West Coast cities alone (LA, Oakland, Portland, Seattle) are only ~10% of hexes. Phoenix, Denver, Las Vegas, and North Las Vegas were added to the West **test** set so the hex split stays near **80/20**, while still testing on western cities after training on eastern / non-western cities.

---

## 4. Results side by side

### Best model summary

| Experiment | Best model (@ F1 0.50) | F1 @ 0.50 | Tuned threshold | F1 @ tuned | ROC-AUC (best @ 0.50) |
|------------|------------------------|----------:|----------------:|-----------:|----------------------:|
| 1. Random hexagon | LightGBM | **0.5248** | 0.62 | **0.5350** | 0.8743 |
| 2. City holdout | Random Forest | **0.6325** | 0.42 | **0.6555** | 0.9051 |
| 3. East → West | Random Forest | **0.3989** | 0.44 | **0.4084** | 0.8542 |

### Full metrics @ threshold 0.50

#### Experiment 1 — Random hexagon 80/20

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| LightGBM | 0.8323 | 0.4141 | 0.7162 | **0.5248** | 0.8743 |
| XGBoost | 0.8288 | 0.4062 | 0.7027 | 0.5149 | 0.8723 |
| Logistic Regression | 0.8052 | 0.3698 | 0.7196 | 0.4885 | 0.8687 |
| Random Forest | 0.8755 | 0.5212 | 0.4561 | 0.4865 | 0.8692 |
| Gradient Boosting | 0.8817 | 0.5610 | 0.3885 | 0.4591 | 0.8767 |

#### Experiment 2 — City holdout 80/20

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| Random Forest | 0.8953 | 0.6578 | 0.6092 | **0.6325** | 0.9051 |
| XGBoost | 0.8547 | 0.5055 | 0.8028 | 0.6204 | 0.9141 |
| LightGBM | 0.8484 | 0.4925 | 0.8134 | 0.6135 | 0.9141 |
| Logistic Regression | 0.8464 | 0.4882 | 0.8028 | 0.6072 | 0.9101 |
| Gradient Boosting | 0.8932 | 0.7182 | 0.4577 | 0.5591 | 0.9141 |

#### Experiment 3 — East → West coast ~80/20

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| Random Forest | 0.9050 | 0.3862 | 0.4124 | **0.3989** | 0.8542 |
| LightGBM | 0.8264 | 0.2709 | 0.7514 | 0.3982 | 0.8685 |
| XGBoost | 0.8251 | 0.2635 | 0.7175 | 0.3854 | 0.8687 |
| Logistic Regression | 0.8207 | 0.2571 | 0.7119 | 0.3778 | 0.8550 |
| Gradient Boosting | 0.9205 | 0.4679 | 0.2881 | 0.3566 | 0.8639 |

---

## 5. What the differences mean

1. **Random hexagon (easiest)**  
   Train and test share cities. The model can learn city-specific patterns and still see similar hexes at test time. This is a useful baseline, but it can **overstate** real-world geographic transfer.

2. **City holdout (harder)**  
   Entire cities are unseen at test time, but the held-out cities are a **random** mix (East, Midwest, South, West). Performance stayed strong here (best F1 **0.63**), which suggests the features transfer reasonably across many U.S. cities when the held-out set is not regionally extreme.

3. **East → West (hardest)**  
   The model is trained only on eastern / non-western cities and scored only on western cities. F1 drops to about **0.40**. That is the clearest sign that **regional shift** (East→West) is more difficult than a random city holdout.

### Takeaway in one line

**Same models and target, different splits:** random hexes and random city holdout look strong; forcing an East→West geographic transfer cuts F1 roughly by a third relative to the city holdout.

---

## 6. Where to find each implementation

| Experiment | Folder / file | Key outputs |
|------------|---------------|-------------|
| 1. Random hexagon | `../../Gun_Violence_ML_Pipeline.executed.ipynb` | Main pipeline metrics (unchanged) |
| 2. City holdout | `../city_holdout_80_20/` | `output/city_assignment.csv`, `metrics.csv`, `results.json`, `README.md` |
| 3. East → West | `../coast_holdout_east_west/` | `output/city_assignment.csv`, `metrics.csv`, `results.json`, `README.md` |

**Isolation:** experiments 2 and 3 only **read** the master CSV and write inside their own folders under `tests/`. Existing main project files were not modified for this comparison.
