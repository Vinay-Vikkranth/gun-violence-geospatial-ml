# East → West Coast Holdout (~80/20)

Self-contained experiment in `tests/coast_holdout_east_west/`. Does **not** modify any main project files.

## Setup

| Item | Detail |
|------|--------|
| Data | `../../Master_Hexagon_Gun_Violence_Summary_Table.csv` |
| Train | Eastern / non-western US cities (**27** cities, **9,133** hexes, **79.8%**) |
| Test | West Coast + western US cities (**8** cities, **2,315** hexes, **20.2%**) |
| Target | `Total_Gun_Violence > 0` |
| Pipeline | Same preprocessing + models as the main gun violence ML pipeline |

### Why western cities beyond the Pacific coast?

Strict Pacific West Coast only (LA, Oakland, Portland, Seattle) is ~10% of hexes.  
To keep an **~80/20 hex split** while still testing on the West, the test set also includes Phoenix, Denver, Las Vegas, and North Las Vegas.

**Test cities:** Denver, Las Vegas, Los Angeles, North Las Vegas, Oakland, Phoenix, Portland, Seattle

## Best result

- **Best model @ 0.50:** Random Forest — **F1 = 0.3989**
- **Tuned threshold:** 0.44 — **F1 = 0.4084**

Full metrics: `output/metrics.csv` · Summary: `output/results.json`

## Re-run

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace --ExecutePreprocessor.timeout=600
```
