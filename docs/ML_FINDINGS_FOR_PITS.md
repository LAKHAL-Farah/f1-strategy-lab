
## Pit time loss model — 2026-07-04

- Pit events: 342 usable / 538 raw (196 dropped)
- Train: 233 pit events (rounds 1-15) | Validate: 109 pit events (rounds 16-22)
- laps_since_caution capped at 20 (max observed: 20)

### Hyperparameter search (rolling-origin CV within train rounds only)

- Top 5 spread: 55.52 ms | typical fold std in top 5: 1936.08 ms
|    |   n_estimators |   max_depth |   min_samples_leaf |   mean_mae |   std_mae |   n_folds |
|---:|---------------:|------------:|-------------------:|-----------:|----------:|----------:|
|  0 |            200 |           8 |                  3 |    5723.35 |   1988.68 |         4 |
|  1 |            200 |           6 |                  3 |    5728.45 |   1974.18 |         4 |
|  2 |            100 |           8 |                  3 |    5746.86 |   1942.45 |         4 |
|  3 |            100 |           6 |                  3 |    5755.34 |   1940.13 |         4 |
|  4 |            200 |           4 |                  3 |    5778.87 |   1834.98 |         4 |
- Selected: `{'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 3}`

### Circuit-encoding comparison (CV, never touches the validation set)

|    | encoding                                |   cv_mean_mae |   cv_std_mae |
|---:|:----------------------------------------|--------------:|-------------:|
|  0 | out-of-fold target encoding             |       5347.71 |      1845.19 |
|  1 | plain target encoding (current default) |       5723.35 |      1988.68 |
|  2 | arbitrary ID (control, no target info)  |       8856.27 |      1803.72 |
- Production default used: **plain**

### Feature ablation — laps_since_caution (CV, never touches the validation set)

|    | feature_set                |   cv_mean_mae |   cv_std_mae |
|---:|:---------------------------|--------------:|-------------:|
|  0 | without laps_since_caution |       5844.77 |      2019.48 |
|  1 | with laps_since_caution    |       5723.35 |      1988.68 |

### Final bake-off — simple vs. refined (val touched exactly once per candidate)

Two pre-specified candidates, decided before either was scored: **simple** (fixed pre-refinement hyperparameters, no `laps_since_caution`, plain encoding) vs. **refined** (CV-tuned hyperparameters, `laps_since_caution` included, CV-selected encoding). Whichever wins on val ships — CV's pick is not trusted blindly.

- simple  honest MAE: **4760.81 ms**
- refined honest MAE: **4895.29 ms**
- Refined vs simple: -134.47 ms (positive means refined is better, negative means simple is better)
- **Shipped: simple** — CV-tuned config did not actually beat the simpler one on held-out val; the gap between them is well within fold-to-fold CV noise, so this isn't evidence the refinements were wrong, just that this dataset is too small to tell the difference yet.

### Final result (validation set touched exactly once per candidate, above)

- Baseline (per-circuit historical mean) honest MAE: **5051.37 ms**
- Model honest MAE (simple config): **4760.81 ms**
- Beats baseline: yes

### Feature importances

|              |        0 |
|:-------------|---------:|
| circuit_te   | 0.650426 |
| team_code    | 0.226842 |
| stint_number | 0.122732 |

### Per-circuit breakdown (model MAE vs baseline MAE)

| circuit     |   model_mae |   baseline_mae | model_wins   |
|:------------|------------:|---------------:|:-------------|
| Austin      |     2901.6  |        3301.8  | True         |
| Mexico City |     3037.91 |        3827.25 | True         |
| Yas Island  |     3490.67 |        4190.39 | True         |
| Suzuka      |     4004.17 |        4725.37 | True         |
| Las Vegas   |     6159.79 |        6177.47 | True         |
| Lusail      |     9654.41 |        8877.19 | False        |

