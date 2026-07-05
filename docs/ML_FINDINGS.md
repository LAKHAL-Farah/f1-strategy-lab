
## Tire Degradation Model 

Configuration: tuned hyperparameters ({'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5}) + out-of-fold target-encoded
circuit + laps_since_sc. Selected in 03_model_refinement.ipynb after testing each
change in isolation, then confirming they combine (66ms further gain over target
encoding alone, less than the ~181ms naive sum would suggest the two features
overlap somewhat, but both still contribute).

- Train rounds: 1-15 (15167 rows)
- Validation rounds: 16-22 (6637 rows)
- **Honest MAE (time-aware split): 5102.28 ms**

### MAE by compound
  - HARD: 4687.62 ms
  - MEDIUM: 6335.32 ms
  - SOFT: 2451.55 ms

### MAE by circuit
  - Las Vegas: 9632.13 ms
  - Lusail: 7090.75 ms
  - Suzuka: 6544.03 ms
  - Sao Paulo: 6512.46 ms
  - Mexico City: 4953.66 ms
  - Austin: 2293.06 ms
  - Yas Island: 1303.99 ms

### Known limitations
- Fuel-load confound: `lap_number` proxies fuel burn-off, confounded with genuine degradation.
- Target-encoded circuit is noisier for circuits with few training-round appearances;
  circuits unseen in training fall back to the global training mean.
- WET/INTERMEDIATE compounds underrepresented in a single season low confidence.
- `laps_since_sc` does not close the gap on safety-car-affected laps: MAE was still
  markedly worse just after a restart (laps_since_sc <= 3) than at steady state in
  validation testing the feature helps overall, it does not solve this limitation.
- Rookie / partial-season drivers have thin data relative to full-season drivers.
- Baselines computed across the full season, not just the training split (see features.py docstring).
- Data known to still contain gaps as of the last data-quality pass see
  CHECKPOINT_DATA_EXPLORATION_CLEANING.md (rounds 6/13 previously missing;
  compound-label cleanup and pit_stops parsing tracked separately, the latter
  doesn't affect this model).

---

## Tire Degradation Model � 2026-07-04 14:08 UTC

Configuration: tuned hyperparameters ({'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5}) + out-of-fold target-encoded
circuit + laps_since_sc. Selected in 03_model_refinement.ipynb after testing each
change in isolation, then confirming they combine (66ms further gain over target
encoding alone, less than the ~181ms naive sum would suggest the two features
overlap somewhat, but both still contribute).

- Train rounds: 1-15 (15167 rows)
- Validation rounds: 16-22 (6637 rows)
- **Honest MAE (time-aware split): 5102.28 ms**

### MAE by compound
  - HARD: 4687.62 ms
  - MEDIUM: 6335.32 ms
  - SOFT: 2451.55 ms

### MAE by circuit
  - Las Vegas: 9632.13 ms
  - Lusail: 7090.75 ms
  - Suzuka: 6544.03 ms
  - Sao Paulo: 6512.46 ms
  - Mexico City: 4953.66 ms
  - Austin: 2293.06 ms
  - Yas Island: 1303.99 ms

### Known limitations
- Fuel-load confound: `lap_number` proxies fuel burn-off, confounded with genuine degradation.
- Target-encoded circuit is noisier for circuits with few training-round appearances;
  circuits unseen in training fall back to the global training mean.
- WET/INTERMEDIATE compounds underrepresented in a single season low confidence.
- `laps_since_sc` does not close the gap on safety-car-affected laps: MAE was still
  markedly worse just after a restart (laps_since_sc <= 3) than at steady state in
  validation testing  the feature helps overall, it does not solve this limitation.
- Rookie / partial-season drivers have thin data relative to full-season drivers.
- Baselines computed across the full season, not just the training split (see features.py docstring).
- Data known to still contain gaps as of the last data-quality pass see
  CHECKPOINT_DATA_EXPLORATION_CLEANING.md (rounds 6/13 previously missing;
  compound-label cleanup and pit_stops parsing tracked separately, the latter
  doesn't affect this model).

---
