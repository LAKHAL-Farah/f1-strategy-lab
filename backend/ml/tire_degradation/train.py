"""Repeatable training pipeline for the tire degradation model.

Final configuration, validated in 03_model_refinement.ipynb against real data:
tuned hyperparameters + out-of-fold target-encoded circuit + laps_since_sc.
Combined honest MAE was 5102 ms in that validation run, vs. 7209 ms for the
original untuned/arbitrary-ID/no-SC-feature baseline in 02_tire_degradation_model.ipynb.

This script only ever produces the honest, time-aware-split metric — there is
no naive-split code path here on purpose, so a bad number can never
accidentally get appended to ML_FINDINGS.md.

Usage:
    cd backend
    uv run python ml/tire_degradation/train.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from ml.tire_degradation.features import (  # noqa: E402
    build_training_frame,
    assemble_feature_matrix,
    circuit_target_encoding,
    apply_circuit_encoding,
    TUNED_PARAMS,
)

MODEL_DIR = BACKEND / "ml" / "tire_degradation"
MODEL_PATH = MODEL_DIR / "model.pkl"
FINDINGS_PATH = BACKEND / "ML_FINDINGS.md"

TRAIN_ROUNDS = list(range(1, 16))
VAL_ROUNDS = list(range(16, 23))


def train_and_evaluate(engine) -> dict:
    data = build_training_frame(engine)

    train_mask = data["round"].isin(TRAIN_ROUNDS)
    val_mask = data["round"].isin(VAL_ROUNDS)

    if val_mask.sum() == 0:
        raise RuntimeError(
            f"No validation rows for rounds {VAL_ROUNDS} — check that those "
            f"rounds are actually ingested before training. See "
            f"CHECKPOINT_DATA_EXPLORATION_CLEANING.md for known ingestion gaps."
        )

    # Circuit encoding MUST be fit on training rows only — see features.py docstring.
    te_map, global_mean = circuit_target_encoding(data, train_mask)
    data["circuit_target_enc"] = apply_circuit_encoding(data["circuit"], te_map, global_mean)

    X = assemble_feature_matrix(data)
    y = data["lap_time_delta"]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    model = RandomForestRegressor(**TUNED_PARAMS, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    honest_mae = mean_absolute_error(y_val, pred)

    val_results = data[val_mask].copy()
    val_results["pred"] = pred
    val_results["abs_error"] = (val_results["lap_time_delta"] - val_results["pred"]).abs()

    by_compound = val_results.groupby("compound")["abs_error"].mean().round(2).to_dict()
    by_circuit = (
        val_results.groupby("circuit")["abs_error"].mean().round(2).sort_values(ascending=False).to_dict()
    )

    return {
        "model": model,
        "te_map": te_map,
        "global_mean": global_mean,
        "feature_cols": list(X.columns),
        "honest_mae": honest_mae,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "train_rounds": TRAIN_ROUNDS,
        "val_rounds": VAL_ROUNDS,
        "by_compound": by_compound,
        "by_circuit": by_circuit,
    }


def save_model(results: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": results["model"],
        "te_map": results["te_map"],
        "global_mean": results["global_mean"],
        "feature_cols": results["feature_cols"],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")


def append_findings(results: dict) -> None:
    """Append-only. The metric that goes in is always the honest, time-aware
    split metric produced by this script — never hand-edit a number in here."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    compound_lines = "\n".join(f"  - {c}: {mae} ms" for c, mae in sorted(results["by_compound"].items()))
    circuit_lines = "\n".join(f"  - {c}: {mae} ms" for c, mae in results["by_circuit"].items())

    entry = f"""
## Tire Degradation Model — {timestamp}

Configuration: tuned hyperparameters ({TUNED_PARAMS}) + out-of-fold target-encoded
circuit + laps_since_sc. Selected in 03_model_refinement.ipynb after testing each
change in isolation, then confirming they combine (66ms further gain over target
encoding alone, less than the ~181ms naive sum would suggest — the two features
overlap somewhat, but both still contribute).

- Train rounds: {results['train_rounds'][0]}-{results['train_rounds'][-1]} ({results['n_train']} rows)
- Validation rounds: {results['val_rounds'][0]}-{results['val_rounds'][-1]} ({results['n_val']} rows)
- **Honest MAE (time-aware split): {results['honest_mae']:.2f} ms**

### MAE by compound
{compound_lines}

### MAE by circuit
{circuit_lines}

### Known limitations
- Fuel-load confound: `lap_number` proxies fuel burn-off, confounded with genuine degradation.
- Target-encoded circuit is noisier for circuits with few training-round appearances;
  circuits unseen in training fall back to the global training mean.
- WET/INTERMEDIATE compounds underrepresented in a single season — low confidence.
- `laps_since_sc` does not close the gap on safety-car-affected laps: MAE was still
  markedly worse just after a restart (laps_since_sc <= 3) than at steady state in
  validation testing — the feature helps overall, it does not solve this limitation.
- Rookie / partial-season drivers have thin data relative to full-season drivers.
- Baselines computed across the full season, not just the training split (see features.py docstring).
- Data known to still contain gaps as of the last data-quality pass — see
  CHECKPOINT_DATA_EXPLORATION_CLEANING.md (rounds 6/13 previously missing;
  compound-label cleanup and pit_stops parsing tracked separately, the latter
  doesn't affect this model).

---
"""
    with open(FINDINGS_PATH, "a") as f:
        f.write(entry)
    print(f"Appended results to {FINDINGS_PATH}")


def main():
    load_dotenv(BACKEND / ".env")
    database_url = os.getenv("DATABASE_URL")
    if database_url is None:
        raise RuntimeError(f"DATABASE_URL not found in {BACKEND / '.env'}")

    engine = create_engine(database_url)

    print("Building training frame and training model (honest, time-aware split only)...")
    results = train_and_evaluate(engine)

    print(f"Honest MAE: {results['honest_mae']:.2f} ms")
    print(f"Train rows: {results['n_train']}  |  Val rows: {results['n_val']}")

    save_model(results)
    append_findings(results)


if __name__ == "__main__":
    main()
