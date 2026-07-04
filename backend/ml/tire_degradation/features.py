"""Single source of truth for tire degradation feature engineering.

Winning configuration, validated in 03_model_refinement.ipynb against real data:
tuned hyperparameters + out-of-fold target-encoded circuit + laps_since_sc.
Combined honest MAE: 5102 ms, vs. 7209 ms for the original notebook-02 baseline
(untuned, arbitrary circuit ID, no SC feature). See ML_FINDINGS.md for the full
comparison table and the caveat that laps_since_sc's marginal contribution once
circuit is properly encoded is modest (~66ms) — real, but smaller than it looked
in isolation against the weaker arbitrary-ID baseline.

Both train.py and model.py import from here. If you feel the urge to write
feature logic inside model.py, train.py, or a notebook directly instead —
that urge is wrong. Add a function here instead.

KNOWN DATA CAVEAT (see CHECKPOINT_DATA_EXPLORATION_CLEANING.md):
ingest.py currently has a bug where some laps get the literal string "None"
written into `compound` instead of a true SQL NULL. Until that's fixed at
the source, every query in this module explicitly excludes `compound = 'None'`
in addition to `IS NULL` and `'UNKNOWN'`. Remove that clause once ingest.py
is fixed and the column is re-ingested.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

CLEAN_LAPS_QUERY = text("""
    SELECT
        l.race_id,
        r.round,
        r.circuit,
        r.race_date,
        l.driver_id,
        l.lap_number,
        l.lap_time_ms,
        l.compound,
        l.tire_age,
        l.stint_number,
        l.track_status,
        l.ambient_temp_c,
        l.track_temp_c,
        l.rainfall
    FROM laps l
    JOIN races r ON r.id = l.race_id
    WHERE l.lap_time_ms IS NOT NULL
      AND l.compound IS NOT NULL
      AND l.compound != 'UNKNOWN'
      AND l.compound != 'None'
      AND l.track_status = '1'
    ORDER BY r.round, l.driver_id, l.lap_number
""")

FULL_LAP_HISTORY_QUERY = text("""
    SELECT race_id, driver_id, lap_number, track_status
    FROM laps
    ORDER BY race_id, driver_id, lap_number
""")

TUNED_PARAMS = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5}

# circuit_target_enc replaces the old arbitrary circuit_code integer.
FEATURE_BASE_COLS = ["tire_age", "lap_number", "circuit_target_enc", "ambient_temp_c", "track_temp_c", "laps_since_sc"]
KNOWN_COMPOUNDS = ["HARD", "MEDIUM", "SOFT", "INTER", "WET"]


def load_clean_laps(engine: Engine) -> pd.DataFrame:
    """Pull all usable 2023-season laps: non-null lap time, real compound label,
    green-flag track status only. See CLEAN_LAPS_QUERY for the exact filter."""
    return pd.read_sql(CLEAN_LAPS_QUERY, engine)


def load_full_lap_history(engine: Engine) -> pd.DataFrame:
    """Pull EVERY lap (including SC/VSC/red-flag laps), needed only to derive
    laps_since_sc — this is intentionally not filtered like load_clean_laps,
    since the recovery signal requires seeing the disruptions themselves."""
    return pd.read_sql(FULL_LAP_HISTORY_QUERY, engine)


def compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Fresh-tire baseline lap time (median, tire_age <= 2) per (circuit, compound),
    computed across the FULL season rather than just the training split.

    Deliberate tradeoff: fresh-tire laps are a small fraction of total laps at
    any one circuit, so restricting this to only the training rounds would make
    many (circuit, compound) baselines noisy or entirely missing for validation
    rounds. Computing baselines season-wide leaks a small amount of full-season
    information into the target definition itself (not into the model's
    features or training rows) — a narrower, deliberate compromise, not the
    same leakage class the time-aware train/val split fixes.
    """
    fresh = df[df["tire_age"] <= 2]
    return (
        fresh.groupby(["circuit", "compound"])["lap_time_ms"]
        .median()
        .reset_index()
        .rename(columns={"lap_time_ms": "baseline_ms"})
    )


def add_laps_since_sc(df: pd.DataFrame) -> pd.DataFrame:
    """Per (race, driver): laps since the most recent non-green track_status.
    Vectorized groupby+ffill — NOT groupby().apply(), which breaks on
    single-group edge cases (verified during development). Laps before any
    disruption in the race get a sentinel of 99, then the whole column is
    capped at 20 so the "long ago" tail doesn't dominate the feature's scale."""
    df = df.sort_values(["race_id", "driver_id", "lap_number"]).copy()
    disrupted = df["track_status"] != "1"
    df["_disruption_lap"] = df["lap_number"].where(disrupted)
    df["_disruption_lap"] = df.groupby(["race_id", "driver_id"])["_disruption_lap"].ffill()
    df["laps_since_sc"] = (df["lap_number"] - df["_disruption_lap"]).fillna(99)
    df["laps_since_sc"] = df["laps_since_sc"].clip(upper=20)
    return df.drop(columns="_disruption_lap")


def circuit_target_encoding(data: pd.DataFrame, train_mask: pd.Series) -> tuple[dict, float]:
    """Out-of-fold-safe circuit encoding: each circuit's mean lap_time_delta,
    computed using ONLY rows where train_mask is True. Never compute this from
    the full dataset — that leaks target information across the train/val
    split, which is an easy and easy-to-miss mistake with target encoding
    specifically (verified safe in 03_model_refinement.ipynb).

    Returns (circuit -> mean_delta dict, global_train_mean). Circuits present
    in validation/inference but absent from training fall back to
    global_train_mean — the model has no real information about them
    otherwise, and that's an honest fallback rather than a crash.
    """
    train_df = data[train_mask]
    circuit_means = train_df.groupby("circuit")["lap_time_delta"].mean().to_dict()
    global_mean = float(train_df["lap_time_delta"].mean())
    return circuit_means, global_mean


def apply_circuit_encoding(circuits: pd.Series, te_map: dict, global_mean: float) -> pd.Series:
    return circuits.map(lambda c: te_map.get(c, global_mean))


def build_training_frame(engine: Engine) -> pd.DataFrame:
    """DB -> merged per-lap frame with target + laps_since_sc attached.
    Does NOT include circuit encoding or the final feature matrix — those
    depend on knowing the train/val split first (see circuit_target_encoding),
    so callers compute those after masking. Carries `round`, `circuit`,
    `compound`, `race_id`, `driver_id`, `lap_number` for split logic, error
    breakdowns, and the SC-history merge key.
    """
    raw = load_clean_laps(engine)
    baselines = compute_baselines(raw)

    data = raw.merge(baselines, on=["circuit", "compound"], how="inner")
    data["lap_time_delta"] = data["lap_time_ms"] - data["baseline_ms"]

    full_history = load_full_lap_history(engine)
    full_history = add_laps_since_sc(full_history)
    data = data.merge(
        full_history[["race_id", "driver_id", "lap_number", "laps_since_sc"]],
        on=["race_id", "driver_id", "lap_number"], how="left",
    )
    data["laps_since_sc"] = data["laps_since_sc"].fillna(20)

    return data


def assemble_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Requires `circuit_target_enc` already attached to `data` (see
    circuit_target_encoding + apply_circuit_encoding). Builds the final X,
    with a stable column set regardless of which compounds are present in
    this particular data pull."""
    compound_dummies = pd.get_dummies(data["compound"], prefix="compound")
    for c in KNOWN_COMPOUNDS:
        col = f"compound_{c}"
        if col not in compound_dummies.columns:
            compound_dummies[col] = False

    X = pd.concat([data[FEATURE_BASE_COLS], compound_dummies], axis=1)
    return X.fillna(X.median(numeric_only=True))


def build_inference_row(features: dict, te_map: dict, global_mean: float, feature_cols: list[str]) -> dict:
    """Turn a single raw-feature dict into a row matching the trained model's
    column order/encoding exactly. This is the ONLY place that should ever
    translate a `{"compound": ..., "tire_age": ...}`-style dict into model
    input — model.py calls this, nothing else re-derives it.

    `features` must include: compound, tire_age, lap_number, circuit,
    ambient_temp_c, track_temp_c, laps_since_sc (laps since the most recent
    non-green track status in the current race — 20 if none yet / unknown).

    Unlike the old arbitrary-ID version, an unseen `circuit` does NOT raise —
    it falls back to global_mean, matching exactly how validation rows with
    an unseen circuit were handled during training. This is a deliberate,
    disclosed approximation, not silent failure.
    """
    circuit_enc = te_map.get(features["circuit"], global_mean)
    row = {
        "tire_age": features["tire_age"],
        "lap_number": features["lap_number"],
        "circuit_target_enc": circuit_enc,
        "ambient_temp_c": features.get("ambient_temp_c"),
        "track_temp_c": features.get("track_temp_c"),
        "laps_since_sc": features.get("laps_since_sc", 20),
    }
    for c in KNOWN_COMPOUNDS:
        row[f"compound_{c}"] = (features["compound"] == c)

    return {col: row.get(col, 0) for col in feature_cols}
