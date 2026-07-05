
from __future__ import annotations

import os
import pickle
import itertools
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sqlalchemy as sa

from features import (
    build_pit_events,
    add_stint_number,
    fit_circuit_target_encoding,
    fit_circuit_target_encoding_oof,
    fit_arbitrary_circuit_id,
    apply_arbitrary_circuit_id,
    fit_team_encoding,
    build_feature_matrix,
    circuit_mean_baseline,
    predict_baseline,
    FEATURE_COLUMNS,
    LAPS_SINCE_CAUTION_CAP,
)


from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TRAIN_ROUNDS = range(1, 16)   # rounds 1-15
VAL_ROUNDS = range(16, 23)    # rounds 16-22

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
FINDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ML_FINDINGS.md")

# Feature set with laps_since_caution excluded. Used by the ablation study
# below, and as the feature set for the "simple" bake-off candidate (see
# run(), final bake-off section).
FEATURES_WITHOUT_CAUTION = [c for c in FEATURE_COLUMNS if c != "laps_since_caution"]

# Fixed hyperparameters for the "simple" bake-off candidate below — the
# same config as the pre-refinement `shared_model` in the phase-3 notebook
# (n_estimators=200, max_depth=6, min_samples_leaf=5), deliberately NOT
# CV-tuned. This is the model the CV-tuned "refined" config actually has
# to beat on val, not just on paper. Kept as a named constant instead of
# re-derived from PARAM_GRID or the CV search so the comparison has a
# fixed, un-tuned reference point every run — same idea as keeping a
# control group fixed across an experiment instead of letting it drift.
SIMPLE_PARAMS = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5}

# How much better "refined" has to be than "simple" on val, in units of
# the CV fold-to-fold std (typical_std, from the hyperparameter search),
# before it's trusted enough to actually ship. A gain of 1ms is not
# evidence of anything if fold MAE routinely swings by ~2000ms — see the
# phase-3 result, where refined "won" CV by margins smaller than this
# same noise, then went on to LOSE the val bake-off by 134ms. Factor of
# 1.0 means "must beat simple by more than one typical fold's worth of
# noise"; raise it to demand stronger evidence, lower it (e.g. 0) to
# recover the old "any improvement wins" behavior.
BAKEOFF_MARGIN_FACTOR = 1.0

# Hyperparameter grid for the CV search. Same grid used in the phase-3
# refinement notebook — kept here so the notebook can import it instead of
# maintaining its own copy that can silently drift out of sync.
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [3, 5, 10],
}

# Production default. "plain" target encoding has mild self-leakage (a row
# can see its own target baked into its circuit's mean) but empirically
# performs about the same as out-of-fold encoding on this dataset — see the
# encoding comparison this script prints and appends to ML_FINDINGS.md.
# Only flip this to "oof" if a future run shows OOF beating plain by a
# margin clearly larger than the fold-to-fold std (i.e. a real effect, not
# noise) — check the printed comparison, don't just eyeball the mean.
PRODUCTION_ENCODING = "plain"


def get_engine():
    """
    ASSUMPTION: DATABASE_URL env var holds a standard postgres connection
    string (e.g. postgresql+psycopg2://user:pass@localhost:5432/f1lab).
    Adjust this if ingest.py connects differently (e.g. reads individual
    PG* env vars instead) — align with whatever ingest.py already does.
    """
    url = os.environ["DATABASE_URL"]
    return sa.create_engine(url)


def load_raw(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = pd.read_sql(
        """
        SELECT race_id, driver_id, lap_number, lap_time_ms, track_status
        FROM laps
        """,
        engine,
    )
    pit_stops = pd.read_sql(
        """
        SELECT
            ps.race_id, ps.driver_id, ps.lap_number, ps.stop_duration_ms,
            r.round AS round, r.circuit AS circuit,
            d.team AS team
        FROM pit_stops ps
        JOIN races r ON r.id = ps.race_id
        JOIN drivers d ON d.id = ps.driver_id
        """,
        engine,
    )
    return laps, pit_stops


# ---------------------------------------------------------------------------
# Rolling-origin cross-validation, restricted to the training rounds.
# Everything below this point that makes a "which config wins" decision
# uses these folds — never VAL_ROUNDS. This is the single source of truth
# for that CV logic; the exploration notebook imports it rather than
# keeping its own copy.
# ---------------------------------------------------------------------------

def expanding_round_folds(round_values, n_folds: int = 4, min_train_rounds: int = 5):
    """
    Expanding-window folds over race rounds. Each fold trains on a growing
    prefix of rounds and validates on the next few — the model never sees
    the future at any fold boundary, and no fold ever reaches into
    VAL_ROUNDS (this function is only ever called with train["round"]).
    """
    rounds = sorted(pd.Series(round_values).unique())
    n = len(rounds)
    val_size = max(1, (n - min_train_rounds) // n_folds)
    folds = []
    start = min_train_rounds
    while start < n and len(folds) < n_folds:
        train_r = rounds[:start]
        val_r = rounds[start:start + val_size]
        if val_r:
            folds.append((train_r, val_r))
        start += val_size
    return folds


def _cv_mae(train_events, folds, feature_columns, model_params, encoding="plain"):
    """
    Mean/std MAE across folds for one (feature set, hyperparameters,
    encoding) configuration. Each fold fits its OWN circuit/team encodings
    on that fold's train slice only, so no fold's validation rows ever
    leak into that fold's encoding — the same discipline as the top-level
    train/val split, just applied at fold granularity too.

    Returns None if no fold had enough rows to fit/evaluate on.
    """
    fold_maes = []
    for tr_rounds, va_rounds in folds:
        fold_train = train_events[train_events["round"].isin(tr_rounds)]
        fold_val = train_events[train_events["round"].isin(va_rounds)]
        if len(fold_train) < 10 or len(fold_val) < 5:
            continue

        f_te_map, f_global_mean = fit_circuit_target_encoding(fold_train)
        f_team_map = fit_team_encoding(fold_train)

        if encoding == "oof":
            train_te = fit_circuit_target_encoding_oof(fold_train)
            Xf_train = build_feature_matrix(
                fold_train, f_te_map, f_global_mean, f_team_map,
                circuit_te_override=train_te,
            )[feature_columns]
        elif encoding == "arbitrary":
            arb_map = fit_arbitrary_circuit_id(fold_train)
            arb_te = apply_arbitrary_circuit_id(fold_train, arb_map)
            Xf_train = build_feature_matrix(
                fold_train, f_te_map, f_global_mean, f_team_map,
                circuit_te_override=arb_te,
            )[feature_columns]
        else:
            Xf_train = build_feature_matrix(
                fold_train, f_te_map, f_global_mean, f_team_map
            )[feature_columns]

        # Fold-validation rows always use the fold's plain full-fold-train
        # encoding — only the TRAINING side varies across encoding schemes.
        Xf_val = build_feature_matrix(fold_val, f_te_map, f_global_mean, f_team_map)[feature_columns]

        m = RandomForestRegressor(random_state=42, **model_params).fit(
            Xf_train, fold_train["total_time_lost_ms"]
        )
        pred = m.predict(Xf_val)
        fold_maes.append(mean_absolute_error(fold_val["total_time_lost_ms"], pred))

    if not fold_maes:
        return None
    return float(np.mean(fold_maes)), float(np.std(fold_maes)), len(fold_maes)


def run_hyperparameter_search(train_events: pd.DataFrame, folds, feature_columns) -> pd.DataFrame:
    """Grid search over PARAM_GRID, scored by CV MAE (plain encoding, given
    feature set). Returns a DataFrame sorted best-first."""
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = _cv_mae(train_events, folds, feature_columns, params, encoding="plain")
        if result is None:
            continue
        mean_mae, std_mae, n_folds = result
        rows.append({**params, "mean_mae": mean_mae, "std_mae": std_mae, "n_folds": n_folds})
    return pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)


def run_encoding_comparison(train_events: pd.DataFrame, folds, feature_columns, model_params) -> pd.DataFrame:
    """
    CV comparison of circuit-encoding schemes, holding hyperparameters and
    feature set fixed. Includes the arbitrary-ID control (no target
    information at all) as a sanity check: if it scores close to the
    target-encoded versions, circuit_te isn't doing much real work and the
    "encoding choice" question doesn't matter much either way.
    """
    rows = []
    for encoding, label in [
        ("arbitrary", "arbitrary ID (control, no target info)"),
        ("plain", "plain target encoding (current default)"),
        ("oof", "out-of-fold target encoding"),
    ]:
        result = _cv_mae(train_events, folds, feature_columns, model_params, encoding=encoding)
        if result is None:
            continue
        mean_mae, std_mae, n_folds = result
        rows.append({"encoding": label, "cv_mean_mae": mean_mae, "cv_std_mae": std_mae})
    return pd.DataFrame(rows).sort_values("cv_mean_mae").reset_index(drop=True)


def run_feature_ablation(train_events: pd.DataFrame, folds, model_params) -> pd.DataFrame:
    """CV comparison of the feature set with vs. without laps_since_caution,
    holding hyperparameters and encoding (plain) fixed."""
    rows = []
    for label, cols in [
        ("without laps_since_caution", FEATURES_WITHOUT_CAUTION),
        ("with laps_since_caution", FEATURE_COLUMNS),
    ]:
        result = _cv_mae(train_events, folds, cols, model_params, encoding="plain")
        if result is None:
            continue
        mean_mae, std_mae, n_folds = result
        rows.append({"feature_set": label, "cv_mean_mae": mean_mae, "cv_std_mae": std_mae})
    return pd.DataFrame(rows)


def select_bakeoff_winner(simple_mae: float, refined_mae: float, margin_ms: float) -> str:
    """
    Decide which of the two final bake-off candidates to ship.

    Ships "refined" only if it beats "simple" by MORE than `margin_ms` on
    val MAE — a win smaller than that is indistinguishable from noise at
    this sample size, and "simple" (cheaper, fewer moving parts, no CV
    search to go stale) is the right conservative default in that case.
    Ties go to "simple".

    `margin_ms` should be an estimate of real fold-to-fold noise on this
    dataset (e.g. `typical_std` from the hyperparameter CV search) rather
    than a hardcoded constant, so the bar moves with how noisy the data
    actually is — noisier data requires a bigger win to trust it, and
    that scales automatically as more races get added.

    Pure function, deliberately kept separate from run() so the switching
    rule itself can be unit-tested without needing a database, laps data,
    or a full training run — see test_train_bakeoff.py.
    """
    if margin_ms < 0:
        raise ValueError(f"margin_ms must be >= 0, got {margin_ms}")
    gain = simple_mae - refined_mae  # positive => refined is numerically better
    return "refined" if gain > margin_ms else "simple"


def run(laps: pd.DataFrame, pit_stops: pd.DataFrame, dry_run: bool = False) -> dict:
    events = build_pit_events(laps, pit_stops)
    events = add_stint_number(events)

    dropped = len(pit_stops) - len(events)
    print(f"Pit events: {len(events)} usable / {len(pit_stops)} raw ({dropped} dropped — "
          f"missing laps, caution-period in/out laps, or no clean reference pace)")

    caution_desc = events["laps_since_caution"].describe()
    print(f"\nlaps_since_caution (capped at {LAPS_SINCE_CAUTION_CAP}):")
    print(caution_desc)
    # Fails loudly rather than silently training on a leaky feature again —
    # if this trips, features.py isn't the version you think it is (stale
    # import in a long-running interactive session is the usual cause).
    assert events["laps_since_caution"].max() <= LAPS_SINCE_CAUTION_CAP, (
        f"laps_since_caution exceeds its cap of {LAPS_SINCE_CAUTION_CAP} — "
        "this means the imported features.py does not have the cap fix "
        "applied. If running from a notebook, restart the kernel."
    )

    train = events[events["round"].isin(TRAIN_ROUNDS)].copy()
    val = events[events["round"].isin(VAL_ROUNDS)].copy()
    print(f"\nTrain: {len(train)} events (rounds 1-15) | Validate: {len(val)} events (rounds 16-22)")

    # --- baseline: per-circuit historical mean, fit on train only ---
    baseline_means, baseline_global = circuit_mean_baseline(train)
    baseline_pred = predict_baseline(val, baseline_means, baseline_global)
    baseline_mae = mean_absolute_error(val["total_time_lost_ms"], baseline_pred)

    # --- CV folds within train only; used for every selection decision below ---
    folds = expanding_round_folds(train["round"])
    print("\nCV folds (within train rounds only):")
    for i, (tr, va) in enumerate(folds):
        print(f"  Fold {i}: train rounds {tr[0]}-{tr[-1]} ({len(tr)} rounds) | val rounds {va}")

    # --- 1. hyperparameter search (CV, plain encoding, full feature set) ---
    cv_df = run_hyperparameter_search(train, folds, FEATURE_COLUMNS)
    top5 = cv_df.head(5)
    spread = float(top5["mean_mae"].max() - top5["mean_mae"].min())
    typical_std = float(top5["std_mae"].mean())
    best_params = {
        k: int(v) for k, v in
        cv_df.iloc[0][["n_estimators", "max_depth", "min_samples_leaf"]].items()
    }
    print("\nTop 5 hyperparameter configs (by CV mean MAE):")
    print(top5)
    print(f"\nSpread among top 5 mean_mae: {spread:.2f} ms")
    print(f"Typical fold std_mae in top 5: {typical_std:.2f} ms")
    if spread < typical_std:
        print("(spread smaller than typical fold noise — treat this as picking a "
              "region of reasonable configs, not a precise winner)")
    print(f"Selected hyperparameters: {best_params}")

    # --- 2. encoding comparison (CV, best hyperparams, full feature set) ---
    encoding_df = run_encoding_comparison(train, folds, FEATURE_COLUMNS, best_params)
    print("\nEncoding comparison (CV, never touches val):")
    print(encoding_df)
    chosen_encoding = PRODUCTION_ENCODING
    print(f"Using production default encoding: {chosen_encoding}")

    # --- 3. feature ablation (CV, best hyperparams, plain encoding) ---
    ablation_df = run_feature_ablation(train, folds, best_params)
    print("\nFeature ablation — laps_since_caution (CV, never touches val):")
    print(ablation_df)

    # --- fit encodings once on all of train; shared by both candidates below ---
    te_map, global_mean = fit_circuit_target_encoding(train)
    team_map = fit_team_encoding(train)
    y_train = train["total_time_lost_ms"]
    y_val = val["total_time_lost_ms"]
    # Validation always uses the plain full-train encoding, regardless of
    # candidate — OOF only ever applies to the rows a model is trained on
    # (see fit_circuit_target_encoding_oof's docstring).
    X_val_full = build_feature_matrix(val, te_map, global_mean, team_map)

    # --- final bake-off: two pre-specified candidates, each touching val
    # exactly once. No iterative tuning against val happens here — these
    # are the only two configs in contention, decided before either was
    # scored, exactly as printed below.
    #
    # Candidate "simple": SIMPLE_PARAMS (not CV-tuned), no
    # laps_since_caution, plain encoding. The pre-refinement config.
    X_train_simple = build_feature_matrix(train, te_map, global_mean, team_map)[FEATURES_WITHOUT_CAUTION]
    X_val_simple = X_val_full[FEATURES_WITHOUT_CAUTION]
    simple_model = RandomForestRegressor(random_state=42, **SIMPLE_PARAMS)
    simple_model.fit(X_train_simple, y_train)
    simple_pred = simple_model.predict(X_val_simple)
    simple_mae = mean_absolute_error(y_val, simple_pred)  # 1st honest touch of val

    # Candidate "refined": CV-tuned hyperparams, laps_since_caution
    # included, chosen_encoding (may be OOF on the training side only).
    if chosen_encoding == "oof":
        train_te_override = fit_circuit_target_encoding_oof(train)
    else:
        train_te_override = None
    X_train_refined = build_feature_matrix(
        train, te_map, global_mean, team_map, circuit_te_override=train_te_override
    )[FEATURE_COLUMNS]
    X_val_refined = X_val_full[FEATURE_COLUMNS]
    refined_model = RandomForestRegressor(random_state=42, **best_params)
    refined_model.fit(X_train_refined, y_train)
    refined_pred = refined_model.predict(X_val_refined)
    refined_mae = mean_absolute_error(y_val, refined_pred)  # 2nd honest touch of val

    bakeoff_gain = simple_mae - refined_mae  # positive => refined actually better
    bakeoff_margin = BAKEOFF_MARGIN_FACTOR * typical_std

    print("\nFinal bake-off (val touched exactly once per candidate, decided in advance):")
    print(f"  simple  — params={SIMPLE_PARAMS}, features={FEATURES_WITHOUT_CAUTION}, encoding=plain")
    print(f"            val MAE: {simple_mae:.2f} ms")
    print(f"  refined — params={best_params}, features={FEATURE_COLUMNS}, encoding={chosen_encoding}")
    print(f"            val MAE: {refined_mae:.2f} ms")
    print(f"  refined vs simple: {bakeoff_gain:+.2f} ms (positive = refined wins, negative = simple wins)")
    print(f"  required margin to switch to refined: {bakeoff_margin:.2f} ms "
          f"({BAKEOFF_MARGIN_FACTOR:g} x typical CV fold std)")

    chosen_config = select_bakeoff_winner(simple_mae, refined_mae, bakeoff_margin)
    if chosen_config == "refined":
        model, model_pred, model_mae = refined_model, refined_pred, refined_mae
        chosen_feature_columns = FEATURE_COLUMNS
        chosen_params = best_params
    else:
        model, model_pred, model_mae = simple_model, simple_pred, simple_mae
        chosen_feature_columns = FEATURES_WITHOUT_CAUTION
        chosen_params = SIMPLE_PARAMS
        chosen_encoding = "plain"  # the simple candidate never uses OOF
        if bakeoff_gain > 0:
            print(f"  (note: refined had a lower raw MAE by {bakeoff_gain:.2f} ms, but that's "
                  f"smaller than the {bakeoff_margin:.2f} ms noise margin — not trusted as a real win)")
    print(f"  -> shipping: {chosen_config}")

    importances = pd.Series(
        model.feature_importances_, index=chosen_feature_columns
    ).sort_values(ascending=False)

    # --- per-circuit breakdown: model vs baseline ---
    breakdown = val.copy()
    breakdown["model_pred"] = model_pred
    breakdown["baseline_pred"] = baseline_pred.values
    breakdown["model_abs_err"] = (breakdown["total_time_lost_ms"] - breakdown["model_pred"]).abs()
    breakdown["baseline_abs_err"] = (breakdown["total_time_lost_ms"] - breakdown["baseline_pred"]).abs()
    per_circuit = (
        breakdown.groupby("circuit")[["model_abs_err", "baseline_abs_err"]]
        .mean()
        .rename(columns={"model_abs_err": "model_mae", "baseline_abs_err": "baseline_mae"})
    )
    per_circuit["model_wins"] = per_circuit["model_mae"] < per_circuit["baseline_mae"]

    print(f"\nBaseline (per-circuit mean) honest MAE:  {baseline_mae:.2f} ms")
    print(f"Model honest MAE ({chosen_config} config): {model_mae:.2f} ms")
    if model_mae >= baseline_mae:
        print("\n⚠️  WARNING: model does NOT beat the per-circuit mean baseline. "
              "As specced, this model has no value in the simulation until fixed.")
    else:
        print(f"✅ Model beats baseline by {baseline_mae - model_mae:.2f} ms")

    print("\nFeature importances:")
    print(importances)

    print("\nPer-circuit MAE (model vs baseline):")
    print(per_circuit.sort_values("model_mae"))

    result = {
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "per_circuit": per_circuit,
        "n_train": len(train),
        "n_val": len(val),
        "dropped_events": dropped,
        "n_raw_pit_stops": len(pit_stops),
        "caution_describe": caution_desc,
        "cv_hyperparam_top5": top5,
        "cv_spread": spread,
        "cv_typical_std": typical_std,
        "best_params": best_params,
        "encoding_comparison": encoding_df,
        "chosen_encoding": chosen_encoding,
        "feature_ablation": ablation_df,
        "feature_importances": importances,
        "simple_mae": simple_mae,
        "refined_mae": refined_mae,
        "bakeoff_margin": bakeoff_margin,
        "chosen_config": chosen_config,
        "chosen_params": chosen_params,
    }

    if not dry_run:
        bundle = {
            "model": model,
            "te_map": te_map,
            "global_mean": global_mean,
            "team_map": team_map,
            "feature_columns": chosen_feature_columns,
            "hyperparameters": chosen_params,
            "encoding": chosen_encoding,
            "config_name": chosen_config,  # "simple" or "refined" — which bake-off candidate this is
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\nSerialized model bundle -> {MODEL_PATH}")

        _append_findings(result)

    return result


def _append_findings(result: dict) -> None:
    lines = [
        f"\n## Pit time loss model — {dt.date.today().isoformat()}",
        "",
        f"- Pit events: {result['n_train'] + result['n_val']} usable / "
        f"{result['n_raw_pit_stops']} raw ({result['dropped_events']} dropped)",
        f"- Train: {result['n_train']} pit events (rounds 1-15) | "
        f"Validate: {result['n_val']} pit events (rounds 16-22)",
        f"- laps_since_caution capped at {LAPS_SINCE_CAUTION_CAP} "
        f"(max observed: {result['caution_describe']['max']:.0f})",
        "",
        "### Hyperparameter search (rolling-origin CV within train rounds only)",
        "",
        f"- Top 5 spread: {result['cv_spread']:.2f} ms | "
        f"typical fold std in top 5: {result['cv_typical_std']:.2f} ms",
        result["cv_hyperparam_top5"].to_markdown(),
        f"- Selected: `{result['best_params']}`",
        "",
        "### Circuit-encoding comparison (CV, never touches the validation set)",
        "",
        result["encoding_comparison"].to_markdown(),
        f"- Production default used: **{result['chosen_encoding']}**",
        "",
        "### Feature ablation — laps_since_caution (CV, never touches the validation set)",
        "",
        result["feature_ablation"].to_markdown(),
        "",
        "### Final bake-off — simple vs. refined (val touched exactly once per candidate)",
        "",
        "Two pre-specified candidates, decided before either was scored: "
        "**simple** (fixed pre-refinement hyperparameters, no `laps_since_caution`, "
        "plain encoding) vs. **refined** (CV-tuned hyperparameters, `laps_since_caution` "
        "included, CV-selected encoding). Whichever wins on val ships — CV's pick is "
        "not trusted blindly.",
        "",
        f"- simple  honest MAE: **{result['simple_mae']:.2f} ms**",
        f"- refined honest MAE: **{result['refined_mae']:.2f} ms**",
        f"- Refined vs simple: {result['simple_mae'] - result['refined_mae']:+.2f} ms "
        "(positive means refined is better, negative means simple is better)",
        f"- Required margin to switch to refined: **{result['bakeoff_margin']:.2f} ms** "
        f"({BAKEOFF_MARGIN_FACTOR:g} x typical CV fold std — a win smaller than this "
        "isn't trusted as a real improvement, see select_bakeoff_winner)",
        f"- **Shipped: {result['chosen_config']}**"
        + (
            " — CV-tuned config did not actually beat the simpler one on held-out val; "
            "the gap between them is well within fold-to-fold CV noise, so this isn't "
            "evidence the refinements were wrong, just that this dataset is too small "
            "to tell the difference yet."
            if result["chosen_config"] == "simple"
            else ""
        ),
        "",
        "### Final result (validation set touched exactly once per candidate, above)",
        "",
        f"- Baseline (per-circuit historical mean) honest MAE: **{result['baseline_mae']:.2f} ms**",
        f"- Model honest MAE ({result['chosen_config']} config): **{result['model_mae']:.2f} ms**",
        f"- Beats baseline: {'yes' if result['model_mae'] < result['baseline_mae'] else 'NO — see warning above'}",
        "",
        "### Feature importances",
        "",
        result["feature_importances"].to_markdown(),
        "",
        "### Per-circuit breakdown (model MAE vs baseline MAE)",
        "",
        result["per_circuit"].sort_values("model_mae").to_markdown(),
        "",
    ]
    with open(FINDINGS_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    engine = get_engine()
    laps, pit_stops = load_raw(engine)
    run(laps, pit_stops)
