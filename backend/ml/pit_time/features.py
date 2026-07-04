

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Target construction
# ---------------------------------------------------------------------------

REFERENCE_WINDOW = 2  # how many clean laps before/after the stop to use as reference pace
CLEAN_TRACK_STATUS = "1"  # green flag only — see tire model's lesson on track_status filtering
MIN_LAPS_SINCE_CAUTION = 2  # see compute_laps_since_caution — restart-bunching recovery buffer
LAPS_SINCE_CAUTION_CAP = 20  # see compute_laps_since_caution — collapses "comfortably clear of
                              # any caution" and "no caution ever happened this race" into one
                              # bucket, instead of letting the counter keep climbing and silently
                              # re-encode race progress / stint_number through the back door


def _clean_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Green-flag laps only. Substring containment, not equality — a lap's
    track_status can be a multi-digit code (e.g. '24' = yellow then SC)."""
    return laps["track_status"].astype(str).apply(lambda s: s == CLEAN_TRACK_STATUS)


def compute_laps_since_caution(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Per (race_id, lap_number): how many laps have elapsed since the most
    recent lap where ANY driver's track_status was non-green, anywhere in
    the field that lap. track_status is a race-wide condition (SC/VSC/red
    flag applies to everyone), so this is computed at the race+lap level,
    not per driver — same pattern as the tire model's `laps_since_sc`,
    derived from the FULL unfiltered lap history (not the training split).

    Why this matters here: `track_status == '1'` on the in/out-lap itself
    only tells you the flag was green THAT lap. It says nothing about
    whether the field is still bunched up / recovering pace 1-2 laps after
    a restart, which is exactly the kind of thing that inflates a pit
    stop's measured in/out-lap delta without the pit stop being the cause.
    Filtering (or feeding this in as a feature) on laps_since_caution, not
    just single-lap status, is the fix for that.

    The counter is capped at LAPS_SINCE_CAUTION_CAP. Without a cap, a
    race with no caution (or a stop long after the last one) produces a
    counter that just keeps climbing lap after lap — at that point it
    isn't measuring "distance from a caution" anymore, it's a
    thinly-disguised proxy for how late in the race the stop occurred
    (i.e. it re-derives race progress / stint_number through the back
    door). Past ~20 clean laps, "no caution recently" and "no caution at
    all this race" are indistinguishable in their effect on pit-loss, so
    they're collapsed into the same bucket instead of drifting apart as
    two different-looking numbers that mean the same thing.

    Returns a DataFrame with columns [race_id, lap_number, laps_since_caution].
    """
    race_lap_status = (
        laps.assign(_dirty=laps["track_status"].astype(str) != CLEAN_TRACK_STATUS)
        .groupby(["race_id", "lap_number"])["_dirty"]
        .any()
        .reset_index()
    )

    out_rows = []
    for race_id, grp in race_lap_status.groupby("race_id"):
        grp = grp.sort_values("lap_number")
        counter = LAPS_SINCE_CAUTION_CAP  # "no caution yet this race" starts already-capped,
                                           # i.e. treated the same as "comfortably clear"
        for _, row in grp.iterrows():
            if row["_dirty"]:
                counter = 0
            else:
                counter = min(counter + 1, LAPS_SINCE_CAUTION_CAP)
            out_rows.append(
                {"race_id": race_id, "lap_number": row["lap_number"], "laps_since_caution": counter}
            )
    return pd.DataFrame(out_rows)


def _reference_pace(laps: pd.DataFrame, race_id, driver_id, pit_lap: int) -> float | None:
    """
    Median lap time of the driver's own clean laps in a small window
    around the pit lap, EXCLUDING the in-lap and out-lap themselves.
    Using the driver's own nearby pace (not a field average) controls for
    that driver's specific pace level; using a small window (not the whole
    stint) controls for degradation drift within the stint.

    Reference laps must ALSO satisfy laps_since_caution >= MIN_LAPS_SINCE_CAUTION
    (not just single-lap track_status == '1') — a lap can be officially
    green while the field is still bunched from a restart 1 lap earlier,
    which would otherwise contaminate the "clean pace" baseline itself.

    Returns None if no clean reference laps are available (stop gets dropped).
    """
    driver_laps = laps[(laps.race_id == race_id) & (laps.driver_id == driver_id)]
    window = driver_laps[
        (driver_laps.lap_number >= pit_lap - REFERENCE_WINDOW)
        & (driver_laps.lap_number <= pit_lap + 1 + REFERENCE_WINDOW)
        & (~driver_laps.lap_number.isin([pit_lap, pit_lap + 1]))  # exclude in-lap & out-lap
    ]
    window = window[_clean_lap_mask(window)]
    window = window[window["laps_since_caution"] >= MIN_LAPS_SINCE_CAUTION]
    window = window.dropna(subset=["lap_time_ms"])
    if len(window) == 0:
        return None
    return float(window["lap_time_ms"].median())


def build_pit_events(laps: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pit stop, with the target `total_time_lost_ms` computed as:

        stop_duration_ms (stationary box time, from Ergast via pit_stops)
      + (in_lap_time_ms  - reference_pace_ms)
      + (out_lap_time_ms - reference_pace_ms)

    Convention: `pit_stops.lap_number` is the IN-lap (the lap the driver
    pits at the end of). Adjust the +1 offsets below if your ingest.py
    records it differently (e.g. as the out-lap).

    Rows are dropped (not imputed) when:
      - in-lap or out-lap lap_time_ms is missing
      - no clean reference laps exist nearby (can't establish baseline pace)
      - the in-lap or out-lap itself is under a safety car / VSC / red flag
        (time loss during a caution period isn't a normal pit-loss signal —
        same reasoning as the Phase 2 pit-stop-duration red-flag bug)
      - the in-lap or out-lap is within MIN_LAPS_SINCE_CAUTION laps of the
        most recent caution, even if officially marked green (SC/VSC
        restart-recovery bunching — see compute_laps_since_caution)

    `laps_since_caution` (measured at the in-lap) is kept as a column on
    surviving rows and exposed as a model feature — for the events that
    pass the filter but still occurred somewhat close to a caution period,
    this lets the model account for residual variance instead of treating
    it as noise.
    """
    laps = laps.copy()
    pit_stops = pit_stops.copy()

    caution_lookup = compute_laps_since_caution(laps)
    laps = laps.merge(caution_lookup, on=["race_id", "lap_number"], how="left")
    laps["laps_since_caution"] = laps["laps_since_caution"].fillna(LAPS_SINCE_CAUTION_CAP)

    rows = []
    for _, stop in pit_stops.iterrows():
        race_id, driver_id, in_lap = stop.race_id, stop.driver_id, int(stop.lap_number)
        out_lap = in_lap + 1

        driver_laps = laps[(laps.race_id == race_id) & (laps.driver_id == driver_id)]
        in_row = driver_laps[driver_laps.lap_number == in_lap]
        out_row = driver_laps[driver_laps.lap_number == out_lap]

        if in_row.empty or out_row.empty:
            continue
        if pd.isna(in_row.iloc[0]["lap_time_ms"]) or pd.isna(out_row.iloc[0]["lap_time_ms"]):
            continue
        # drop stops whose in/out lap happened under caution — not a clean signal
        if str(in_row.iloc[0]["track_status"]) != CLEAN_TRACK_STATUS:
            continue
        if str(out_row.iloc[0]["track_status"]) != CLEAN_TRACK_STATUS:
            continue
        # drop stops still inside the restart-recovery window, even if flagged green
        if in_row.iloc[0]["laps_since_caution"] < MIN_LAPS_SINCE_CAUTION:
            continue
        if out_row.iloc[0]["laps_since_caution"] < MIN_LAPS_SINCE_CAUTION:
            continue

        ref = _reference_pace(laps, race_id, driver_id, in_lap)
        if ref is None:
            continue

        in_delta = float(in_row.iloc[0]["lap_time_ms"]) - ref
        out_delta = float(out_row.iloc[0]["lap_time_ms"]) - ref
        total_ms = float(stop["stop_duration_ms"]) + in_delta + out_delta

        rows.append(
            {
                "race_id": race_id,
                "round": stop.get("round"),
                "circuit": stop.get("circuit"),
                "driver_id": driver_id,
                "team": stop.get("team"),
                "lap_number": in_lap,
                "stop_duration_ms": float(stop["stop_duration_ms"]),
                "in_lap_delta_ms": in_delta,
                "out_lap_delta_ms": out_delta,
                "total_time_lost_ms": total_ms,
                "laps_since_caution": float(in_row.iloc[0]["laps_since_caution"]),
            }
        )

    return pd.DataFrame(rows)


def add_stint_number(events: pd.DataFrame) -> pd.DataFrame:
    """Stint number = 1-indexed rank of this pit stop among the driver's
    stops in that race, ordered by lap. Stop 1 ends stint 1, etc."""
    events = events.sort_values(["race_id", "driver_id", "lap_number"]).copy()
    events["stint_number"] = (
        events.groupby(["race_id", "driver_id"]).cumcount() + 1
    )
    return events


# ---------------------------------------------------------------------------
# 2. Circuit target encoding (fit on train split ONLY — same discipline as
#    the tire degradation model's te_map / global_mean pattern)
# ---------------------------------------------------------------------------

def fit_circuit_target_encoding(train_events: pd.DataFrame) -> tuple[dict, float]:
    global_mean = float(train_events["total_time_lost_ms"].mean())
    te_map = (
        train_events.groupby("circuit")["total_time_lost_ms"].mean().to_dict()
    )
    return te_map, global_mean


def apply_circuit_encoding(events: pd.DataFrame, te_map: dict, global_mean: float) -> pd.Series:
    return events["circuit"].map(te_map).fillna(global_mean)


def fit_circuit_target_encoding_oof(
    train_events: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> pd.Series:
    """
    Out-of-fold circuit target encoding for the TRAINING rows only.

    The plain version above (fit on all of train, applied to all of train)
    lets a row see its own target baked into its own feature — mild
    self-leakage that flatters training fit without helping validation.
    Here, each row's circuit_te is computed from the OTHER folds' circuit
    means only, so no row ever sees its own target. Validation/inference
    still use the plain full-train te_map (fit_circuit_target_encoding) —
    OOF only matters for the rows the model is actually trained on.

    Returns a Series aligned to train_events.index.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = pd.Series(index=train_events.index, dtype=float)
    global_mean = float(train_events["total_time_lost_ms"].mean())

    for fold_train_idx, fold_val_idx in kf.split(train_events):
        fold_train = train_events.iloc[fold_train_idx]
        fold_val = train_events.iloc[fold_val_idx]
        means = fold_train.groupby("circuit")["total_time_lost_ms"].mean()
        result.iloc[fold_val_idx] = fold_val["circuit"].map(means).fillna(global_mean).values

    return result


def fit_arbitrary_circuit_id(train_events: pd.DataFrame) -> dict:
    """Arbitrary integer per circuit (alphabetical), no target information at
    all — the deliberately-weak baseline used to quantify how much target
    encoding is actually contributing (same comparison Phase 2 ran for
    the tire model's circuit encoding)."""
    circuits = sorted(train_events["circuit"].dropna().unique().tolist())
    return {c: i for i, c in enumerate(circuits)}


def apply_arbitrary_circuit_id(events: pd.DataFrame, id_map: dict) -> pd.Series:
    unknown_code = len(id_map)
    return events["circuit"].map(id_map).fillna(unknown_code)


def fit_team_encoding(train_events: pd.DataFrame) -> dict:
    """Simple label map for team, fit on train only. Unseen teams at
    inference fall back to a dedicated 'UNKNOWN' code (see model.py)."""
    teams = sorted(train_events["team"].dropna().unique().tolist())
    return {t: i for i, t in enumerate(teams)}


def apply_team_encoding(events: pd.DataFrame, team_map: dict) -> pd.Series:
    unknown_code = len(team_map)  # reserved code for unseen teams
    return events["team"].map(team_map).fillna(unknown_code)


# ---------------------------------------------------------------------------
# 3. Per-circuit baseline (historical mean pit loss) — the number the model
#    must beat. Computed on the TRAIN split only, so it's a fair comparison
#    against the model's honest MAE (both are "trained" on the same rows).
# ---------------------------------------------------------------------------

def circuit_mean_baseline(train_events: pd.DataFrame) -> tuple[dict, float]:
    global_mean = float(train_events["total_time_lost_ms"].mean())
    means = train_events.groupby("circuit")["total_time_lost_ms"].mean().to_dict()
    return means, global_mean


def predict_baseline(events: pd.DataFrame, means: dict, global_mean: float) -> pd.Series:
    return events["circuit"].map(means).fillna(global_mean)


# ---------------------------------------------------------------------------
# 4. Full feature matrix builder
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["circuit_te", "team_code", "stint_number", "laps_since_caution"]


def build_feature_matrix(
    events: pd.DataFrame,
    te_map: dict,
    global_mean: float,
    team_map: dict,
    circuit_te_override: pd.Series | None = None,
) -> pd.DataFrame:
    """
    circuit_te_override: if provided, used in place of apply_circuit_encoding
    (e.g. the OOF-encoded Series for training rows). Validation/inference
    should never pass this — they always use the plain full-train te_map.
    """
    out = pd.DataFrame(index=events.index)
    if circuit_te_override is not None:
        out["circuit_te"] = circuit_te_override
    else:
        out["circuit_te"] = apply_circuit_encoding(events, te_map, global_mean)
    out["team_code"] = apply_team_encoding(events, team_map)
    out["stint_number"] = events["stint_number"]
    out["laps_since_caution"] = events["laps_since_caution"]
    return out[FEATURE_COLUMNS]
