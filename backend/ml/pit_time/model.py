
from __future__ import annotations

import os
import pickle
import pandas as pd

from features import FEATURE_COLUMNS, LAPS_SINCE_CAUTION_CAP

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


class PitTimeLossModel:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._model = bundle["model"]
        self._te_map = bundle["te_map"]
        self._global_mean = bundle["global_mean"]
        self._team_map = bundle["team_map"]
        # Bundles produced by the current train.py always include this; the
        # fallback exists only so an old model.pkl (trained before this
        # column was added) doesn't hard-crash on load — it'll just be
        # missing the caution feature until retrained.
        self._feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)

    def predict(self, features: dict) -> float:
        """
        features: {
            "circuit": str,                  # e.g. "Monza" — falls back to global mean if unseen
            "team": str,                      # e.g. "Red Bull" — falls back to reserved UNKNOWN code if unseen
            "stint_number": int,              # 1-indexed
            "laps_since_caution": int,        # optional. How many clean laps since the most
                                               # recent caution *at the moment of this stop*.
                                               # Defaults to LAPS_SINCE_CAUTION_CAP (i.e. "assume
                                               # conditions are clear") if the caller doesn't know
                                               # or doesn't track this. Values above the cap are
                                               # clamped down to it, matching how this feature was
                                               # capped at training time (see features.py) — the
                                               # model was never shown values above the cap, so it
                                               # has no idea what to do with them.
        }
        Returns predicted total time lost to the pit stop, in milliseconds.
        """
        circuit_te = self._te_map.get(features["circuit"], self._global_mean)
        team_code = self._team_map.get(features["team"], len(self._team_map))
        raw_laps_since_caution = features.get("laps_since_caution", LAPS_SINCE_CAUTION_CAP)
        laps_since_caution = min(raw_laps_since_caution, LAPS_SINCE_CAUTION_CAP)

        row_values = {
            "circuit_te": circuit_te,
            "team_code": team_code,
            "stint_number": features["stint_number"],
            "laps_since_caution": laps_since_caution,
        }
        row = pd.DataFrame(
            [[row_values[col] for col in self._feature_columns]],
            columns=self._feature_columns,
        )
        return float(self._model.predict(row)[0])
