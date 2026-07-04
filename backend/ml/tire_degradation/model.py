
from pathlib import Path
import joblib
import pandas as pd

from ml.tire_degradation.features import build_inference_row  # single source of truth

MODEL_PATH = Path(__file__).parent / "model.pkl"


class TireDegradationModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.te_map = bundle["te_map"]
        self.global_mean = bundle["global_mean"]
        self.feature_cols = bundle["feature_cols"]

    def predict(self, features: dict) -> float:
        """features: {"compound": "MEDIUM", "tire_age": 15, "lap_number": 30,
                       "circuit": "Bahrain International Circuit",
                       "ambient_temp_c": 28.0, "track_temp_c": 42.0,
                       "laps_since_sc": 20}
        laps_since_sc: how many laps since the most recent non-green track
        status in the CURRENT race. Pass 20 (or omit) if there has been no
        safety car / VSC yet this race.

        Returns predicted lap_time_delta in milliseconds vs. that compound's
        fresh-tire baseline at that circuit.

        Note on unseen circuits: an unseen `circuit` does NOT raise — it
        falls back to the global training mean, matching how validation rows
        with an unseen circuit were handled during training.
        """
        row = build_inference_row(features, self.te_map, self.global_mean, self.feature_cols)
        pred = self.model.predict(pd.DataFrame([row], columns=self.feature_cols))
        return float(pred[0])
