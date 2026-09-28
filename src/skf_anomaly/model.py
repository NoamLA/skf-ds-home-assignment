"""Robust temperature baseline and causal alarm detector."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


@dataclass(frozen=True)
class DetectorConfig:
    rolling_hours: int = 72
    minimum_observations: int = 6
    alarm_threshold_c: float = 1.0
    confirmation_readings: int = 3
    confirmation_hours: int = 24
    huber_max_iter: int = 1000


def create_time_features(sensor_data: pd.DataFrame) -> pd.DataFrame:
    timestamps = sensor_data["timestamp"]
    hours = timestamps.dt.hour + (timestamps.dt.minute / 60)
    days = timestamps.dt.dayofyear
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
            "year_sin": np.sin(2 * np.pi * (days - 1) / 365.25),
            "year_cos": np.cos(2 * np.pi * (days - 1) / 365.25),
        },
        index=sensor_data.index,
    )


class TemperatureAlarmDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.baseline_model: HuberRegressor | None = None

    def fit(self, training_data: pd.DataFrame) -> "TemperatureAlarmDetector":
        self.baseline_model = HuberRegressor(
            max_iter=self.config.huber_max_iter,
            fit_intercept=True,
        ).fit(create_time_features(training_data), training_data["temp_c"])
        return self

    def predict(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        if self.baseline_model is None:
            raise RuntimeError("The detector must be fitted before prediction")

        predictions = (
            sensor_data[["timestamp", "temp_c"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
            .copy()
        )
        predictions["expected_temp_c"] = self.baseline_model.predict(
            create_time_features(predictions)
        )
        predictions["residual_c"] = (
            predictions["temp_c"] - predictions["expected_temp_c"]
        )

        indexed_residuals = predictions.set_index("timestamp")["residual_c"]
        predictions["rolling_residual_c"] = indexed_residuals.rolling(
            f"{self.config.rolling_hours}h",
            min_periods=self.config.minimum_observations,
        ).median().to_numpy()
        alarm_candidates = (
            predictions["rolling_residual_c"] > self.config.alarm_threshold_c
        )
        consecutive_candidates = alarm_candidates.astype(int).rolling(
            self.config.confirmation_readings,
            min_periods=self.config.confirmation_readings,
        ).sum()
        confirmation_span = predictions["timestamp"] - predictions[
            "timestamp"
        ].shift(self.config.confirmation_readings - 1)
        predictions["alarm"] = (
            consecutive_candidates.eq(self.config.confirmation_readings)
            & (
                confirmation_span
                <= pd.Timedelta(hours=self.config.confirmation_hours)
            )
        ).fillna(False).astype(bool)
        return predictions
