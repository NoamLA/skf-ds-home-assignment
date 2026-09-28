import numpy as np
import pandas as pd

from skf_anomaly.model import TemperatureAlarmDetector


def stable_measurements(periods: int = 720) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="h")
    hours = timestamps.hour.to_numpy()
    days = timestamps.dayofyear.to_numpy()
    temperatures = (
        20.0
        + 1.2 * np.sin(2 * np.pi * hours / 24)
        - 0.8 * np.cos(2 * np.pi * hours / 24)
        + 0.5 * np.sin(2 * np.pi * (days - 1) / 365.25)
    )
    return pd.DataFrame({"timestamp": timestamps, "temp_c": temperatures})


def test_predict_returns_one_readable_result_per_measurement() -> None:
    measurements = stable_measurements()
    detector = TemperatureAlarmDetector().fit(measurements.iloc[:480])

    predictions = detector.predict(measurements)

    assert len(predictions) == len(measurements)
    assert predictions.columns.tolist() == [
        "timestamp",
        "temp_c",
        "expected_temp_c",
        "residual_c",
        "rolling_residual_c",
        "alarm",
    ]
    assert predictions["alarm"].dtype == bool
    assert predictions.loc[:4, "rolling_residual_c"].isna().all()
    assert not predictions.loc[:4, "alarm"].any()


def test_isolated_spike_does_not_trigger_alarm() -> None:
    measurements = stable_measurements()
    detector = TemperatureAlarmDetector().fit(measurements.iloc[:480])
    measurements.loc[550, "temp_c"] += 100.0

    predictions = detector.predict(measurements)

    assert not predictions["alarm"].any()


def test_sustained_positive_residual_triggers_alarm() -> None:
    measurements = stable_measurements()
    detector = TemperatureAlarmDetector().fit(measurements.iloc[:480])
    measurements.loc[600:, "temp_c"] += 2.0

    predictions = detector.predict(measurements)

    assert predictions.loc[600:, "alarm"].any()


def test_sparse_candidates_more_than_24_hours_apart_do_not_confirm() -> None:
    measurements = stable_measurements().iloc[::13].reset_index(drop=True)
    detector = TemperatureAlarmDetector().fit(measurements)
    measurements.loc[10:, "temp_c"] += 2.0

    predictions = detector.predict(measurements)

    assert not predictions["alarm"].any()


def test_future_measurements_do_not_change_past_predictions() -> None:
    measurements = stable_measurements()
    detector = TemperatureAlarmDetector().fit(measurements.iloc[:480])
    original_predictions = detector.predict(measurements)
    modified_measurements = measurements.copy()
    modified_measurements.loc[600:, "temp_c"] += 50.0

    modified_predictions = detector.predict(modified_measurements)

    pd.testing.assert_frame_equal(
        original_predictions.iloc[:600],
        modified_predictions.iloc[:600],
    )
