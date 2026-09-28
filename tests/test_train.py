from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from skf_anomaly.data import PredictionRequest, SensorDataRequest
from skf_anomaly.model import TemperatureAlarmDetector
from skf_anomaly.predict import generate_predictions, main as predict_main
from skf_anomaly.train import main as train_main, train_model


def stable_measurements(periods: int = 4800) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="h")
    temperatures = 20.0 + np.sin(2 * np.pi * timestamps.hour / 24)
    return pd.DataFrame({"timestamp": timestamps, "temp_c": temperatures})


def train_example(
    tmp_path: Path,
    write_sensor_data,
) -> tuple[Path, Path, pd.DataFrame]:
    measurements = stable_measurements()
    data_dir = write_sensor_data(measurements, "sensor-a")
    output_dir = tmp_path / "artifacts"
    request = SensorDataRequest(
        sensor_id="sensor-a",
        start_datetime=datetime.fromisoformat("2023-01-01T00:00:00"),
        end_datetime=datetime.fromisoformat("2023-07-02T00:00:00"),
    )
    artifact_dir = train_model(
        request=request,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    return data_dir, artifact_dir, measurements


def test_training_creates_simple_reusable_artifact(
    tmp_path: Path,
    write_sensor_data,
) -> None:
    _, artifact_dir, measurements = train_example(tmp_path, write_sensor_data)

    assert is_dataclass(SensorDataRequest)
    assert is_dataclass(PredictionRequest)
    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "metadata.json",
        "model.joblib",
        "training.log",
    ]
    artifact = joblib.load(artifact_dir / "model.joblib")
    assert set(artifact) == {"model", "metadata"}
    assert artifact["metadata"]["sensor_id"] == "sensor-a"
    assert artifact["metadata"]["measurement_count"] == 4368
    assert len(artifact["model"].predict(measurements.iloc[:24])) == 24

    log_text = (artifact_dir / "training.log").read_text()
    for stage in ("load_data", "fit_model", "calculate_diagnostics", "save_artifact"):
        assert f"{stage} STARTED" in log_text
        assert f"{stage} COMPLETED" in log_text


def test_training_failure_records_stage_and_traceback(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"

    exit_code = train_main(
        [
            "--data-dir",
            str(tmp_path / "missing-data"),
            "--output-dir",
            str(output_dir),
            "--sensor-id",
            "sensor-a",
            "--start-datetime",
            "2023-01-01T00:00:00",
            "--end-datetime",
            "2023-02-01T00:00:00",
        ]
    )

    assert exit_code == 1
    log_text = (output_dir / "sensor-a" / "training.log").read_text()
    assert "load_data FAILED" in log_text
    assert "Traceback" in log_text


def test_prediction_cli_writes_one_result_per_requested_measurement(
    tmp_path: Path,
    write_sensor_data,
) -> None:
    data_dir, artifact_dir, _ = train_example(tmp_path, write_sensor_data)
    output_path = tmp_path / "predictions.parquet"

    exit_code = predict_main(
        [
            "--model-path",
            str(artifact_dir / "model.joblib"),
            "--data-dir",
            str(data_dir),
            "--sensor-id",
            "sensor-a",
            "--start-datetime",
            "2023-06-01T00:00:00",
            "--end-datetime",
            "2023-06-02T00:00:00",
            "--output-path",
            str(output_path),
        ]
    )

    assert exit_code == 0
    predictions = pd.read_parquet(output_path)
    assert len(predictions) == 24
    assert predictions.columns.tolist() == [
        "timestamp",
        "temp_c",
        "expected_temp_c",
        "residual_c",
        "rolling_residual_c",
        "alarm",
    ]


class ZeroBaseline:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features))


def test_prediction_includes_enough_history_for_alarm_state(
    tmp_path: Path,
    write_sensor_data,
) -> None:
    requested_start = pd.Timestamp("2024-01-05 00:00:00")
    timestamps = pd.date_range(
        requested_start - pd.Timedelta(hours=100),
        requested_start + pd.Timedelta(hours=2),
        freq="h",
    )
    measurements = pd.DataFrame(
        {"timestamp": timestamps, "temp_c": np.zeros(len(timestamps))}
    )
    for hour_offset in range(-71, -35):
        measurements.loc[
            measurements["timestamp"]
            == requested_start + pd.Timedelta(hours=hour_offset),
            "temp_c",
        ] = 2.0
    for hour_offset in (-1, 0):
        measurements.loc[
            measurements["timestamp"]
            == requested_start + pd.Timedelta(hours=hour_offset),
            "temp_c",
        ] = 2.0

    detector = TemperatureAlarmDetector()
    detector.baseline_model = ZeroBaseline()
    artifact = {
        "model": detector,
        "metadata": {"sensor_id": "sensor-a"},
    }
    artifact_dir = tmp_path / "artifacts" / "sensor-a"
    artifact_dir.mkdir(parents=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(artifact, model_path)
    data_dir = write_sensor_data(measurements, "sensor-a")
    request = PredictionRequest(
        sensor_id="sensor-a",
        start_datetime=requested_start.to_pydatetime(),
        end_datetime=(requested_start + pd.Timedelta(hours=1)).to_pydatetime(),
    )
    actual = generate_predictions(
        request=request,
        model_path=model_path,
        data_dir=data_dir,
    )

    expected = detector.predict(measurements).loc[
        lambda predictions: predictions["timestamp"] == requested_start
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
