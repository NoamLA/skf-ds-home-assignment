"""Executable training pipeline for the temperature alarm detector."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from skf_anomaly.data import SensorDataRequest, load_sensor_data, parse_datetime
from skf_anomaly.model import DetectorConfig, TemperatureAlarmDetector


def train_model(
    request: SensorDataRequest,
    data_dir: str | Path = "data",
    output_dir: str | Path = "artifacts",
    config: DetectorConfig | None = None,
) -> Path:
    artifact_dir = Path(output_dir) / request.sensor_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(artifact_dir / "training.log")
    current_stage = "load_data"

    try:
        logger.info("%s STARTED", current_stage)
        training_data = load_sensor_data(data_dir, request)
        logger.info("%s COMPLETED rows=%d", current_stage, len(training_data))

        current_stage = "fit_model"
        logger.info("%s STARTED", current_stage)
        detector = TemperatureAlarmDetector(config).fit(training_data)
        logger.info("%s COMPLETED", current_stage)

        current_stage = "calculate_diagnostics"
        logger.info("%s STARTED", current_stage)
        predictions = detector.predict(training_data)
        residuals = predictions["residual_c"]
        residual_median = float(residuals.median())
        diagnostics = {
            "median_absolute_residual_c": float(residuals.abs().median()),
            "residual_mad_c": float(
                (residuals - residual_median).abs().median()
            ),
            "huber_scale": float(detector.baseline_model.scale_),
            "training_alarm_rate": float(predictions["alarm"].mean()),
        }
        logger.info(
            "%s COMPLETED residual_mad_c=%.3f",
            current_stage,
            diagnostics["residual_mad_c"],
        )

        metadata = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "sensor_id": request.sensor_id,
            "requested_start_datetime": request.start_datetime.isoformat(),
            "requested_end_datetime": request.end_datetime.isoformat(),
            "data_start_datetime": training_data["timestamp"].min().isoformat(),
            "data_end_datetime": training_data["timestamp"].max().isoformat(),
            "measurement_count": len(training_data),
            "model_parameters": asdict(detector.config),
            "diagnostics": diagnostics,
        }

        current_stage = "save_artifact"
        logger.info("%s STARTED", current_stage)
        joblib.dump(
            {"model": detector, "metadata": metadata},
            artifact_dir / "model.joblib",
        )
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        logger.info("%s COMPLETED path=%s", current_stage, artifact_dir)
        return artifact_dir
    except Exception:
        logger.exception("%s FAILED", current_stage)
        raise
    finally:
        close_logger(logger)


def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"skf_anomaly.training.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the temperature alarm model.")
    parser.add_argument("--sensor-id", required=True)
    parser.add_argument("--start-datetime", required=True)
    parser.add_argument("--end-datetime", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parsed = parser.parse_args(arguments)

    try:
        request = SensorDataRequest(
            sensor_id=parsed.sensor_id,
            start_datetime=parse_datetime(parsed.start_datetime, "start_datetime"),
            end_datetime=parse_datetime(parsed.end_datetime, "end_datetime"),
        )
        artifact_dir = train_model(
            request=request,
            data_dir=parsed.data_dir,
            output_dir=parsed.output_dir,
        )
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1

    print(f"Training completed: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
