"""Batch inference using a saved temperature alarm model."""

import argparse
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd

from skf_anomaly.data import (
    PredictionRequest,
    SensorDataRequest,
    load_sensor_data,
    parse_datetime,
)


def generate_predictions(
    request: PredictionRequest,
    model_path: str | Path,
    data_dir: str | Path,
) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    detector = artifact["model"]
    metadata = artifact["metadata"]
    if metadata["sensor_id"] != request.sensor_id:
        trained_sensor_id = metadata["sensor_id"]
        raise ValueError(
            f"Model was trained for {trained_sensor_id}, not {request.sensor_id}"
        )

    context_hours = (
        detector.config.rolling_hours + detector.config.confirmation_hours
    )
    context_request = SensorDataRequest(
        sensor_id=request.sensor_id,
        start_datetime=request.start_datetime - timedelta(hours=context_hours),
        end_datetime=request.end_datetime,
    )
    sensor_data = load_sensor_data(data_dir, context_request)
    predictions = detector.predict(sensor_data)
    requested_rows = (
        (predictions["timestamp"] >= request.start_datetime)
        & (predictions["timestamp"] < request.end_datetime)
    )
    selected_predictions = predictions.loc[requested_rows].reset_index(drop=True)
    if selected_predictions.empty:
        raise ValueError("No measurements found inside the prediction interval")
    return selected_predictions


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate alarm predictions.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sensor-id", required=True)
    parser.add_argument("--start-datetime", required=True)
    parser.add_argument("--end-datetime", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--data-dir", default="data")
    parsed = parser.parse_args(arguments)

    try:
        output_path = Path(parsed.output_path)
        if output_path.suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("output_path must end with .csv or .parquet")
        request = PredictionRequest(
            sensor_id=parsed.sensor_id,
            start_datetime=parse_datetime(parsed.start_datetime, "start_datetime"),
            end_datetime=parse_datetime(parsed.end_datetime, "end_datetime"),
        )
        predictions = generate_predictions(
            request=request,
            model_path=parsed.model_path,
            data_dir=parsed.data_dir,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            predictions.to_csv(output_path, index=False)
        else:
            predictions.to_parquet(output_path, index=False)
    except Exception as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1

    print(f"Predictions written: {output_path} rows={len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
