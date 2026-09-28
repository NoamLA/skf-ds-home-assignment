"""Load and validate timestamped temperature measurements."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "temp_c"]


@dataclass(frozen=True)
class SensorDataRequest:
    sensor_id: str
    start_datetime: datetime
    end_datetime: datetime

    def __post_init__(self) -> None:
        if not self.sensor_id or Path(self.sensor_id).name != self.sensor_id:
            raise ValueError("sensor_id must be a non-empty directory name")
        timezone_aware = (
            self.start_datetime.tzinfo is not None
            or self.end_datetime.tzinfo is not None
        )
        if timezone_aware:
            raise ValueError("start_datetime and end_datetime must be timezone-naive")
        if self.start_datetime >= self.end_datetime:
            raise ValueError("start_datetime must be before end_datetime")


@dataclass(frozen=True)
class PredictionRequest(SensorDataRequest):
    pass


def parse_datetime(value: str | datetime, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid ISO-8601 datetime") from error
    if parsed.tzinfo is not None:
        raise ValueError(f"{name} must be timezone-naive")
    return parsed


def load_sensor_data(
    data_dir: str | Path,
    request: SensorDataRequest,
) -> pd.DataFrame:
    sensor_dir = Path(data_dir) / request.sensor_id
    measurement_files = sorted(sensor_dir.glob("*/data.parquet"))
    if not measurement_files:
        raise ValueError(f"No data files found for sensor {request.sensor_id}")

    daily_measurements = []
    for measurement_file in measurement_files:
        measurements = pd.read_parquet(measurement_file)
        missing_columns = set(REQUIRED_COLUMNS) - set(measurements.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{measurement_file} is missing columns: {missing}")
        daily_measurements.append(measurements[REQUIRED_COLUMNS])

    sensor_data = pd.concat(daily_measurements, ignore_index=True)
    sensor_data["timestamp"] = pd.to_datetime(
        sensor_data["timestamp"], errors="coerce"
    )
    sensor_data["temp_c"] = pd.to_numeric(sensor_data["temp_c"], errors="coerce")
    if sensor_data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("timestamp and temp_c must contain valid values")
    if not np.isfinite(sensor_data["temp_c"].to_numpy(dtype=float)).all():
        raise ValueError("temp_c must contain finite values")
    if sensor_data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps are not supported")

    inside_interval = (
        (sensor_data["timestamp"] >= request.start_datetime)
        & (sensor_data["timestamp"] < request.end_datetime)
    )
    selected_data = (
        sensor_data.loc[inside_interval, REQUIRED_COLUMNS]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if selected_data.empty:
        raise ValueError("No measurements found inside the requested interval")
    return selected_data
