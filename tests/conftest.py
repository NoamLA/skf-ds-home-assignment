from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def write_sensor_data(
    tmp_path: Path,
) -> Callable[[pd.DataFrame, str], Path]:
    def write_partitions(
        measurements: pd.DataFrame,
        sensor_id: str = "sensor-a",
    ) -> Path:
        partition_dates = measurements["timestamp"].dt.strftime("%Y-%m-%d")
        for partition_date, daily_measurements in measurements.groupby(
            partition_dates
        ):
            partition_directory = tmp_path / sensor_id / partition_date
            partition_directory.mkdir(parents=True, exist_ok=True)
            daily_measurements.to_parquet(
                partition_directory / "data.parquet",
                index=False,
            )
        return tmp_path

    return write_partitions
