# SKF Temperature Alarm Detection

This repository contains the analysis, model, and executable training operation for detecting persistent temperature increases. Every input measurement receives an `alarm` value.

## Repository contents

- `notebooks/analysis_and_model_selection.ipynb` — data analysis and chronological model selection.
- `src/skf_anomaly/` — data loading, model, training, and inference code.
- `artifacts/Stream_f0da779c-4b8f-419f-9a6a-6db2ffe13e03/` — a fitted model with metadata and a training log.
- `presentation/SKF_Model_Selection_Presentation.pptx` — slides summarizing the analysis and model selection.
- `data/` — the default local data location; raw measurements are ignored by Git.

The supplied Parquet data is excluded from Git.

## Model

The selected detector uses:

1. Huber regression with daily and annual sine/cosine features.
2. The difference between observed and expected temperature.
3. A causal 72-hour rolling median to ignore isolated spikes.
4. A `1°C` threshold confirmed by three consecutive candidates within 24 hours.

The data has no confirmed anomaly labels, so the notebook evaluates behavior chronologically instead of reporting supervised accuracy metrics.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Data location

Raw data is not stored in this repository. By default, commands look under
`data/<sensor-id>/<date>/data.parquet`. Alternatively, pass the directory that
contains the sensor folders with `--data-dir`:

```text
/path/to/data/
└── <sensor-id>/
    └── YYYY-MM-DD/
        └── data.parquet
```

## Train

```bash
python -m skf_anomaly.train \
  --sensor-id Stream_f0da779c-4b8f-419f-9a6a-6db2ffe13e03 \
  --start-datetime 2023-01-01T00:00:00 \
  --end-datetime 2023-10-01T00:00:00 \
  --data-dir /path/to/Machine_e6db14f9-d562-45cd-90e1-ab396592835e \
  --output-dir artifacts
```

`--data-dir` and `--output-dir` default to `data` and `artifacts`. Intervals are start-inclusive and end-exclusive.
The three required training inputs are represented internally by the
`SensorDataRequest` dataclass.

Training reports four stages to the terminal and `training.log`:

```text
load_data
fit_model
calculate_diagnostics
save_artifact
```

The output directory contains:

- `model.joblib` — the fitted detector and its metadata.
- `metadata.json` — readable training interval, parameters, and diagnostics.
- `training.log` — stage outcomes and a traceback if training fails.

## Predict

```bash
python -m skf_anomaly.predict \
  --model-path artifacts/Stream_f0da779c-4b8f-419f-9a6a-6db2ffe13e03/model.joblib \
  --sensor-id Stream_f0da779c-4b8f-419f-9a6a-6db2ffe13e03 \
  --start-datetime 2024-01-01T00:00:00 \
  --end-datetime 2025-01-01T00:00:00 \
  --data-dir /path/to/Machine_e6db14f9-d562-45cd-90e1-ab396592835e \
  --output-path predictions.parquet
```

The included example model can be used without retraining. Inference still
requires the measurements to classify. Its three sensor/time inputs are stored
in a `PredictionRequest` dataclass.

The output keeps every requested row and adds the expected temperature,
residual, rolling residual, and Boolean alarm. The detector uses only the
current and earlier measurements. Batch inference loads 96 hours before the
requested interval so the rolling state is available without using future data.

## Tests

```bash
python -m pytest
```

The focused tests cover row-level output, causality, spike resistance, persistent alarms, artifact reuse, stage visibility, failure troubleshooting, and inference history.

## Limitations

- The threshold was selected from one unlabeled machine and should be validated against additional machines and maintenance outcomes.
- The suspected late-2023 and late-2024 periods are contextual evidence, not confirmed fault labels.
- A streaming deployment would retain recent per-sensor history; this assignment implements batch inference.
