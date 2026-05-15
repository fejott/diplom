"""
ERA5 Pre-training script for the LSTM weather forecaster.

Downloads 5 years of hourly ERA5 reanalysis data (temperature, relative
humidity, surface pressure) from the Open-Meteo Historical Weather API for
the station coordinates, trains the base LSTM, and saves the weights as the
warm-start checkpoint.

Why this approach works
-----------------------
The live LSTM (lstm_forecast.py) uses:
    - SEQUENCE_LENGTH = 12  (12 five-minute buckets = 1 h context)
    - FORECAST_STEPS  = [12, 24, 36]  (→ +1 h / +2 h / +3 h)

This script trains on HOURLY ERA5 data using:
    - SEQUENCE_LENGTH = 12  (12 hourly readings = 12 h context)
    - _STEPS_ERA5     = [1, 2, 3]   (hourly offsets → +1 h / +2 h / +3 h)

The LSTM and Dense layer shapes are IDENTICAL (determined only by
SEQUENCE_LENGTH and n_features=3, not by time spacing), so the ERA5 weights
transfer to the live model as a warm start.  The first local retraining then
adapts the model to 5-minute-bucket resolution and the station microclimate.

Usage (from weather_station/):
    python research/era5_pretrain.py
    python research/era5_pretrain.py --years 3   # shorter history
    python research/era5_pretrain.py --lat 59.83 --lon 30.36
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import config
from utils.logger import get_logger

logger = get_logger("research.era5_pretrain")

# ── Default station coordinates (from GPS fix) ────────────────────────────────
DEFAULT_LAT   = 59.829457
DEFAULT_LON   = 30.364461
DEFAULT_YEARS = 5

# ERA5 pre-training temporal parameters
_SEQ_LEN    = config.SEQUENCE_LENGTH   # 12 hourly steps = 12 h context
_STEPS_ERA5 = [1, 2, 3]               # hourly offsets → +1 h / +2 h / +3 h
_N_STEPS    = len(_STEPS_ERA5)         # 3


# ── Download ──────────────────────────────────────────────────────────────────

def download_era5(lat: float, lon: float, years: int) -> dict:
    """Fetch hourly ERA5 reanalysis from the Open-Meteo Historical API.

    Returns the raw JSON response dict.  No API key required.
    """
    import requests
    from datetime import date, timedelta

    # ERA5 reanalysis has a ~5-day processing lag on Open-Meteo
    end   = date.today() - timedelta(days=5)
    start = date(end.year - years, end.month, end.day)

    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "hourly":     "temperature_2m,relative_humidity_2m,surface_pressure",
        "timezone":   "UTC",
    }

    print(f"  Координаты:  {lat:.6f}°N  {lon:.6f}°E")
    print(f"  Период ERA5: {start} → {end}  ({years} лет)")
    print("  Загрузка... ", end="", flush=True)

    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()

    kb = len(r.content) // 1024
    print(f"получено {kb} КБ")
    return r.json()


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_era5(raw: dict) -> np.ndarray:
    """Convert API JSON response to a (N, 3) float32 array [temp, hum, pres].

    Rows where any variable is null are dropped.
    """
    h    = raw["hourly"]
    temp = h["temperature_2m"]
    hum  = h["relative_humidity_2m"]
    pres = h["surface_pressure"]
    n    = len(temp)

    rows = []
    n_null = 0
    for i in range(n):
        t, hu, p = temp[i], hum[i], pres[i]
        if t is None or hu is None or p is None:
            n_null += 1
            continue
        rows.append([float(t), float(hu), float(p)])

    data = np.array(rows, dtype=np.float32)
    if n_null:
        print(f"  Пропущено null-значений: {n_null}")
    print(f"  ERA5 записей: {len(data):,}  "
          f"(≈ {len(data) / 24 / 365.25:.1f} лет почасовых данных)")
    return data


# ── Filter (same logic as LSTMForecaster._filter_data) ───────────────────────

def _filter_data(data: np.ndarray) -> np.ndarray:
    result = data.astype(np.float64)

    bounds = [
        (config.FILTER_TEMP_MIN,      config.FILTER_TEMP_MAX),
        (0.0,                          100.0),
        (config.FILTER_PRESSURE_MIN,   config.FILTER_PRESSURE_MAX),
    ]
    for col, (lo, hi) in enumerate(bounds):
        mask = (result[:, col] < lo) | (result[:, col] > hi)
        result[mask, col] = np.nan

    k = config.FILTER_IQR_MULTIPLIER
    for col in range(result.shape[1]):
        vals  = result[:, col]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 4:
            continue
        q1, q3 = np.percentile(valid, 25), np.percentile(valid, 75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        spikes = (vals < q1 - k * iqr) | (vals > q3 + k * iqr)
        result[spikes, col] = np.nan

    indices = np.arange(len(result))
    for col in range(result.shape[1]):
        vals = result[:, col]
        nans = np.isnan(vals)
        if not nans.any():
            continue
        good = ~nans
        if good.sum() < 2:
            continue
        result[nans, col] = np.interp(indices[nans], indices[good], vals[good])

    valid_rows = ~np.isnan(result).any(axis=1)
    n_dropped  = len(data) - valid_rows.sum()
    if n_dropped:
        print(f"  Отфильтровано выбросов: {n_dropped}")
    return result[valid_rows].astype(np.float32)


# ── Sequence builder ──────────────────────────────────────────────────────────

def _build_sequences(norm: np.ndarray, seq: int, steps: list[int]):
    X_list, y_list = [], []
    max_step = max(steps)
    for i in range(len(norm) - seq - max_step):
        X_list.append(norm[i: i + seq])
        targets = []
        for st in steps:
            targets.extend(norm[i + seq + st - 1])
        y_list.append(targets)
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32))


# ── Train & save ──────────────────────────────────────────────────────────────

def train_on_era5(data: np.ndarray) -> None:
    """Scale, build sequences, train LSTM, save weights and scaler."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        print(f"  TensorFlow недоступен: {exc}")
        sys.exit(1)

    print(f"\n  TensorFlow {tf.__version__}")

    # ── Scaler ────────────────────────────────────────────────────────────────
    s_min = data.min(axis=0)
    s_max = data.max(axis=0)
    rng   = s_max - s_min
    rng[rng == 0] = 1.0
    norm  = (data - s_min) / rng

    print("  Диапазон данных ERA5:")
    for lbl, lo, hi in zip(
        ["Темп (°C)", "Влаж. (%)", "Давл. (hPa)"],
        s_min, s_max,
    ):
        print(f"    {lbl:<14} {lo:.1f} – {hi:.1f}")

    # ── Sequences ─────────────────────────────────────────────────────────────
    X, y = _build_sequences(norm, _SEQ_LEN, _STEPS_ERA5)
    n_seq = len(X)
    print(f"\n  Последовательностей: {n_seq:,}")
    if n_seq < 100:
        print("  Слишком мало последовательностей — выход.")
        sys.exit(1)

    split      = int(n_seq * 0.8)
    X_tr, X_v = X[:split], X[split:]
    y_tr, y_v = y[:split], y[split:]
    print(f"  Обучение: {len(X_tr):,}  /  Валидация: {len(X_v):,}")

    # ── Model (identical architecture to LSTMForecaster) ─────────────────────
    n_out = 3 * _N_STEPS   # 9 outputs — same as live model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(_SEQ_LEN, 3)),
        tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(config.LSTM_UNITS),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_out),
    ])
    model.compile(optimizer="adam", loss="mse")

    print()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5,
            verbose=1,
        ),
    ]

    print("\n  Обучение (может занять 5–15 мин на Pi 4)...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=100,
        batch_size=512,   # larger batches → faster epochs on big ERA5 dataset
        callbacks=callbacks,
        verbose=1,
    )

    best_val_loss = float(min(history.history["val_loss"]))
    print(f"\n  Лучший val_loss: {best_val_loss:.4f}")

    # ── MAE breakdown ─────────────────────────────────────────────────────────
    y_pred_norm = model.predict(X_v, batch_size=64, verbose=0)
    rng_tiled   = np.tile(rng,   _N_STEPS)
    min_tiled   = np.tile(s_min, _N_STEPS)
    y_pred_d    = y_pred_norm * rng_tiled + min_tiled
    y_v_d       = y_v         * rng_tiled + min_tiled

    print("\n  MAE на валидации (ERA5 часовой горизонт):")
    for i, lbl in enumerate(["+1ч", "+2ч", "+3ч"]):
        mae_t = float(np.mean(np.abs(y_pred_d[:, i * 3 + 0] - y_v_d[:, i * 3 + 0])))
        mae_p = float(np.mean(np.abs(y_pred_d[:, i * 3 + 2] - y_v_d[:, i * 3 + 2])))
        print(f"    {lbl}  Темп: {mae_t:.2f}°C   Давл: {mae_p:.2f} hPa")

    # ── Save scaler ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.SCALER_PATH) or ".", exist_ok=True)
    with open(config.SCALER_PATH, "w") as fh:
        json.dump({"min": s_min.tolist(), "max": s_max.tolist()}, fh)
    print(f"\n  Скейлер сохранён  → {config.SCALER_PATH}")

    # ── Save weights (same .npz format as LSTMForecaster) ────────────────────
    weights_path = config.WEIGHTS_PATH
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    np.savez(weights_path, *model.get_weights())
    print(f"  Веса сохранены    → {weights_path}")

    # ── Save a metrics stub so correction model knows the deployment time ─────
    import datetime
    try:
        mae_t1 = float(np.mean(np.abs(y_pred_d[:, 0] - y_v_d[:, 0])))
        mae_p1 = float(np.mean(np.abs(y_pred_d[:, 2] - y_v_d[:, 2])))
        metrics_stub = {
            "source":      "era5_pretrain",
            "val_loss":    best_val_loss,
            "mae_temp_1h": mae_t1,
            "mae_pres_1h": mae_p1,
            "trained_at":  datetime.datetime.utcnow().isoformat(),
        }
        os.makedirs(os.path.dirname(config.METRICS_PATH) or ".", exist_ok=True)
        with open(config.METRICS_PATH, "w") as fh:
            json.dump(metrics_stub, fh, indent=2)
        print(f"  Метрики сохранены → {config.METRICS_PATH}")
    except Exception as exc:
        print(f"  Предупреждение: не удалось сохранить метрики: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train LSTM on ERA5 reanalysis before local fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat",   type=float, default=DEFAULT_LAT,
                        help="Station latitude")
    parser.add_argument("--lon",   type=float, default=DEFAULT_LON,
                        help="Station longitude")
    parser.add_argument("--years", type=int,   default=DEFAULT_YEARS,
                        help="Years of ERA5 history to download")
    args = parser.parse_args()

    print("═" * 60)
    print("  ERA5 ПРЕДОБУЧЕНИЕ LSTM")
    print("═" * 60)

    raw  = download_era5(args.lat, args.lon, args.years)
    data = parse_era5(raw)
    data = _filter_data(data)

    min_needed = _SEQ_LEN + max(_STEPS_ERA5) + 100
    if len(data) < min_needed:
        print(f"  Недостаточно данных: {len(data)} < {min_needed}")
        sys.exit(1)

    train_on_era5(data)

    print("\n" + "═" * 60)
    print("  Предобучение завершено.")
    print("  Следующие шаги:")
    print("    1. pkill -f main.py")
    print("    2. python research/calibration_cli.py rollback-correction")
    print("    3. python main.py &")
    print("  Первый локальный ретрейн использует ERA5-веса как тёплый старт.")
    print("═" * 60)


if __name__ == "__main__":
    main()
