"""
Residual Correction Model — a small feedforward network that learns to correct
the systematic bias of the base LSTM forecast using locally accumulated
verified forecast errors.

Inference is pure numpy (no TF dependency at runtime).
TensorFlow is imported only inside train().

Architecture:
    11 inputs → Dense(32, ReLU) → Dropout(0.1) → Dense(16, ReLU) → Dense(6, linear)

Inputs (11):
    temp_1h, temp_2h, temp_3h, pres_1h, pres_2h, pres_3h,
    pressure_trend, sin_hour, cos_hour, sin_dow, cos_dow

Outputs (6):
    delta_temp_1h, delta_temp_2h, delta_temp_3h,
    delta_pres_1h, delta_pres_2h, delta_pres_3h
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

import config
from utils.logger import get_logger

logger = get_logger("forecasting.correction")


@dataclass
class CorrectionResult:
    """Outcome of a correction model training run."""
    success:     bool
    n_samples:   int
    mae_before:  Optional[float]   # MAE temp +1h before correction
    mae_after:   Optional[float]   # MAE temp +1h after correction (on held-out set)
    message:     str


class CorrectionModel:
    """Residual correction model for base LSTM forecasts.

    Pure numpy at inference time.  TF is imported only inside train().
    Weights are stored as a .npz file; scaler as JSON.
    """

    def __init__(self) -> None:
        self._w1: Optional[np.ndarray] = None
        self._b1: Optional[np.ndarray] = None
        self._w2: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None
        self._w3: Optional[np.ndarray] = None
        self._b3: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std:  Optional[np.ndarray] = None
        self._loaded = False
        self._try_load()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True when weights are loaded and inference can run."""
        return self._loaded

    def can_train(self, research_db_path: str) -> tuple[bool, str]:
        """Check whether enough verified rows exist to train.

        Returns:
            (ok, message) — ok is True when training is possible.
        """
        try:
            with sqlite3.connect(research_db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM forecast_verification fv "
                    "JOIN forecast_log fl ON fv.forecast_id = fl.id "
                    "WHERE fv.signed_error_temp_1h IS NOT NULL "
                    "  AND fl.mode IN ('lstm', 'lstm_corrected')"
                ).fetchone()
            n = row["n"] if row else 0
            need = config.CORRECTION_MIN_VERIFIED
            if n >= need:
                return True, f"Достаточно данных: {n} / {need}"
            return False, f"Недостаточно данных: {n} / {need}"
        except Exception as exc:
            return False, f"Ошибка проверки БД: {exc}"

    def train(self, research_db_path: str) -> CorrectionResult:
        """Train the correction network on locally accumulated errors.

        Args:
            research_db_path: Path to research_data.db.

        Returns:
            CorrectionResult with before/after MAE and success flag.
        """
        try:
            import tensorflow as tf
        except Exception as exc:
            return CorrectionResult(False, 0, None, None,
                                    f"TensorFlow недоступен: {exc}")

        # ── Load data ────────────────────────────────────────────────────────
        try:
            with sqlite3.connect(research_db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT "
                    "  fl.temp_1h, fl.temp_2h, fl.temp_3h, "
                    "  fl.pressure_1h, fl.pressure_2h, fl.pressure_3h, "
                    "  fl.pressure_trend, fl.timestamp, "
                    "  fv.signed_error_temp_1h, fv.signed_error_temp_2h, fv.signed_error_temp_3h, "
                    "  fv.signed_error_pres_1h, fv.signed_error_pres_2h, fv.signed_error_pres_3h "
                    "FROM forecast_verification fv "
                    "JOIN forecast_log fl ON fv.forecast_id = fl.id "
                    "WHERE fv.signed_error_temp_1h IS NOT NULL "
                    "  AND fl.mode IN ('lstm', 'lstm_corrected') "
                    "ORDER BY fl.timestamp"
                ).fetchall()
        except Exception as exc:
            return CorrectionResult(False, 0, None, None,
                                    f"Ошибка чтения БД: {exc}")

        # Keep only forecasts from the training period (exclude held-out data)
        cutoff_ts = self._get_training_cutoff_ts()
        if cutoff_ts:
            rows = [r for r in rows if r["timestamp"] < cutoff_ts]
            logger.info(
                "Correction train: %d rows after cutoff filter (cutoff=%s)",
                len(rows), cutoff_ts[:16],
            )

        if len(rows) < config.CORRECTION_MIN_VERIFIED:
            return CorrectionResult(False, len(rows), None, None,
                                    f"Мало данных: {len(rows)} / {config.CORRECTION_MIN_VERIFIED}")

        # ── Build feature matrix X and target matrix y ───────────────────────
        X_list, y_list = [], []
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["timestamp"])
            except Exception:
                ts = datetime.utcnow()
            sin_h   = math.sin(2 * math.pi * ts.hour / 24)
            cos_h   = math.cos(2 * math.pi * ts.hour / 24)
            sin_dow = math.sin(2 * math.pi * ts.weekday() / 7)
            cos_dow = math.cos(2 * math.pi * ts.weekday() / 7)

            x = [
                r["temp_1h"]        or 0.0,
                r["temp_2h"]        or 0.0,
                r["temp_3h"]        or 0.0,
                r["pressure_1h"]    or 0.0,
                r["pressure_2h"]    or 0.0,
                r["pressure_3h"]    or 0.0,
                r["pressure_trend"] or 0.0,
                sin_h, cos_h, sin_dow, cos_dow,
            ]
            y = [
                r["signed_error_temp_1h"] or 0.0,
                r["signed_error_temp_2h"] or 0.0,
                r["signed_error_temp_3h"] or 0.0,
                r["signed_error_pres_1h"] or 0.0,
                r["signed_error_pres_2h"] or 0.0,
                r["signed_error_pres_3h"] or 0.0,
            ]
            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # ── Normalise inputs ─────────────────────────────────────────────────
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        std[std == 0] = 1.0
        X_norm = (X - mean) / std

        # ── Train / val split ────────────────────────────────────────────────
        split = int(len(X_norm) * 0.8)
        X_tr, X_v = X_norm[:split], X_norm[split:]
        y_tr, y_v = y[:split], y[split:]

        # MAE before correction (baseline)
        mae_before = float(np.mean(np.abs(y_v[:, 0]))) if len(y_v) > 0 else None

        # ── Build Keras model ────────────────────────────────────────────────
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(11,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(6),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.CORRECTION_LR),
            loss='mse',
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=config.CORRECTION_PATIENCE,
                restore_best_weights=True,
            ),
        ]
        model.fit(
            X_tr, y_tr,
            validation_data=(X_v, y_v),
            epochs=config.CORRECTION_EPOCHS,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        # MAE after correction
        if len(X_v) > 0:
            preds = model.predict(X_v, verbose=0)
            residuals = y_v[:, 0] - preds[:, 0]
            mae_after = float(np.mean(np.abs(residuals)))
        else:
            mae_after = None

        # ── Save weights ─────────────────────────────────────────────────────
        os.makedirs(config.CORRECTION_DIR, exist_ok=True)
        weights_path = config.CORRECTION_WEIGHTS_PATH
        if not weights_path.endswith(".npz"):
            weights_path = weights_path + ".npz"
        np.savez(weights_path, *model.get_weights())

        # ── Save scaler ───────────────────────────────────────────────────────
        with open(config.CORRECTION_SCALER_PATH, "w") as fh:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, fh)

        # ── Save meta ─────────────────────────────────────────────────────────
        meta = {
            "trained_at":  datetime.utcnow().isoformat(),
            "n_samples":   len(rows),
            "mae_before":  mae_before,
            "mae_after":   mae_after,
        }
        with open(config.CORRECTION_META_PATH, "w") as fh:
            json.dump(meta, fh, indent=2)

        # ── Reload into self ──────────────────────────────────────────────────
        self._try_load()

        logger.info(
            "Correction model trained: n=%d  MAE before=%.4f  after=%.4f",
            len(rows), mae_before or 0.0, mae_after or 0.0,
        )
        return CorrectionResult(
            success=True,
            n_samples=len(rows),
            mae_before=mae_before,
            mae_after=mae_after,
            message="Обучение завершено успешно",
        )

    def predict_correction_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch-predict corrections for N samples.

        Args:
            X: (N, 11) float32 feature matrix — same column order as the
               single-sample path: temp_1h, temp_2h, temp_3h,
               pres_1h, pres_2h, pres_3h, pressure_trend,
               sin_h, cos_h, sin_dow, cos_dow.

        Returns:
            (N, 6) float32 delta array, or np.zeros((N, 6)) on any error.
        """
        if not self._loaded:
            return np.zeros((len(X), 6), dtype=np.float32)
        try:
            x_norm = (X - self._mean) / self._std
            h1  = np.maximum(0, x_norm @ self._w1 + self._b1)
            h2  = np.maximum(0, h1     @ self._w2 + self._b2)
            return (h2 @ self._w3 + self._b3).astype(np.float32)
        except Exception as exc:
            logger.warning("predict_correction_batch error: %s", exc)
            return np.zeros((len(X), 6), dtype=np.float32)

    def predict_correction(self, base_forecast) -> np.ndarray:
        """Predict residual corrections for a base forecast.

        Args:
            base_forecast: ForecastResult from the base LSTM.

        Returns:
            np.ndarray of shape (6,) — deltas for
            [temp_1h, temp_2h, temp_3h, pres_1h, pres_2h, pres_3h].
            Returns np.zeros(6) on any error.
        """
        try:
            if not self._loaded:
                return np.zeros(6, dtype=np.float32)

            ts = datetime.now()
            sin_h   = math.sin(2 * math.pi * ts.hour / 24)
            cos_h   = math.cos(2 * math.pi * ts.hour / 24)
            sin_dow = math.sin(2 * math.pi * ts.weekday() / 7)
            cos_dow = math.cos(2 * math.pi * ts.weekday() / 7)

            x = np.array([[
                base_forecast.temp_in_1h     or 0.0,
                base_forecast.temp_in_2h     or 0.0,
                base_forecast.temp_in_3h     or 0.0,
                base_forecast.pressure_in_1h or 0.0,
                base_forecast.pressure_in_2h or 0.0,
                base_forecast.pressure_in_3h or 0.0,
                base_forecast.pressure_trend or 0.0,
                sin_h, cos_h, sin_dow, cos_dow,
            ]], dtype=np.float32)

            x_norm = (x - self._mean) / self._std

            # Forward pass (pure numpy)
            h1  = np.maximum(0, x_norm @ self._w1 + self._b1)
            h2  = np.maximum(0, h1     @ self._w2 + self._b2)
            out = h2 @ self._w3 + self._b3

            return out[0].astype(np.float32)
        except Exception as exc:
            logger.warning("predict_correction error (returning zeros): %s", exc)
            return np.zeros(6, dtype=np.float32)

    def apply_correction(self, base, deltas: np.ndarray):
        """Apply delta corrections to a ForecastResult.

        Args:
            base:   Original ForecastResult (method="lstm").
            deltas: np.ndarray shape (6,) from predict_correction().

        Returns:
            New ForecastResult with method="lstm_corrected" and adjusted values.
        """
        from dataclasses import replace

        def _add(val, delta):
            if val is None:
                return None
            return round(val + float(delta), 4)

        return replace(
            base,
            method="lstm_corrected",
            temp_in_1h=_add(base.temp_in_1h, deltas[0]),
            temp_in_2h=_add(base.temp_in_2h, deltas[1]),
            temp_in_3h=_add(base.temp_in_3h, deltas[2]),
            pressure_in_1h=_add(base.pressure_in_1h, deltas[3]),
            pressure_in_2h=_add(base.pressure_in_2h, deltas[4]),
            pressure_in_3h=_add(base.pressure_in_3h, deltas[5]),
            correction_applied=True,
            correction_delta_temp_1h=float(deltas[0]),
            correction_delta_pres_1h=float(deltas[3]),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_training_cutoff_ts(self) -> Optional[str]:
        """Return the ISO timestamp at the training/held-out split boundary.

        Reads ``config.DB_PATH`` (weather_history.db), takes the row at
        index ``int(N * (1 - VALIDATION_SPLIT))``, and returns its timestamp.
        Returns None on any error.
        """
        try:
            with sqlite3.connect(config.DB_PATH) as conn:
                ts_rows = conn.execute(
                    "SELECT timestamp FROM readings ORDER BY id"
                ).fetchall()
            if not ts_rows:
                return None
            cutoff_idx = int(len(ts_rows) * (1.0 - config.VALIDATION_SPLIT))
            cutoff_idx = min(cutoff_idx, len(ts_rows) - 1)
            return ts_rows[cutoff_idx][0]
        except Exception as exc:
            logger.warning("_get_training_cutoff_ts error: %s", exc)
            return None

    def _try_load(self) -> None:
        """Attempt to load weights and scaler from disk."""
        try:
            weights_path = config.CORRECTION_WEIGHTS_PATH
            if not weights_path.endswith(".npz"):
                weights_path = weights_path + ".npz"

            if not os.path.exists(weights_path):
                return
            if not os.path.exists(config.CORRECTION_SCALER_PATH):
                return

            data = np.load(weights_path, allow_pickle=False)
            self._w1 = data["arr_0"]
            self._b1 = data["arr_1"]
            self._w2 = data["arr_2"]
            self._b2 = data["arr_3"]
            self._w3 = data["arr_4"]
            self._b3 = data["arr_5"]

            with open(config.CORRECTION_SCALER_PATH, "r") as fh:
                sc = json.load(fh)
            self._mean = np.array(sc["mean"], dtype=np.float32)
            self._std  = np.array(sc["std"],  dtype=np.float32)
            self._std[self._std == 0] = 1.0

            self._loaded = True
            logger.info("CorrectionModel loaded from %s", weights_path)
        except Exception as exc:
            self._loaded = False
            logger.warning("CorrectionModel load failed: %s", exc)
