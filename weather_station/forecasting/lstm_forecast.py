"""
LSTM-based weather nowcaster.

Training and inference both use full TensorFlow/Keras.
The trained model is saved as a Keras SavedModel directory and reloaded
on startup.  Training always runs in a background thread — it never
blocks the sensor loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

import config
from forecasting.correction_model import CorrectionModel
from forecasting.forecast_result import ForecastResult
from sensors.bme280_sensor import WeatherData
from utils.logger import get_logger

if TYPE_CHECKING:
    from forecasting.data_store import DataStore

logger = get_logger("forecasting.lstm")


class LSTMForecaster:
    """LSTM weather nowcaster with background retraining.

    Args:
        data_store: DataStore instance used by _retrain_if_needed().
    """

    def __init__(self, data_store: "DataStore") -> None:
        self._data_store   = data_store
        self._model        = None   # tf.keras.Model, loaded after training
        self._scaler_min: Optional[np.ndarray] = None
        self._scaler_max: Optional[np.ndarray] = None
        self._last_val_loss: float = 1.0
        self._last_train_count: int = 0
        self._last_train_time: float = 0.0
        self._is_training: bool = False
        self._training_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Pre-import TensorFlow in the main thread so the background training
        # thread can use the cached sys.modules entry (TF init is not thread-safe).
        self._tf_available = False
        try:
            import tensorflow as _tf  # noqa: F401
            self._tf_available = True
            logger.info("TensorFlow %s available for training.", _tf.__version__)
        except Exception as exc:
            logger.warning("TensorFlow not available — LSTM training disabled: %s", exc)

        self._correction = CorrectionModel()
        self._try_load_from_disk()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True when the model is loaded and the DB has enough rows."""
        with self._lock:
            return (
                self._model is not None
                and self._scaler_min is not None
                and self._data_store.count() >= config.FORECAST_MIN_READINGS
                and os.path.exists(
                    config.WEIGHTS_PATH if config.WEIGHTS_PATH.endswith(".npz")
                    else config.WEIGHTS_PATH + ".npz"
                )
            )

    def predict(self, recent: List[WeatherData]) -> ForecastResult:
        """Run Keras inference on the most recent readings.

        Falls back to RuleForecaster on any error.

        Args:
            recent: Chronological list of WeatherData (at least SEQUENCE_LENGTH items).

        Returns:
            ForecastResult with method="lstm", or a rule-based/insufficient result.
        """
        if len(recent) < config.SEQUENCE_LENGTH:
            return self._insufficient_data_result()

        try:
            with self._lock:
                model      = self._model
                scaler_min = self._scaler_min.copy()
                scaler_max = self._scaler_max.copy()

            if model is None:
                raise RuntimeError("Model not loaded")

            # Build (1, seq_len, 3) input tensor
            raw = np.array(
                [[r.temperature, r.humidity, r.pressure]
                 for r in recent[-config.SEQUENCE_LENGTH:]],
                dtype=np.float32,
            )
            rng           = scaler_max - scaler_min
            rng[rng == 0] = 1.0
            norm          = (raw - scaler_min) / rng
            inp           = norm[np.newaxis]  # (1, seq_len, 3)

            output = model(inp, training=False).numpy()[0]  # (9,)

            # Denormalise: output is [t,h,p] × 3 forecast steps
            preds = []
            for i in range(len(config.FORECAST_STEPS)):
                step_norm = output[i * 3: i * 3 + 3]
                preds.append(step_norm * rng + scaler_min)

            temp_1h, temp_2h, temp_3h = (float(p[0]) for p in preds)
            pres_1h, pres_2h, pres_3h = (float(p[2]) for p in preds)

            current_pressure = recent[-1].pressure
            pressure_trend   = pres_3h - current_pressure
            forecast_text    = self._trend_to_text(pressure_trend)
            confidence       = max(0.0, min(1.0, 1.0 - self._last_val_loss))

            pp = self._pressure_drop_to_precip_prob
            result = ForecastResult(
                method="lstm",
                forecast_text=forecast_text,
                confidence=confidence,
                pressure_trend=pressure_trend,
                temp_in_1h=temp_1h,
                temp_in_2h=temp_2h,
                temp_in_3h=temp_3h,
                precip_prob_1h=pp(pres_1h - current_pressure),
                precip_prob_2h=pp(pres_2h - current_pressure),
                precip_prob_3h=pp(pres_3h - current_pressure),
                pressure_in_1h=pres_1h,
                pressure_in_2h=pres_2h,
                pressure_in_3h=pres_3h,
                valid_until=datetime.now() + timedelta(hours=3),
                model_version="lstm_v1",
            )

            # Apply residual correction if model is ready
            if self._correction.is_ready():
                deltas = self._correction.predict_correction(result)
                result = self._correction.apply_correction(result, deltas)
            return result

        except Exception as exc:
            logger.error("LSTM predict error — falling back to rules: %s", exc)
            from forecasting.rule_forecast import RuleForecaster
            return RuleForecaster().predict(recent)

    def train(self, readings: List[WeatherData]) -> Dict:
        """Train LSTM on *readings*, save as Keras SavedModel, reload.

        Args:
            readings: Full chronological history from DataStore.

        Returns:
            Dict with metric keys, or empty dict on failure.
        """
        if not self._tf_available:
            logger.error("TensorFlow not available — cannot train.")
            return {}
        try:
            import tensorflow as tf
        except Exception as exc:
            logger.error("TensorFlow import failed in training thread: %s", exc)
            return {}

        if len(readings) < config.SEQUENCE_LENGTH + max(config.FORECAST_STEPS) + 10:
            logger.warning("train(): not enough readings (%d).", len(readings))
            return {}

        readings = readings[-config.LSTM_MAX_TRAIN_READINGS:]
        logger.info("LSTM training started on %d readings.", len(readings))

        # Feature matrix
        data = np.array(
            [[r.temperature, r.humidity, r.pressure] for r in readings],
            dtype=np.float32,
        )
        data = self._filter_data(data)
        if len(data) < config.SEQUENCE_LENGTH + max(config.FORECAST_STEPS) + 10:
            logger.warning("train(): too few readings after filtering (%d).", len(data))
            return {}
        s_min = data.min(axis=0)
        s_max = data.max(axis=0)
        rng   = s_max - s_min
        rng[rng == 0] = 1.0
        norm  = (data - s_min) / rng

        # Persist scaler
        os.makedirs(os.path.dirname(config.SCALER_PATH) or '.', exist_ok=True)
        with open(config.SCALER_PATH, 'w') as fh:
            json.dump({'min': s_min.tolist(), 'max': s_max.tolist()}, fh)
        with self._lock:
            self._scaler_min = s_min
            self._scaler_max = s_max

        # Build sequences
        seq   = config.SEQUENCE_LENGTH
        steps = config.FORECAST_STEPS
        X, y  = [], []
        for i in range(len(norm) - seq - max(steps)):
            X.append(norm[i: i + seq])
            targets = []
            for st in steps:
                targets.extend(norm[i + seq + st - 1])
            y.append(targets)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if len(X) < 10:
            logger.warning("train(): too few sequences (%d).", len(X))
            return {}

        split      = int(len(X) * 0.8)
        X_tr, X_v = X[:split], X[split:]
        y_tr, y_v = y[:split], y[split:]

        with self._lock:
            warm_weights = (self._model.get_weights()
                            if self._model is not None else None)

        model    = self._build_and_train_keras(X_tr, y_tr, X_v, y_v, warm_weights)
        val_loss = float(model.evaluate(X_v, y_v, verbose=0))
        self._last_val_loss = val_loss

        # ── Metrics on validation set ─────────────────────────────────────────
        y_pred = model.predict(X_v, verbose=0)

        n_steps     = len(config.FORECAST_STEPS)
        s_min_tiled = np.tile(s_min, n_steps)
        rng_tiled   = np.tile(rng,   n_steps)

        y_pred_d = y_pred * rng_tiled + s_min_tiled
        y_v_d    = y_v    * rng_tiled + s_min_tiled

        step_labels = ['1h', '2h', '3h']
        metrics: Dict[str, float] = {'val_loss': val_loss}
        for i, label in enumerate(step_labels[:n_steps]):
            pred_t = y_pred_d[:, i * 3 + 0];  true_t = y_v_d[:, i * 3 + 0]
            pred_p = y_pred_d[:, i * 3 + 2];  true_p = y_v_d[:, i * 3 + 2]
            metrics[f'rmse_temp_{label}'] = float(np.sqrt(np.mean((pred_t - true_t) ** 2)))
            metrics[f'mae_temp_{label}']  = float(np.mean(np.abs(pred_t - true_t)))
            metrics[f'rmse_pres_{label}'] = float(np.sqrt(np.mean((pred_p - true_p) ** 2)))
            metrics[f'mae_pres_{label}']  = float(np.mean(np.abs(pred_p - true_p)))

        pres_current  = X_v[:, -1, 2] * rng[2] + s_min[2]
        pres_pred_end = y_pred_d[:, (n_steps - 1) * 3 + 2]
        pres_true_end = y_v_d[:,   (n_steps - 1) * 3 + 2]

        pred_rising = (pres_pred_end > pres_current).astype(np.int8)
        true_rising = (pres_true_end > pres_current).astype(np.int8)

        tp = int(np.sum((pred_rising == 1) & (true_rising == 1)))
        fp = int(np.sum((pred_rising == 1) & (true_rising == 0)))
        fn = int(np.sum((pred_rising == 0) & (true_rising == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics.update({'precision': precision, 'recall': recall, 'f1': f1,
                        'tp': tp, 'fp': fp, 'fn': fn})

        try:
            os.makedirs(os.path.dirname(config.METRICS_PATH) or '.', exist_ok=True)
            with open(config.METRICS_PATH, 'w') as fh:
                json.dump(metrics, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not save metrics file: %s", exc)

        logger.info(
            "LSTM training done — val_loss=%.4f | "
            "RMSE T/P (1h): %.2f°C / %.2f hPa | "
            "MAE T/P (1h): %.2f°C / %.2f hPa | "
            "precision=%.2f  recall=%.2f  F1=%.2f",
            val_loss,
            metrics.get('rmse_temp_1h', 0), metrics.get('rmse_pres_1h', 0),
            metrics.get('mae_temp_1h',  0), metrics.get('mae_pres_1h',  0),
            precision, recall, f1,
        )

        # Save weights as numpy arrays — bypasses all Keras/TFLite serialization
        os.makedirs(os.path.dirname(config.WEIGHTS_PATH) or '.', exist_ok=True)
        np.savez(config.WEIGHTS_PATH, *model.get_weights())
        logger.info("Model weights saved → %s", config.WEIGHTS_PATH)

        with self._lock:
            self._model = model

        return metrics

    def _retrain_if_needed(self, research=None) -> None:
        """Trigger background retraining when thresholds are met."""
        if self._is_training:
            return

        current_count    = self._data_store.count()
        time_since_train = time.monotonic() - self._last_train_time

        first_run       = self._last_train_time == 0.0
        enough_new_data = current_count >= self._last_train_count + config.RETRAIN_THRESHOLD
        enough_time     = first_run or time_since_train >= config.LSTM_RETRAIN_INTERVAL

        if not (enough_new_data and enough_time):
            return

        readings       = self._data_store.get_all()
        self._is_training = True
        snapshot_count = current_count

        def _run():
            t0 = time.monotonic()
            try:
                metrics  = self.train(readings)
                duration = time.monotonic() - t0
                self._last_train_count = snapshot_count
                self._last_train_time  = time.monotonic()
                logger.info("Background retrain complete (count=%d).", snapshot_count)

                if research is not None and metrics:
                    try:
                        research.log_lstm_training({
                            "readings_count": snapshot_count,
                            "mae_temp":       metrics.get("mae_temp_1h", 0.0),
                            "mae_pressure":   metrics.get("mae_pres_1h", 0.0),
                            "rmse_temp":      metrics.get("rmse_temp_1h", 0.0),
                            "rmse_pressure":  metrics.get("rmse_pres_1h", 0.0),
                            "duration_sec":   round(duration, 2),
                        })
                    except Exception as exc:
                        logger.warning("research.log_lstm_training error: %s", exc)
            except Exception as exc:
                logger.error("Background retrain failed: %s", exc)
            finally:
                self._is_training = False

        self._training_thread = threading.Thread(target=_run, daemon=True, name="lstm-train")
        self._training_thread.start()
        logger.info("Background retraining started (count=%d).", current_count)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _try_load_from_disk(self) -> None:
        """Load scaler and numpy weights from disk if they exist."""
        try:
            if os.path.exists(config.SCALER_PATH):
                with open(config.SCALER_PATH, 'r') as fh:
                    params = json.load(fh)
                self._scaler_min = np.array(params['min'], dtype=np.float32)
                self._scaler_max = np.array(params['max'], dtype=np.float32)
                logger.info("Scaler loaded from %s.", config.SCALER_PATH)

            weights_file = config.WEIGHTS_PATH + ".npz" if not config.WEIGHTS_PATH.endswith(".npz") else config.WEIGHTS_PATH
            if os.path.exists(weights_file) and self._tf_available:
                import tensorflow as tf
                n_out = 3 * len(config.FORECAST_STEPS)
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(config.SEQUENCE_LENGTH, 3)),
                    tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(config.LSTM_UNITS),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(n_out),
                ])
                data = np.load(weights_file, allow_pickle=False)
                model.set_weights([data[f'arr_{i}'] for i in range(len(data.files))])
                self._model = model
                logger.info("Model loaded from weights → %s", weights_file)
        except Exception as exc:
            logger.warning("Could not load model/scaler from disk: %s", exc)

    @staticmethod
    def _filter_data(data: np.ndarray) -> np.ndarray:
        """Remove physically implausible values and IQR spikes, then interpolate.

        Args:
            data: (N, 3) float32 array of [temperature, humidity, pressure].

        Returns:
            Cleaned array with outliers replaced by linear interpolation.
        """
        result = data.astype(np.float64)

        # Hard physical bounds per column: [temp, humidity, pressure]
        bounds = [
            (config.FILTER_TEMP_MIN,     config.FILTER_TEMP_MAX),
            (0.0,                         100.0),
            (config.FILTER_PRESSURE_MIN, config.FILTER_PRESSURE_MAX),
        ]
        for col, (lo, hi) in enumerate(bounds):
            mask = (result[:, col] < lo) | (result[:, col] > hi)
            result[mask, col] = np.nan

        # IQR spike filter per column
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

        # Linear interpolation for NaN values
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

        # Drop any edge rows that are still NaN
        valid_rows = ~np.isnan(result).any(axis=1)
        n_dropped = len(data) - valid_rows.sum()
        if n_dropped:
            logger.info("Filtered %d outlier readings from training data.", n_dropped)
        return result[valid_rows].astype(np.float32)

    @staticmethod
    def _build_and_train_keras(X_train, y_train, X_val, y_val, warm_weights=None):
        import tensorflow as tf

        n_out = 3 * len(config.FORECAST_STEPS)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(config.SEQUENCE_LENGTH, 3)),
            tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(config.LSTM_UNITS),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_out),
        ])
        model.compile(optimizer='adam', loss='mse')

        if warm_weights is not None:
            try:
                model.set_weights(warm_weights)
                logger.info("Warm start: loaded weights from previous model.")
            except Exception as exc:
                logger.warning("Warm start failed, using random init: %s", exc)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
        ]
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )
        return model

    @staticmethod
    def _pressure_drop_to_precip_prob(drop: float) -> float:
        if drop < -3:  return 0.85
        if drop < -2:  return 0.70
        if drop < -1:  return 0.50
        if drop <  0:  return 0.30
        if drop <  1:  return 0.10
        return 0.05

    @staticmethod
    def _trend_to_text(trend_hpa: float) -> str:
        if trend_hpa < -3.0:
            return "Ухудшение погоды, возможны осадки"
        if trend_hpa < -1.0:
            return "Небольшое ухудшение"
        if trend_hpa <= 1.0:
            return "Погода без существенных изменений"
        return "Улучшение погоды"

    @staticmethod
    def _insufficient_data_result() -> ForecastResult:
        return ForecastResult(
            method="insufficient_data",
            forecast_text="Недостаточно данных",
            confidence=0.0,
            pressure_trend=0.0,
            temp_in_1h=None,
            temp_in_2h=None,
            temp_in_3h=None,
            precip_prob_1h=None,
            precip_prob_2h=None,
            precip_prob_3h=None,
            pressure_in_1h=None,
            pressure_in_2h=None,
            pressure_in_3h=None,
            valid_until=datetime.now(),
            model_version="none",
        )
