"""
Hybrid forecaster — combines Online API, LSTM, and rule-based engines.

Priority order:
  1. OnlineForecaster   — internet available AND GPS has a valid fix
  2. LSTMForecaster     — offline (or no GPS) AND enough data collected
  3. RuleForecaster     — always available as last resort

All fallbacks are transparent to the caller: every path returns a
:class:`HybridForecastResult` (a ForecastResult subclass) that also
carries metadata about which mode was used and why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Literal, Optional

import config
from forecasting.forecast_result import ForecastResult
from forecasting.gps_pressure_correction import validate_gps_for_forecast
from forecasting.online_forecast import OnlineForecaster
from utils.logger import get_logger

if TYPE_CHECKING:
    from forecasting.data_store import DataStore
    from forecasting.lstm_forecast import LSTMForecaster
    from forecasting.rule_forecast import RuleForecaster
    from sensors.bme280_sensor import WeatherData
    from sensors.gps_sensor import GpsData

logger = get_logger("forecasting.hybrid")


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class HybridForecastResult(ForecastResult):
    """ForecastResult extended with hybrid-mode metadata.

    All parent fields are inherited unchanged.  The extra fields below
    describe which engine produced the result and why.

    Attributes:
        mode:               Active forecasting engine.
        internet_available: True when the API was reachable this cycle.
        gps_used:           True when GPS coordinates drove the API request.
        lstm_ready:         True when the LSTM model is trained and has enough data.
        data_collected:     Current number of rows in the history database.
        data_required:      Minimum rows needed before LSTM becomes active.
        api_last_success:   Datetime of the last successful API fetch, or None.
        precip_probability: Precipitation probability % from the API, or None.
    """

    # Required child fields (no defaults) — must come before optional fields
    mode:               Literal["online", "lstm", "rule-based"]
    internet_available: bool
    gps_used:           bool
    lstm_ready:         bool
    data_collected:     int
    data_required:      int

    # Optional child fields (with defaults)
    api_last_success:   Optional[datetime] = None
    precip_probability: Optional[float]    = None


# ── Forecaster ─────────────────────────────────────────────────────────────────

class HybridForecaster:
    """Orchestrates Online / LSTM / Rule-based forecasters by priority.

    Args:
        data_store:      DataStore used to count collected readings.
        lstm_forecaster: Pre-constructed LSTMForecaster.
        rule_forecaster: Pre-constructed RuleForecaster.
    """

    def __init__(
        self,
        data_store:      "DataStore",
        lstm_forecaster: "LSTMForecaster",
        rule_forecaster: "RuleForecaster",
    ) -> None:
        self._data_store  = data_store
        self._lstm        = lstm_forecaster
        self._rules       = rule_forecaster
        self._online      = OnlineForecaster()
        self._api_last_ok: Optional[datetime] = None
        self._mode: Literal["online", "lstm", "rule-based"] = "rule-based"

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        gps:             Optional["GpsData"],
        recent:          List["WeatherData"],
        current_weather: Optional["WeatherData"] = None,
    ) -> HybridForecastResult:
        """Produce the best available forecast for the current conditions.

        Tries engines in priority order and falls through on any failure.

        Args:
            gps:             Latest GpsData snapshot, or None if GPS failed.
            recent:          Recent WeatherData readings (chronological).
            current_weather: Most recent reading — passed to RuleForecaster
                             as the extrapolation baseline.

        Returns:
            :class:`HybridForecastResult` from whichever engine succeeded.
        """
        gps_valid = validate_gps_for_forecast(gps)
        internet  = config.ONLINE_FORECAST_ENABLED and self._online.is_available()

        # ── 1. Online API ─────────────────────────────────────────────────────
        if internet and gps_valid:
            try:
                base = self._online.fetch(
                    lat=gps.latitude,
                    lon=gps.longitude,
                    altitude=gps.altitude,
                )
            except Exception as exc:
                logger.warning("OnlineForecaster.fetch raised: %s", exc)
                base = None

            if base is not None:
                self._api_last_ok = datetime.now()
                self._mode        = "online"
                logger.debug("Forecast mode: online API.")
                return self._wrap(base, mode="online",
                                  internet_available=True, gps_used=True)

        # ── 2. LSTM ───────────────────────────────────────────────────────────
        if self._lstm.is_ready():
            try:
                base = self._lstm.predict(recent)
                if base.method != "insufficient_data":
                    self._mode = "lstm"
                    logger.debug("Forecast mode: LSTM.")
                    return self._wrap(base, mode="lstm",
                                      internet_available=internet, gps_used=False)
            except Exception as exc:
                logger.warning("LSTMForecaster.predict raised: %s", exc)

        # ── 3. Rule-based fallback ────────────────────────────────────────────
        try:
            base = self._rules.predict(recent, current_weather)
        except Exception as exc:
            logger.error("RuleForecaster.predict raised: %s — using stub.", exc)
            base = self._stub_result()

        self._mode = "rule-based"
        logger.debug("Forecast mode: rule-based.")
        return self._wrap(base, mode="rule-based",
                          internet_available=internet, gps_used=False)

    def get_mode(self) -> Literal["online", "lstm", "rule-based"]:
        """Return the mode used in the most recent :meth:`predict` call."""
        return self._mode

    def get_status(self) -> dict:
        """Return a status snapshot for diagnostics / display.

        Returns:
            Dict with keys: ``mode``, ``data_count``, ``lstm_ready``,
            ``internet_available``, ``api_last_success``.
        """
        return {
            "mode":               self._mode,
            "data_count":         self._data_store.count(),
            "lstm_ready":         self._lstm.is_ready(),
            "internet_available": self._online.is_available(),
            "api_last_success":   self._api_last_ok,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _wrap(
        self,
        base:               ForecastResult,
        mode:               Literal["online", "lstm", "rule-based"],
        internet_available: bool,
        gps_used:           bool,
    ) -> HybridForecastResult:
        """Promote a plain ForecastResult to HybridForecastResult."""
        # Convenience max-precip for TFT display
        probs = [p for p in (base.precip_prob_1h, base.precip_prob_2h,
                              base.precip_prob_3h) if p is not None]
        max_precip = max(probs) if probs else None

        return HybridForecastResult(
            # ── inherited ForecastResult fields ──
            method         = base.method,
            forecast_text  = base.forecast_text,
            confidence     = base.confidence,
            pressure_trend = base.pressure_trend,
            temp_in_1h     = base.temp_in_1h,
            temp_in_2h     = base.temp_in_2h,
            temp_in_3h     = base.temp_in_3h,
            precip_prob_1h = base.precip_prob_1h,
            precip_prob_2h = base.precip_prob_2h,
            precip_prob_3h = base.precip_prob_3h,
            pressure_in_1h = base.pressure_in_1h,
            pressure_in_2h = base.pressure_in_2h,
            pressure_in_3h = base.pressure_in_3h,
            valid_until    = base.valid_until,
            model_version  = base.model_version,
            # ── hybrid metadata ──
            mode               = mode,
            internet_available = internet_available,
            gps_used           = gps_used,
            lstm_ready         = self._lstm.is_ready(),
            data_collected     = self._data_store.count(),
            data_required      = config.FORECAST_MIN_READINGS,
            api_last_success   = self._api_last_ok,
            precip_probability = max_precip,
        )

    @staticmethod
    def _stub_result() -> ForecastResult:
        from datetime import datetime
        return ForecastResult(
            method         = "insufficient_data",
            forecast_text  = "Нет данных",
            confidence     = 0.0,
            pressure_trend = 0.0,
            temp_in_1h     = None,
            temp_in_2h     = None,
            temp_in_3h     = None,
            precip_prob_1h = None,
            precip_prob_2h = None,
            precip_prob_3h = None,
            pressure_in_1h = None,
            pressure_in_2h = None,
            pressure_in_3h = None,
            valid_until    = datetime.now(),
            model_version  = "none",
        )
