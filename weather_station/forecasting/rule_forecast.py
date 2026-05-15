"""
Rule-based weather forecaster.

Used as a fallback while the LSTM model is not yet ready (insufficient data).
Analyses recent pressure, humidity and temperature trends using simple linear
regression and a fixed decision table.

All three forecast horizons (+1 h, +2 h, +3 h) are always populated:
  - Temperature: extrapolated via linear regression on recent readings.
  - Pressure: extrapolated from the computed hPa/hour slope.
  - Precipitation probability: derived from the per-step pressure drop.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from forecasting.forecast_result import ForecastResult
from sensors.bme280_sensor import WeatherData
from utils.logger import get_logger

logger = get_logger("forecasting.rule")


def _pressure_drop_to_precip_prob(drop: float) -> float:
    """Convert a pressure drop (hPa, negative = falling) to a precip probability 0–1."""
    if drop < -3:  return 0.85
    if drop < -2:  return 0.70
    if drop < -1:  return 0.50
    if drop <  0:  return 0.30
    if drop <  1:  return 0.10
    return 0.05


class RuleForecaster:
    """Heuristic forecaster based on pressure trend and sensor thresholds.

    Uses up to the last 60 readings to compute linear-regression pressure and
    temperature slopes (per hour) and applies a priority-ordered rule table.
    """

    WINDOW: int = 60  # readings to include in trend calculation

    def predict(
        self,
        recent: List[WeatherData],
        current_weather: Optional[WeatherData] = None,
    ) -> ForecastResult:
        """Produce a ForecastResult from recent sensor readings.

        Args:
            recent:          Chronologically ordered list of WeatherData readings.
            current_weather: Most recent reading used as the extrapolation baseline.
                             Falls back to ``recent[-1]`` when not supplied.

        Returns:
            ForecastResult with method="rule-based" and all +1/+2/+3 h fields
            populated (never None for an active result).
        """
        if not recent:
            return self._no_data_result()

        data    = recent[-self.WINDOW:]
        baseline = current_weather if current_weather is not None else data[-1]

        pressure_trend = self._pressure_trend_hpa_per_hour(data)
        temp_trend     = self._temp_trend_per_hour(data)
        last_humidity  = data[-1].humidity
        temp_drop      = self._temp_drop_30min(data)

        # Priority-ordered rule table
        if last_humidity > 85 and pressure_trend < 0:
            text, conf = "Высокая вероятность осадков", 0.90
        elif temp_drop > 2.0:
            text, conf = "Возможен холодный фронт", 0.70
        elif pressure_trend < -3.0:
            text, conf = "Быстрое ухудшение, возможна гроза", 0.85
        elif pressure_trend < -1.0:
            text, conf = "Ухудшение погоды, возможен дождь", 0.70
        elif pressure_trend > 1.0:
            text, conf = "Улучшение, прояснение", 0.80
        else:
            text, conf = "Погода без изменений", 0.75

        # ── Temperature extrapolation (linear, clamped at ±5 °C/h noise guard) ──
        t_slope = temp_trend if abs(temp_trend) <= 5.0 else 0.0
        cur_t   = baseline.temperature
        temp_1h = round(cur_t + t_slope * 1, 2)
        temp_2h = round(cur_t + t_slope * 2, 2)
        temp_3h = round(cur_t + t_slope * 3, 2)

        # ── Pressure extrapolation ────────────────────────────────────────────
        cur_p   = baseline.pressure
        pres_1h = round(cur_p + pressure_trend * 1, 2)
        pres_2h = round(cur_p + pressure_trend * 2, 2)
        pres_3h = round(cur_p + pressure_trend * 3, 2)

        # ── Precipitation probability (humidity bonus if > 85%) ───────────────
        hum_factor = 1.2 if last_humidity > 85 else 1.0

        def _precip(drop: float) -> float:
            return min(1.0, _pressure_drop_to_precip_prob(drop) * hum_factor)

        precip_1h = _precip(pres_1h - cur_p)
        precip_2h = _precip(pres_2h - cur_p)
        precip_3h = _precip(pres_3h - cur_p)

        return ForecastResult(
            method         = "rule-based",
            forecast_text  = text,
            confidence     = conf,
            pressure_trend = pressure_trend,
            temp_in_1h     = temp_1h,
            temp_in_2h     = temp_2h,
            temp_in_3h     = temp_3h,
            precip_prob_1h = precip_1h,
            precip_prob_2h = precip_2h,
            precip_prob_3h = precip_3h,
            pressure_in_1h = pres_1h,
            pressure_in_2h = pres_2h,
            pressure_in_3h = pres_3h,
            valid_until    = datetime.utcnow() + timedelta(hours=1),
            model_version  = "rule_v1",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _linreg_slope(data: List[WeatherData], attr: str) -> float:
        """Linear-regression slope of *attr* vs time (units/hour).

        Args:
            data: Chronological readings.
            attr: Attribute name on WeatherData (e.g. ``"pressure"``).

        Returns:
            Slope in attribute-units per hour; 0.0 if fewer than 2 points.
        """
        if len(data) < 2:
            return 0.0
        t0 = data[0].timestamp.timestamp()
        x  = np.array([(r.timestamp.timestamp() - t0) / 3600.0 for r in data],
                      dtype=np.float64)
        y  = np.array([getattr(r, attr) for r in data], dtype=np.float64)
        x_range = x[-1] - x[0]
        if x_range < 1e-6:
            return 0.0
        xm, ym = x.mean(), y.mean()
        cov = ((x - xm) * (y - ym)).sum()
        var = ((x - xm) ** 2).sum()
        return float(cov / var) if var > 1e-10 else 0.0

    def _pressure_trend_hpa_per_hour(self, data: List[WeatherData]) -> float:
        return self._linreg_slope(data, "pressure")

    def _temp_trend_per_hour(self, data: List[WeatherData]) -> float:
        return self._linreg_slope(data, "temperature")

    @staticmethod
    def _temp_drop_30min(data: List[WeatherData]) -> float:
        """Temperature drop over the last 30 minutes (positive = dropped)."""
        if len(data) < 2:
            return 0.0
        cutoff   = data[-1].timestamp.timestamp() - 1800.0
        baseline = next((r for r in data if r.timestamp.timestamp() >= cutoff), data[0])
        return float(max(0.0, baseline.temperature - data[-1].temperature))

    @staticmethod
    def _no_data_result() -> ForecastResult:
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
            valid_until    = datetime.utcnow(),
            model_version  = "none",
        )
