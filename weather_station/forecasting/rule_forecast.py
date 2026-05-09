"""
Rule-based weather forecaster.

Used as a fallback while the LSTM model is not yet ready (insufficient data).
Analyses recent pressure, humidity and temperature trends using simple linear
regression and a fixed decision table.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np

from forecasting.forecast_result import ForecastResult
from sensors.bme280_sensor import WeatherData
from utils.logger import get_logger

logger = get_logger("forecasting.rule")


class RuleForecaster:
    """Heuristic forecaster based on pressure trend and sensor thresholds.

    Uses up to the last 60 readings to compute a linear-regression pressure
    slope (hPa/hour) and applies a priority-ordered rule table.
    """

    WINDOW: int = 60  # readings to include in trend calculation

    def predict(self, recent: List[WeatherData]) -> ForecastResult:
        """Produce a ForecastResult from recent sensor readings.

        Args:
            recent: Chronologically ordered list of WeatherData readings.

        Returns:
            ForecastResult with method="rule-based".
        """
        if not recent:
            return self._no_data_result()

        data = recent[-self.WINDOW:]
        pressure_trend = self._pressure_trend_hpa_per_hour(data)
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

        return ForecastResult(
            method="rule-based",
            forecast_text=text,
            confidence=conf,
            pressure_trend=pressure_trend,
            temp_in_1h=None,
            temp_in_2h=None,
            temp_in_3h=None,
            pressure_in_1h=None,
            pressure_in_2h=None,
            pressure_in_3h=None,
            valid_until=datetime.now() + timedelta(hours=1),
            model_version="rule_v1",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pressure_trend_hpa_per_hour(data: List[WeatherData]) -> float:
        """Linear-regression slope of pressure vs time (hPa/hour).

        Args:
            data: Chronological readings.

        Returns:
            Slope in hPa per hour; 0.0 if fewer than 2 points or flat time axis.
        """
        if len(data) < 2:
            return 0.0

        t0 = data[0].timestamp.timestamp()
        x  = np.array([(r.timestamp.timestamp() - t0) / 3600.0 for r in data],
                      dtype=np.float64)
        y  = np.array([r.pressure for r in data], dtype=np.float64)

        x_range = x[-1] - x[0]
        if x_range < 1e-6:
            return 0.0

        x_mean = x.mean()
        y_mean = y.mean()
        cov    = ((x - x_mean) * (y - y_mean)).sum()
        var    = ((x - x_mean) ** 2).sum()

        return float(cov / var) if var > 1e-10 else 0.0

    @staticmethod
    def _temp_drop_30min(data: List[WeatherData]) -> float:
        """Temperature drop over the last 30 minutes (positive = dropped).

        Args:
            data: Chronological readings.

        Returns:
            °C drop; 0.0 if window is too short.
        """
        if len(data) < 2:
            return 0.0

        cutoff = data[-1].timestamp.timestamp() - 1800.0  # 30 min
        baseline = next(
            (r for r in data if r.timestamp.timestamp() >= cutoff),
            data[0],
        )
        drop = baseline.temperature - data[-1].temperature
        return float(max(0.0, drop))

    @staticmethod
    def _no_data_result() -> ForecastResult:
        return ForecastResult(
            method="insufficient_data",
            forecast_text="Нет данных",
            confidence=0.0,
            pressure_trend=0.0,
            temp_in_1h=None,
            temp_in_2h=None,
            temp_in_3h=None,
            pressure_in_1h=None,
            pressure_in_2h=None,
            pressure_in_3h=None,
            valid_until=datetime.now(),
            model_version="none",
        )
