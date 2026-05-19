"""
Hazard weather detector — detects precursors of dangerous weather
from BME280 trends only (pressure, temperature, humidity).

Never raises — always returns HazardAlert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np

from sensors.bme280_sensor import WeatherData


@dataclass
class HazardAlert:
    level: Literal["NORMAL", "WATCH", "WARNING", "DANGER"]
    phenomenon: str
    description: str
    pressure_trend_1h: float    # hPa/hour, from linear regression
    pressure_trend_10m: float   # hPa/10min
    temp_drop_30m: float        # °C, positive = dropped
    humidity: float
    triggered_rules: list


class HazardDetector:
    """Detects precursors of dangerous weather from BME280 trends only."""

    def detect(self, recent: List[WeatherData]) -> HazardAlert:
        """Analyse recent readings and return a HazardAlert.

        Always returns HazardAlert — never raises.
        Returns NORMAL if fewer than 12 readings available.
        """
        if not recent or len(recent) < 12:
            return self._normal(0.0, 0.0, 0.0, 0.0)

        try:
            p1h   = self._pressure_trend_1h(recent)
            p10m  = self._pressure_trend_10m(recent)
            tdrop = self._temp_drop_30m(recent)
            hum   = recent[-1].humidity
            rules: list = []

            # ── DANGER ────────────────────────────────────────────────────────
            if p1h < -6 and hum > 80:
                rules.append("rapid_pressure_drop_critical")
                return HazardAlert(
                    level="DANGER",
                    phenomenon="⚡ ГРОЗА / ШКВАЛ",
                    description="Критическое падение давления. Возможен шквал.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            if p10m < -1.5 and tdrop > 3.0:
                rules.append("cold_front_signature")
                return HazardAlert(
                    level="DANGER",
                    phenomenon="🌪 ХОЛОДНЫЙ ФРОНТ",
                    description="Резкое падение давления и температуры.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            # ── WARNING ───────────────────────────────────────────────────────
            if p1h < -3 and hum > 70:
                rules.append("storm_likely")
                return HazardAlert(
                    level="WARNING",
                    phenomenon="🌧 ГРОЗА ВЕРОЯТНА",
                    description="Быстрое падение давления при высокой влажности.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            if hum > 92 and p1h < -1:
                rules.append("precipitation_imminent")
                return HazardAlert(
                    level="WARNING",
                    phenomenon="🌧 ОСАДКИ",
                    description="Очень высокая влажность и падающее давление.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            if p1h > 5:
                rules.append("post_frontal_squall")
                return HazardAlert(
                    level="WARNING",
                    phenomenon="💨 ПОСТФРОНТ. ШКВАЛ",
                    description="Резкий рост давления — возможен порывистый ветер.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            # ── WATCH ─────────────────────────────────────────────────────────
            if -3 <= p1h < -1:
                rules.append("pressure_falling")
                return HazardAlert(
                    level="WATCH",
                    phenomenon="🌦 УХУДШЕНИЕ ПОГОДЫ",
                    description=f"Давление падает {p1h:.1f} hPa/ч.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            if tdrop > 2.0 and recent[-1].humidity > recent[0].humidity:
                rules.append("rapid_temp_drop")
                return HazardAlert(
                    level="WATCH",
                    phenomenon="🌬 ПОХОЛОДАНИЕ",
                    description=f"Температура упала на {tdrop:.1f}°C за 30 мин.",
                    pressure_trend_1h=p1h, pressure_trend_10m=p10m,
                    temp_drop_30m=tdrop, humidity=hum,
                    triggered_rules=rules,
                )

            return self._normal(p1h, p10m, tdrop, hum)

        except Exception:
            return self._normal(0.0, 0.0, 0.0, 0.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _linreg_slope(readings: List[WeatherData], attr: str,
                      window_sec: float) -> float:
        """Linear regression slope (units/hour) over the last window_sec seconds."""
        now = readings[-1].timestamp.timestamp()
        cutoff = now - window_sec
        data = [r for r in readings if r.timestamp.timestamp() >= cutoff]
        if len(data) < 2:
            return 0.0
        t0 = data[0].timestamp.timestamp()
        x = np.array([(r.timestamp.timestamp() - t0) / 3600 for r in data],
                     dtype=np.float64)
        y = np.array([getattr(r, attr) for r in data], dtype=np.float64)
        xr = x[-1] - x[0]
        if xr < 1e-6:
            return 0.0
        xm, ym = x.mean(), y.mean()
        cov = ((x - xm) * (y - ym)).sum()
        var = ((x - xm) ** 2).sum()
        return float(cov / var) if var > 1e-10 else 0.0

    def _pressure_trend_1h(self, readings: List[WeatherData]) -> float:
        return self._linreg_slope(readings, "pressure", 3600)

    def _pressure_trend_10m(self, readings: List[WeatherData]) -> float:
        slope_per_hour = self._linreg_slope(readings, "pressure", 600)
        return slope_per_hour / 6

    def _temp_drop_30m(self, readings: List[WeatherData]) -> float:
        cutoff = readings[-1].timestamp.timestamp() - 1800
        baseline = next(
            (r for r in readings if r.timestamp.timestamp() >= cutoff),
            readings[0],
        )
        return float(max(0.0, baseline.temperature - readings[-1].temperature))

    @staticmethod
    def _normal(p1h, p10m, tdrop, hum) -> HazardAlert:
        return HazardAlert(
            level="NORMAL",
            phenomenon="✅ ОПАСНЫХ ЯВЛЕНИЙ НЕ ВЫЯВЛЕНО",
            description="",
            pressure_trend_1h=p1h, pressure_trend_10m=p10m,
            temp_drop_30m=tdrop, humidity=hum,
            triggered_rules=[],
        )
