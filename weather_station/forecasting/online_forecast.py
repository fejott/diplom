"""
Online weather forecaster using the Open-Meteo API.

No API key required.  Requires an active internet connection.
Falls back gracefully when offline — is_available() always returns a bool,
never raises, and fetch() returns None on any error so callers can fall
through to LSTM / rule-based forecasters transparently.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests

import config
from forecasting.forecast_result import ForecastResult
from utils.logger import get_logger

logger = get_logger("forecasting.online")

# WMO Weather Interpretation Code → Russian description
_WMO_TEXTS: Dict[int, str] = {
    0:  "Ясно",
    1:  "Переменная облачность",
    2:  "Переменная облачность",
    3:  "Переменная облачность",
    45: "Туман",
    48: "Туман",
    51: "Морось",
    53: "Морось",
    55: "Морось",
    61: "Дождь",
    63: "Дождь",
    65: "Дождь",
    71: "Снег",
    73: "Снег",
    75: "Снег",
    80: "Ливень",
    81: "Ливень",
    82: "Ливень",
    95: "Гроза",
}


def _wmo_to_text(code: int) -> str:
    return _WMO_TEXTS.get(code, "Переменная погода")


class OnlineForecaster:
    """Fetches weather forecasts from the Open-Meteo API.

    Internet availability is cached for ``config.INTERNET_CHECK_CACHE_SEC``
    seconds to avoid hammering the network on every sensor cycle.

    After a successful :meth:`fetch` call, ``last_precip_probability``
    holds the maximum precipitation probability (0–100) seen in the
    forecast window so callers can display it separately.
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._last_check: float = 0.0
        self.last_precip_probability: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the Open-Meteo API is reachable.

        Result is cached for ``config.INTERNET_CHECK_CACHE_SEC`` seconds.
        Never raises an exception.
        """
        now = time.monotonic()
        if (
            self._available is not None
            and (now - self._last_check) < config.INTERNET_CHECK_CACHE_SEC
        ):
            return self._available

        try:
            resp = requests.get(
                config.INTERNET_CHECK_URL,
                timeout=config.INTERNET_CHECK_TIMEOUT,
            )
            self._available = resp.status_code < 500
        except Exception:
            self._available = False

        self._last_check = now
        logger.debug("Internet availability: %s", self._available)
        return self._available

    def fetch(
        self,
        lat: float,
        lon: float,
        altitude: float,
    ) -> Optional[ForecastResult]:
        """Fetch a 3-hour forecast from Open-Meteo.

        Args:
            lat:      Latitude in decimal degrees.
            lon:      Longitude in decimal degrees.
            altitude: GPS altitude in metres (passed as ``elevation`` to the
                      API for surface-pressure correction at station height).

        Returns:
            :class:`ForecastResult` with ``method="online_api"``, or ``None``
            on any error (network timeout, bad response, missing data, etc.).
        """
        try:
            params: Dict[str, Any] = {
                "latitude":       lat,
                "longitude":      lon,
                "elevation":      altitude,
                "hourly":         ",".join([
                    "temperature_2m",
                    "relativehumidity_2m",
                    "apparent_temperature",
                    "precipitation_probability",
                    "weathercode",
                    "surface_pressure",
                ]),
                "forecast_days":  1,
                "timezone":       "auto",
            }
            resp = requests.get(
                config.OPEN_METEO_BASE_URL,
                params=params,
                timeout=config.INTERNET_CHECK_TIMEOUT + 2,
            )
            resp.raise_for_status()
            data = resp.json()

            hourly = data["hourly"]
            times  = hourly["time"]   # "2026-05-10T14:00" strings

            # Find current-hour index
            now_str = datetime.now().strftime("%Y-%m-%dT%H:00")
            if now_str in times:
                idx = times.index(now_str)
            else:
                now_dt = datetime.now()
                idx = next(
                    (i for i, t in enumerate(times)
                     if datetime.fromisoformat(t) >= now_dt),
                    0,
                )

            n = config.API_FORECAST_HOURS

            def _get(key: str) -> list:
                return hourly[key][idx: idx + n]

            temps        = _get("temperature_2m")
            precip_probs = _get("precipitation_probability")
            wcodes       = _get("weathercode")
            pressures    = _get("surface_pressure")

            if len(temps) < n or len(pressures) < n:
                logger.warning(
                    "Open-Meteo returned fewer than %d hourly slots.", n
                )
                return None

            # Most frequent WMO code wins
            dominant_code = max(set(wcodes), key=wcodes.count)
            forecast_text = _wmo_to_text(int(dominant_code))

            # hPa/hour over the window
            pressure_trend = (pressures[-1] - pressures[0]) / max(n - 1, 1)

            # Store precip for callers to use
            self.last_precip_probability = float(max(precip_probs))

            return ForecastResult(
                method          = "online_api",
                forecast_text   = forecast_text,
                confidence      = 0.92,
                pressure_trend  = pressure_trend,
                temp_in_1h      = float(temps[0]) if len(temps) > 0 else None,
                temp_in_2h      = float(temps[1]) if len(temps) > 1 else None,
                temp_in_3h      = float(temps[2]) if len(temps) > 2 else None,
                pressure_in_1h  = float(pressures[0]) if len(pressures) > 0 else None,
                pressure_in_2h  = float(pressures[1]) if len(pressures) > 1 else None,
                pressure_in_3h  = float(pressures[2]) if len(pressures) > 2 else None,
                valid_until     = datetime.now() + timedelta(hours=n),
                model_version   = "open_meteo_v1",
            )

        except Exception as exc:
            logger.warning("Open-Meteo fetch failed: %s", exc)
            # Force recheck on next cycle
            self._available  = False
            self._last_check = 0.0
            return None

    def get_current_weather(
        self,
        lat: float,
        lon: float,
    ) -> Optional[Dict[str, Any]]:
        """Fetch current conditions for cross-validation with BME280.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Dict with keys ``temp``, ``humidity``, ``pressure``,
            ``weathercode``, ``windspeed``; or ``None`` on failure.
        """
        try:
            params: Dict[str, Any] = {
                "latitude":        lat,
                "longitude":       lon,
                "current_weather": True,
                "hourly":          "relativehumidity_2m,surface_pressure",
                "forecast_days":   1,
                "timezone":        "auto",
            }
            resp = requests.get(
                config.OPEN_METEO_BASE_URL,
                params=params,
                timeout=config.INTERNET_CHECK_TIMEOUT + 2,
            )
            resp.raise_for_status()
            data = resp.json()

            cw     = data.get("current_weather", {})
            hourly = data.get("hourly", {})
            times  = hourly.get("time", [])
            now_str = datetime.now().strftime("%Y-%m-%dT%H:00")
            idx = times.index(now_str) if now_str in times else 0

            return {
                "temp":        cw.get("temperature"),
                "humidity":    (hourly.get("relativehumidity_2m") or [None])[idx],
                "pressure":    (hourly.get("surface_pressure")    or [None])[idx],
                "weathercode": cw.get("weathercode"),
                "windspeed":   cw.get("windspeed"),
            }

        except Exception as exc:
            logger.warning("get_current_weather failed: %s", exc)
            return None
