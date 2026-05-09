"""
Terminal display renderer for the Weather Station.

Clears the screen and redraws the data table on every call to
:func:`display`.  Unicode box-drawing characters are used for the frame.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sensors.bme280_sensor import WeatherData
from sensors.gps_sensor import GpsData

# Box-drawing constants — defined once to keep the render functions readable
_TL = "╔"   # top-left corner
_TR = "╗"   # top-right corner
_BL = "╚"   # bottom-left corner
_BR = "╝"   # bottom-right corner
_H  = "═"   # horizontal bar
_V  = "║"   # vertical bar
_ML = "╠"   # middle-left tee
_MR = "╣"   # middle-right tee

_WIDTH = 36             # total inner width (excluding the two vertical bars)
_INNER = _WIDTH - 2     # usable text width


def _hline(left: str = _ML, right: str = _MR) -> str:
    return f"{left}{_H * _WIDTH}{right}"


def _row(text: str) -> str:
    """Pad *text* to _INNER characters and wrap in vertical bars."""
    return f"{_V} {text:<{_INNER - 1}}{_V}"


def _fmt_float(value: Optional[float], unit: str, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f} {unit}"


def _fmt_bool(value: bool) -> str:
    return "YES" if value else "NO"


def display(
    weather: Optional[WeatherData],
    gps: Optional[GpsData],
    forecast=None,
    data_count: int = 0,
) -> None:
    """Clear the terminal and render the weather/GPS/forecast table.

    Args:
        weather:    Latest BME280 reading, or ``None`` if the sensor failed.
        gps:        Latest GPS reading, or ``None`` if the sensor failed.
        forecast:   ForecastResult from LSTM or rule engine, or None.
        data_count: Current number of rows in the history DB (for status line).
    """
    os.system("clear")  # noqa: S605 – intentional terminal clear

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(_hline(_TL, _TR))
    lines.append(_row("🌤  WEATHER STATION"))
    lines.append(_hline())

    # ── Weather section ─────────────────────────────────────────────────
    if weather is not None:
        lines.append(_row(f"Temperature  : {_fmt_float(weather.temperature, '°C')}"))
        lines.append(_row(f"Humidity     : {_fmt_float(weather.humidity, '%')}"))
        lines.append(_row(f"Pressure     : {_fmt_float(weather.pressure, 'hPa')}"))
        lines.append(_row(f"Pressure SL  : {_fmt_float(weather.pressure_sl, 'hPa')}"))
    else:
        lines.append(_row("Temperature  : N/A"))
        lines.append(_row("Humidity     : N/A"))
        lines.append(_row("Pressure     : N/A"))
        lines.append(_row("Pressure SL  : N/A"))
        lines.append(_row("⚠  WEATHER SENSOR ERROR"))

    lines.append(_hline())

    # ── GPS section ──────────────────────────────────────────────────────
    if gps is not None:
        lines.append(_row(f"GPS Fix      : {_fmt_bool(gps.fix)}"))
        lines.append(_row(f"Latitude     : {_fmt_float(gps.latitude, '°', 6)}"))
        lines.append(_row(f"Longitude    : {_fmt_float(gps.longitude, '°', 6)}"))
        lines.append(_row(f"Altitude     : {_fmt_float(gps.altitude, 'm', 1)}"))
        lines.append(_row(f"Satellites   : {gps.satellites if gps.fix else 'N/A'}"))
        if not gps.fix:
            lines.append(_row("⚠  WAITING FOR GPS FIX"))
    else:
        lines.append(_row("GPS Fix      : N/A"))
        lines.append(_row("Latitude     : N/A"))
        lines.append(_row("Longitude    : N/A"))
        lines.append(_row("Altitude     : N/A"))
        lines.append(_row("Satellites   : N/A"))
        lines.append(_row("⚠  GPS SENSOR ERROR"))

    lines.append(_hline())

    # ── Forecast section ──────────────────────────────────────────────────
    if forecast is not None:
        _append_forecast(lines, forecast, data_count)
    else:
        lines.append(_row("🔮 ПРОГНОЗ: нет данных"))

    lines.append(_hline())

    # ── Footer: last update timestamp ───────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(_row(f"Updated      : {now}"))
    lines.append(_hline(_BL, _BR))

    print("\n".join(lines))


def _append_forecast(lines: list[str], forecast, data_count: int) -> None:
    """Append forecast rows to *lines* depending on method."""
    method = forecast.method

    if method == "lstm":
        lines.append(_row("🔮 ПРОГНОЗ (LSTM)"))
        lines.append(_row(forecast.forecast_text[:_INNER - 1]))

        trend = forecast.pressure_trend
        arrow = "▲" if trend >= 0 else "▼"
        lines.append(_row(f"Давление: {arrow} {trend:+.1f} hPa"))

        if forecast.temp_in_1h is not None:
            lines.append(_row(f"Темп +1ч : {forecast.temp_in_1h:.1f}°C"))
        if forecast.temp_in_2h is not None:
            lines.append(_row(f"Темп +2ч : {forecast.temp_in_2h:.1f}°C"))
        if forecast.temp_in_3h is not None:
            lines.append(_row(f"Темп +3ч : {forecast.temp_in_3h:.1f}°C"))

        lines.append(_row(f"Точность : {forecast.confidence * 100:.0f}%"))
        lines.append(_row(f"До       : {forecast.valid_until.strftime('%H:%M')}"))

    elif method == "rule-based":
        lines.append(_row("🔮 ПРОГНОЗ (правила, сбор данных)"))
        lines.append(_row(forecast.forecast_text[:_INNER - 1]))

        import config
        lines.append(_row(f"Накоплено: {data_count}/{config.FORECAST_MIN_READINGS}"))

        trend = forecast.pressure_trend
        arrow = "▲" if trend >= 0 else "▼"
        lines.append(_row(f"Давление: {arrow} {trend:+.1f} hPa/ч"))

    else:  # insufficient_data
        lines.append(_row("🔮 ПРОГНОЗ: сбор данных..."))
        import config
        lines.append(_row(f"Накоплено: {data_count}/{config.FORECAST_MIN_READINGS}"))
