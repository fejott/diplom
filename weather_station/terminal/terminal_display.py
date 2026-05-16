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


def _precip_emoji(prob: float) -> str:
    if prob >= 0.7: return "🌧"
    if prob >= 0.4: return "🌦"
    if prob >= 0.2: return "🌤"
    return "☀"


def _pressure_arrow(trend: float) -> str:
    if trend > 0.5:  return "▲"
    if trend < -0.5: return "▼"
    return "—"


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
    """Append a unified forecast table to *lines* (identical structure for all modes)."""
    method   = forecast.method
    internet = getattr(forecast, 'internet_available', None)
    gps_used = getattr(forecast, 'gps_used', False)
    lstm_rdy = getattr(forecast, 'lstm_ready', True)

    # ── Mode header ─────────────────────────────────────────────────────────
    if method == "online_api":
        lines.append(_row("🌐 ПРОГНОЗ (Online API)"))
    elif method in ("lstm", "lstm_corrected"):
        if method == "lstm_corrected":
            lines.append(_row("🤖 LSTM+коррекция"))
        else:
            lines.append(_row("🤖 ПРОГНОЗ (LSTM (ERA5))"))
    elif method == "rule-based":
        import config as _cfg
        if internet is False and _cfg.ONLINE_FORECAST_ENABLED:
            lines.append(_row("📊 ПРОГНОЗ (нет интернета)"))
        else:
            lines.append(_row("📊 ПРОГНОЗ (По правилам)"))
    else:  # insufficient_data
        import config as _cfg
        lines.append(_row("📡 ПРОГНОЗ: сбор данных..."))
        lines.append(_row(f"Накоплено: {data_count}/{_cfg.FORECAST_MIN_READINGS}"))
        return

    # ── +1h / +2h / +3h table (same for all active modes) ──────────────────
    lines.append(_row(f" {'Вр':<4}  {'Темп':>6}   Осадки"))
    for h, temp, prob in (
        (1, forecast.temp_in_1h, forecast.precip_prob_1h),
        (2, forecast.temp_in_2h, forecast.precip_prob_2h),
        (3, forecast.temp_in_3h, forecast.precip_prob_3h),
    ):
        t_str = f"{temp:.1f}°C" if temp is not None else "  N/A"
        if prob is not None:
            p_str = f"{_precip_emoji(prob)}{round(prob * 100):3d}%"
        else:
            p_str = "   N/A"
        lines.append(_row(f" +{h}ч   {t_str:>6}   {p_str}"))

    # ── Correction delta (if applied) ────────────────────────────────────────
    correction_applied = getattr(forecast, 'correction_applied', False)
    if correction_applied:
        delta_t = getattr(forecast, 'correction_delta_temp_1h', None)
        delta_p = getattr(forecast, 'correction_delta_pres_1h', None)
        parts = []
        if delta_t is not None:
            parts.append(f"{'%+.1f' % delta_t}°C")
        if delta_p is not None:
            parts.append(f"{'%+.1f' % delta_p}hPa")
        if parts:
            lines.append(_row(f"  Коррекция +1ч: {' / '.join(parts)}"))

    # ── Pressure trend ───────────────────────────────────────────────────────
    arrow = _pressure_arrow(forecast.pressure_trend)
    lines.append(_row(f"Давление: {arrow} {forecast.pressure_trend:+.1f} hPa/ч"))

    # ── Forecast text ────────────────────────────────────────────────────────
    lines.append(_row(f"Прогноз:  {forecast.forecast_text[:_INNER - 11]}"))

    # ── Confidence + valid until ─────────────────────────────────────────────
    lines.append(_row(
        f"Точность: {forecast.confidence * 100:.0f}%"
        f"  До: {forecast.valid_until.strftime('%H:%M')}"
    ))

    # ── Data counter (rule-based only, while LSTM not ready) ─────────────────
    if method == "rule-based" and not lstm_rdy:
        import config as _cfg
        lines.append(_row(f"Накоплено: {data_count}/{_cfg.FORECAST_MIN_READINGS}"))
