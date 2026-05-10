"""
Weather Station — entry point.

Orchestrates sensor initialisation, the main polling loop, and clean
shutdown.  Run with::

    python main.py [--interval N] [--no-tft]

WiFi settings: press the button wired to GPIO 25 (physical pin 22).
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from typing import Optional

import config
from terminal.terminal_display import display as terminal_display
from forecasting import DataStore, LSTMForecaster, RuleForecaster
from forecasting import HybridForecaster, correct_pressure_to_sea_level
from forecasting.forecast_result import ForecastResult
from sensors.bme280_sensor import BME280Sensor, WeatherData
from sensors.gps_sensor import GPSSensor, GpsData
from utils.logger import get_logger

logger = get_logger("main")

try:
    from tft_display.display_module.tft_display import TFTDisplay
    _TFT_AVAILABLE = True
except ImportError:
    _TFT_AVAILABLE = False

try:
    from tft_display.display_module.wifi_screen import WiFiScreen
    _WIFI_AVAILABLE = True
except ImportError:
    _WIFI_AVAILABLE = False

# GPIO pin for the WiFi settings button (physical pin 22)
_WIFI_BUTTON_GPIO = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Weather Station — reads BME280 + GPS and displays data."
    )
    parser.add_argument(
        "--interval",
        metavar="N",
        type=int,
        default=config.UPDATE_INTERVAL,
        help=f"Seconds between screen updates (default: {config.UPDATE_INTERVAL})",
    )
    parser.add_argument(
        "--no-tft",
        action="store_true",
        help="Disable TFT display even if luma.lcd is available",
    )
    return parser.parse_args()


def _safe_read_weather(sensor: BME280Sensor, altitude_m: float) -> Optional[WeatherData]:
    try:
        return sensor.read(altitude_m=altitude_m)
    except Exception as exc:
        logger.error("Unexpected error reading BME280: %s", exc)
        return None


def _safe_read_gps(sensor: GPSSensor) -> Optional[GpsData]:
    try:
        return sensor.read()
    except Exception as exc:
        logger.error("Unexpected error reading GPS: %s", exc)
        return None


def main() -> None:
    args = parse_args()
    interval: int = args.interval

    logger.info("Weather Station starting (update interval: %d s).", interval)

    # ── Sensors ───────────────────────────────────────────────────────────────
    bme = BME280Sensor()
    gps = GPSSensor()
    bme_ok = gps_ok = False

    try:
        bme.connect()
        bme_ok = True
    except OSError as exc:
        logger.error("BME280 connection failed: %s", exc)

    try:
        gps.connect()
        gps_ok = True
    except Exception as exc:
        logger.error("GPS connection failed: %s", exc)

    if not bme_ok and not gps_ok:
        logger.critical("Both sensors failed to initialise — exiting.")
        sys.exit(1)

    # ── Forecasting ───────────────────────────────────────────────────────────
    data_store      = DataStore()
    lstm_forecaster = LSTMForecaster(data_store)
    rule_forecaster = RuleForecaster()
    hybrid          = HybridForecaster(data_store, lstm_forecaster, rule_forecaster)

    # ── TFT display ───────────────────────────────────────────────────────────
    tft: Optional[TFTDisplay] = None
    if _TFT_AVAILABLE and not args.no_tft:
        try:
            tft = TFTDisplay()
            logger.info("TFT display initialised.")
        except Exception as exc:
            logger.warning("TFT display init failed (continuing without it): %s", exc)

    # ── WiFi button ───────────────────────────────────────────────────────────
    _wifi_requested = threading.Event()
    if tft is not None and _WIFI_AVAILABLE:
        try:
            from gpiozero import Button
            wifi_btn = Button(_WIFI_BUTTON_GPIO, pull_up=True, bounce_time=0.1)
            wifi_btn.when_pressed = lambda: _wifi_requested.set()
            logger.info("WiFi button active on GPIO %d.", _WIFI_BUTTON_GPIO)
        except Exception as exc:
            logger.warning("WiFi button setup failed: %s", exc)

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # WiFi settings screen
            if _wifi_requested.is_set() and tft is not None and _WIFI_AVAILABLE:
                _wifi_requested.clear()
                logger.info("Entering WiFi settings screen.")
                try:
                    WiFiScreen(tft).run()
                except Exception as exc:
                    logger.error("WiFi screen error: %s", exc)
                continue

            # Read sensors
            gps_data: Optional[GpsData] = _safe_read_gps(gps) if gps_ok else None

            altitude_m: float = 0.0
            if gps_data is not None and gps_data.fix and gps_data.altitude is not None:
                altitude_m = gps_data.altitude

            weather_data: Optional[WeatherData] = (
                _safe_read_weather(bme, altitude_m) if bme_ok else None
            )

            # Correct sea-level pressure explicitly from GPS altitude
            if (weather_data is not None
                    and gps_data is not None
                    and gps_data.fix
                    and gps_data.altitude):
                weather_data.pressure_sl = round(
                    correct_pressure_to_sea_level(
                        weather_data.pressure, gps_data.altitude
                    ),
                    2,
                )

            # Persist reading
            if weather_data is not None:
                try:
                    data_store.save(weather_data)
                except Exception as exc:
                    logger.error("DataStore.save error: %s", exc)

            # Forecast (hybrid: online API → LSTM → rules)
            forecast: Optional[ForecastResult] = None
            data_count: int = 0
            try:
                data_count = data_store.count()
                forecast   = hybrid.predict(
                    gps_data,
                    data_store.get_last_n(config.SEQUENCE_LENGTH),
                    current_weather=weather_data,
                )
                lstm_forecaster._retrain_if_needed()
            except Exception as exc:
                logger.error("Forecast error: %s", exc)

            # Terminal display
            terminal_display(weather_data, gps_data, forecast, data_count)

            # TFT display
            if tft is not None:
                tft_data = {
                    'temperature': weather_data.temperature if weather_data else None,
                    'humidity':    weather_data.humidity    if weather_data else None,
                    'pressure':    weather_data.pressure    if weather_data else None,
                    'latitude':    gps_data.latitude        if gps_data else None,
                    'longitude':   gps_data.longitude       if gps_data else None,
                    'altitude':    gps_data.altitude        if gps_data else None,
                    'gps_fix':     gps_data.fix             if gps_data else False,
                    'timestamp':   time.time(),
                    'forecast':    forecast,
                    'data_count':  data_count,
                }
                try:
                    tft.render(tft_data)
                except Exception as exc:
                    logger.warning("TFT render error: %s", exc)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStation stopped.")
        logger.info("Keyboard interrupt received — shutting down.")
    finally:
        if tft is not None:
            tft.close()
        if bme_ok:
            bme.close()
        if gps_ok:
            gps.close()
        logger.info("Weather Station shut down cleanly.")


if __name__ == "__main__":
    main()
