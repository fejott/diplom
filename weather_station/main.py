"""
Weather Station — entry point.

Orchestrates sensor initialisation, the main polling loop, and clean
shutdown.  Run with::

    python main.py [--interval N] [--no-tft]
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import config
from display.terminal_display import display as terminal_display
from sensors.bme280_sensor import BME280Sensor, WeatherData
from sensors.gps_sensor import GPSSensor, GpsData
from utils.logger import get_logger

logger = get_logger("main")

try:
    from display.tft_display import TFTDisplay
    _TFT_AVAILABLE = True
except ImportError:
    _TFT_AVAILABLE = False


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

    bme = BME280Sensor()
    gps = GPSSensor()

    bme_ok = False
    gps_ok = False

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

    tft: Optional[TFTDisplay] = None
    if _TFT_AVAILABLE and not args.no_tft:
        try:
            tft = TFTDisplay()
            logger.info("TFT display initialised.")
        except Exception as exc:
            logger.warning("TFT display init failed (continuing without it): %s", exc)

    try:
        while True:
            gps_data: Optional[GpsData] = _safe_read_gps(gps) if gps_ok else None

            altitude_m: float = 0.0
            if gps_data is not None and gps_data.fix and gps_data.altitude is not None:
                altitude_m = gps_data.altitude

            weather_data: Optional[WeatherData] = (
                _safe_read_weather(bme, altitude_m) if bme_ok else None
            )

            terminal_display(weather_data, gps_data)

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
