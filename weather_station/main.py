"""
Weather Station — entry point.

Orchestrates sensor initialisation, the main polling loop, and clean
shutdown.  Run with::

    python main.py [--interval N]
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import config
from display.terminal_display import display
from sensors.bme280_sensor import BME280Sensor, WeatherData
from sensors.gps_sensor import GPSSensor, GpsData
from utils.logger import get_logger

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with attribute ``interval`` (int seconds).
    """
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
    return parser.parse_args()


def _safe_read_weather(sensor: BME280Sensor, altitude_m: float) -> Optional[WeatherData]:
    """Attempt to read from the BME280; return None on any error."""
    try:
        return sensor.read(altitude_m=altitude_m)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error reading BME280: %s", exc)
        return None


def _safe_read_gps(sensor: GPSSensor) -> Optional[GpsData]:
    """Attempt to read from the GPS; return None on any error."""
    try:
        return sensor.read()
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error reading GPS: %s", exc)
        return None


def main() -> None:
    """Main entry point — initialise sensors, run polling loop, shut down cleanly."""
    args = parse_args()
    interval: int = args.interval

    logger.info("Weather Station starting (update interval: %d s).", interval)

    bme = BME280Sensor()
    gps = GPSSensor()

    bme_ok = False
    gps_ok = False

    # Attempt sensor connections — continue even if one fails
    try:
        bme.connect()
        bme_ok = True
    except OSError as exc:
        logger.error("BME280 connection failed: %s", exc)

    try:
        gps.connect()
        gps_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.error("GPS connection failed: %s", exc)

    if not bme_ok and not gps_ok:
        logger.critical("Both sensors failed to initialise — exiting.")
        sys.exit(1)

    try:
        while True:
            # Read GPS first so its altitude can feed the BME280 pressure compensation
            gps_data: Optional[GpsData] = _safe_read_gps(gps) if gps_ok else None

            altitude_m: float = 0.0
            if gps_data is not None and gps_data.fix and gps_data.altitude is not None:
                altitude_m = gps_data.altitude

            weather_data: Optional[WeatherData] = (
                _safe_read_weather(bme, altitude_m) if bme_ok else None
            )

            display(weather_data, gps_data)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStation stopped.")
        logger.info("Keyboard interrupt received — shutting down.")
    finally:
        if bme_ok:
            bme.close()
        if gps_ok:
            gps.close()
        logger.info("Weather Station shut down cleanly.")


if __name__ == "__main__":
    main()
