"""
BME280 temperature / humidity / pressure sensor driver.

Uses the ``smbus2`` and ``bme280`` libraries to communicate with the sensor
over I2C.  Pressure is optionally compensated to sea level using the GPS
altitude when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import smbus2
import bme280

import config
from utils.logger import get_logger

logger = get_logger("sensors.bme280")

# Sea-level pressure compensation exponent (hypsometric formula)
_HYPS_EXP: float = -5.2561
_HYPS_LAPSE: float = 0.0065
_HYPS_T0: float = 288.15


@dataclass
class WeatherData:
    """Immutable snapshot of a single BME280 reading."""

    temperature: float          # degrees Celsius, rounded to 2 dp
    humidity: float             # percent, rounded to 2 dp
    pressure: float             # hPa at measurement altitude, rounded to 2 dp
    pressure_sl: float          # hPa corrected to sea level, rounded to 2 dp
    timestamp: datetime = field(default_factory=datetime.now)


class BME280Sensor:
    """Wrapper around the BME280 I2C sensor.

    Usage::

        sensor = BME280Sensor()
        sensor.connect()
        data = sensor.read()
        sensor.close()

    Or use as a context manager::

        with BME280Sensor() as sensor:
            data = sensor.read()
    """

    def __init__(self) -> None:
        self._bus: Optional[smbus2.SMBus] = None
        self._calibration: Optional[bme280.params] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the I2C bus and load BME280 calibration parameters.

        Raises:
            OSError: If the sensor is not found on the I2C bus.
        """
        logger.info(
            "Connecting to BME280 on I2C bus %d, address 0x%02X",
            config.I2C_BUS,
            config.BME280_ADDRESS,
        )
        self._bus = smbus2.SMBus(config.I2C_BUS)
        self._calibration = bme280.load_calibration_params(
            self._bus, config.BME280_ADDRESS
        )
        logger.info("BME280 connected successfully.")

    def close(self) -> None:
        """Release the I2C bus resource."""
        if self._bus is not None:
            self._bus.close()
            self._bus = None
            logger.info("BME280 I2C bus closed.")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self, altitude_m: float = 0.0) -> Optional[WeatherData]:
        """Read temperature, humidity, and pressure from the sensor.

        Args:
            altitude_m: Current altitude in metres (from GPS) used to
                compensate pressure to sea level.  Defaults to 0 (no
                compensation).

        Returns:
            A :class:`WeatherData` instance, or ``None`` if the read failed.
        """
        if self._bus is None or self._calibration is None:
            logger.error("BME280 read() called before connect().")
            return None

        try:
            raw = bme280.sample(self._bus, config.BME280_ADDRESS, self._calibration)
        except OSError as exc:
            logger.error("BME280 OSError during read: %s", exc)
            return None

        pressure_raw = round(raw.pressure, 2)
        pressure_sl = self._compensate_pressure(pressure_raw, altitude_m)

        return WeatherData(
            temperature=round(raw.temperature, 2),
            humidity=round(raw.humidity, 2),
            pressure=pressure_raw,
            pressure_sl=round(pressure_sl, 2),
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compensate_pressure(pressure_hpa: float, altitude_m: float) -> float:
        """Convert station pressure to sea-level pressure (hypsometric formula).

        Formula:
            P0 = P * (1 - (0.0065 * h) / 288.15) ^ (-5.2561)

        Args:
            pressure_hpa: Measured pressure in hPa.
            altitude_m: Station altitude in metres.

        Returns:
            Sea-level pressure in hPa.
        """
        if altitude_m == 0.0:
            return pressure_hpa
        factor = (1.0 - (_HYPS_LAPSE * altitude_m) / _HYPS_T0) ** _HYPS_EXP
        return pressure_hpa * factor

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BME280Sensor":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
