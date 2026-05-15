"""
GY-NEO6M-V2 GPS sensor driver.

Reads NMEA sentences from a UART serial port using ``pyserial`` and
parses them with ``pynmea2``.  Both GPGGA and GPRMC sentence types are
handled so that all GpsData fields can be populated from a single pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import serial
import pynmea2

import config
from utils.logger import get_logger

logger = get_logger("sensors.gps")


@dataclass
class GpsData:
    """Immutable snapshot of a single GPS reading."""

    latitude: Optional[float]    # decimal degrees, positive = North
    longitude: Optional[float]   # decimal degrees, positive = East
    altitude: Optional[float]    # metres above mean sea level
    satellites: int              # number of satellites in use
    fix: bool                    # True when the receiver has a valid fix
    timestamp: datetime = field(default_factory=datetime.now)


class GPSSensor:
    """UART driver for the GY-NEO6M-V2 GPS module.

    Usage::

        gps = GPSSensor()
        gps.connect()
        data = gps.read()
        gps.close()

    Or use as a context manager::

        with GPSSensor() as gps:
            data = gps.read()
    """

    def __init__(self) -> None:
        self._serial: Optional[serial.Serial] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the UART serial port connected to the GPS module.

        Raises:
            serial.SerialException: If the port cannot be opened.
        """
        logger.info(
            "Opening GPS serial port %s @ %d baud",
            config.GPS_PORT,
            config.GPS_BAUDRATE,
        )
        try:
            self._serial = serial.Serial(
                port=config.GPS_PORT,
                baudrate=config.GPS_BAUDRATE,
                timeout=config.GPS_TIMEOUT,
            )
            logger.info("GPS serial port opened successfully.")
        except serial.SerialException as exc:
            logger.error("Failed to open GPS serial port: %s", exc)
            raise

    def close(self) -> None:
        """Close the UART serial port."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self._serial = None
            logger.info("GPS serial port closed.")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> Optional[GpsData]:
        """Block until a valid GPS fix is obtained or the timeout expires.

        Reads NMEA sentences for up to :data:`config.GPS_FIX_TIMEOUT` seconds,
        accumulating data from GPGGA and GPRMC sentences.  Returns as soon
        as a GGA sentence with fix_quality > 0 is found.

        Returns:
            A :class:`GpsData` instance.  ``fix`` is ``False`` and all
            coordinate fields are ``None`` if no fix was obtained within the
            timeout.  Returns ``None`` only on a hard serial error.
        """
        if self._serial is None:
            logger.error("GPSSensor.read() called before connect().")
            return None

        deadline = time.monotonic() + config.GPS_FIX_TIMEOUT

        # Accumulate partial data from different sentence types
        lat: Optional[float] = None
        lon: Optional[float] = None
        alt: Optional[float] = None
        sats: int = 0
        fix: bool = False

        while time.monotonic() < deadline:
            try:
                raw_line = self._serial.readline()
            except serial.SerialException as exc:
                logger.error("GPS serial read error: %s", exc)
                return None

            if not raw_line:
                continue

            line = raw_line.decode("ascii", errors="replace").strip()

            if not line.startswith("$"):
                continue

            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                logger.debug("Unparseable NMEA sentence: %s", line)
                continue

            # --- GPGGA: position, altitude, satellite count, fix quality ---
            if isinstance(msg, pynmea2.types.talker.GGA):
                fix_quality: int = int(msg.gps_qual) if msg.gps_qual else 0
                if fix_quality > 0:
                    lat = self._to_decimal(msg.lat, msg.lat_dir)
                    lon = self._to_decimal(msg.lon, msg.lon_dir)
                    alt = float(msg.altitude) if msg.altitude else None
                    sats = int(msg.num_sats) if msg.num_sats else 0
                    fix = True
                    logger.debug(
                        "GPGGA fix: lat=%.6f lon=%.6f alt=%s sats=%d",
                        lat, lon, alt, sats,
                    )
                    # GGA has everything we need — return immediately
                    return GpsData(
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        satellites=sats,
                        fix=fix,
                        timestamp=datetime.utcnow(),
                    )

            # --- GPRMC: fallback for lat/lon when GGA has no fix yet ---
            elif isinstance(msg, pynmea2.types.talker.RMC):
                if msg.status == "A":  # A = Active (valid fix)
                    lat = self._to_decimal(msg.lat, msg.lat_dir)
                    lon = self._to_decimal(msg.lon, msg.lon_dir)
                    fix = True
                    logger.debug("GPRMC fix: lat=%.6f lon=%.6f", lat, lon)

        # Timeout reached with no confirmed fix
        logger.warning(
            "GPS fix not obtained within %d seconds.", config.GPS_FIX_TIMEOUT
        )
        return GpsData(
            latitude=None,
            longitude=None,
            altitude=None,
            satellites=0,
            fix=False,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_decimal(raw: str, direction: str) -> float:
        """Convert an NMEA coordinate string to a signed decimal degree.

        NMEA format: ``DDDMM.MMMM`` where DDD are degrees and MM.MMMM are
        minutes.

        Args:
            raw: Raw NMEA coordinate string (e.g. ``"5545.3480"``).
            direction: Hemisphere indicator: ``"N"``, ``"S"``, ``"E"``, or ``"W"``.

        Returns:
            Signed decimal degrees.
        """
        if not raw:
            return 0.0
        # Split at the last two digits before the decimal point (minutes)
        dot_pos = raw.index(".")
        degrees = float(raw[: dot_pos - 2])
        minutes = float(raw[dot_pos - 2 :])
        decimal = degrees + minutes / 60.0
        if direction in ("S", "W"):
            decimal = -decimal
        return decimal

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "GPSSensor":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
