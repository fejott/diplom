"""
GPS-based pressure correction and coordinate validation utilities.

Pure functions — no state, no dependencies on the rest of the project
(except the GpsData type hint, which is imported lazily to avoid
circular imports).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sensors.gps_sensor import GpsData

# Hypsometric formula constants
_HYPS_LAPSE: float = 0.0065   # temperature lapse rate, K/m
_HYPS_T0: float    = 288.15   # reference temperature, K
_HYPS_EXP: float   = -5.2561  # barometric exponent


def correct_pressure_to_sea_level(
    pressure_hpa: float,
    altitude_m: float,
) -> float:
    """Convert station pressure to sea-level pressure (hypsometric formula).

    Formula:
        P0 = P * (1 - (0.0065 * h) / 288.15) ^ (-5.2561)

    Args:
        pressure_hpa: Measured pressure in hPa at station altitude.
        altitude_m:   Station altitude in metres above mean sea level.

    Returns:
        Sea-level pressure in hPa.  Returns *pressure_hpa* unchanged when
        *altitude_m* is None, 0, or effectively 0.
    """
    if not altitude_m:
        return pressure_hpa
    factor = (1.0 - (_HYPS_LAPSE * altitude_m) / _HYPS_T0) ** _HYPS_EXP
    return pressure_hpa * factor


def validate_gps_for_forecast(gps: Optional["GpsData"]) -> bool:
    """Return True only when GPS data is valid enough to drive an API request.

    Checks:
        - gps is not None
        - gps.fix is True
        - latitude and longitude are present and within valid ranges
        - altitude is present and within a realistic range

    Args:
        gps: GpsData snapshot, or None.

    Returns:
        True if the fix is valid and all coordinates are plausible.
    """
    if gps is None:
        return False
    if not gps.fix:
        return False
    if gps.latitude is None or gps.longitude is None:
        return False
    if not (abs(gps.latitude) <= 90 and abs(gps.longitude) <= 180):
        return False
    if gps.altitude is None or not (-500 < gps.altitude < 9000):
        return False
    return True


def format_coordinates(gps: Optional["GpsData"]) -> str:
    """Return a human-readable coordinate string.

    Args:
        gps: GpsData snapshot, or None.

    Returns:
        E.g. ``"55.7558°N, 37.6173°E"`` or ``"GPS: no fix"``.
    """
    if gps is None or not gps.fix or gps.latitude is None or gps.longitude is None:
        return "GPS: no fix"
    ns = "N" if gps.latitude  >= 0 else "S"
    ew = "E" if gps.longitude >= 0 else "W"
    return f"{abs(gps.latitude):.4f}°{ns}, {abs(gps.longitude):.4f}°{ew}"
