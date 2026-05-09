"""
SQLite-backed time-series store for WeatherData readings.

Thread-safe: all DB operations are serialised through a single threading.Lock.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from typing import List

import config
from sensors.bme280_sensor import WeatherData
from utils.logger import get_logger

logger = get_logger("forecasting.data_store")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    temperature REAL    NOT NULL,
    humidity    REAL    NOT NULL,
    pressure    REAL    NOT NULL
)
"""


class DataStore:
    """Persistent store for sensor readings.

    Args:
        db_path: Path to the SQLite database file (default: config.DB_PATH).
    """

    def __init__(self, db_path: str = config.DB_PATH) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the readings table if it does not already exist."""
        try:
            with self._connect() as conn:
                conn.execute(_CREATE_TABLE)
        except sqlite3.Error as exc:
            logger.error("DataStore._init_db failed: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, check_same_thread=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, data: WeatherData) -> None:
        """Persist one WeatherData snapshot.

        Args:
            data: The reading to store.
        """
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO readings (timestamp, temperature, humidity, pressure)"
                        " VALUES (?, ?, ?, ?)",
                        (data.timestamp.isoformat(),
                         data.temperature,
                         data.humidity,
                         data.pressure),
                    )
            except sqlite3.Error as exc:
                logger.error("DataStore.save error: %s", exc)

    def get_last_n(self, n: int) -> List[WeatherData]:
        """Return the *n* most recent readings in chronological order.

        Args:
            n: Maximum number of readings to retrieve.

        Returns:
            List of WeatherData objects, oldest first.
        """
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT timestamp, temperature, humidity, pressure"
                        " FROM readings ORDER BY id DESC LIMIT ?",
                        (n,),
                    ).fetchall()
                return [
                    WeatherData(
                        temperature=row[1],
                        humidity=row[2],
                        pressure=row[3],
                        pressure_sl=row[3],
                        timestamp=datetime.fromisoformat(row[0]),
                    )
                    for row in reversed(rows)
                ]
            except sqlite3.Error as exc:
                logger.error("DataStore.get_last_n error: %s", exc)
                return []

    def get_all(self) -> List[WeatherData]:
        """Return every stored reading in chronological order.

        Returns:
            List of WeatherData objects, oldest first.
        """
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT timestamp, temperature, humidity, pressure"
                        " FROM readings ORDER BY id",
                    ).fetchall()
                return [
                    WeatherData(
                        temperature=row[1],
                        humidity=row[2],
                        pressure=row[3],
                        pressure_sl=row[3],
                        timestamp=datetime.fromisoformat(row[0]),
                    )
                    for row in rows
                ]
            except sqlite3.Error as exc:
                logger.error("DataStore.get_all error: %s", exc)
                return []

    def count(self) -> int:
        """Return the total number of stored readings.

        Returns:
            Row count, or 0 on error.
        """
        with self._lock:
            try:
                with self._connect() as conn:
                    return conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
            except sqlite3.Error as exc:
                logger.error("DataStore.count error: %s", exc)
                return 0
