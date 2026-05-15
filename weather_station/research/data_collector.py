"""
ResearchCollector — logs sensor readings, forecasts, verification errors,
LSTM training metrics and cycle timings to a dedicated SQLite database
for diploma thesis experiments.

Design constraints:
  - All public methods silently catch every exception — never affect main loop.
  - Thread-safe: one Lock guards all DB access.
  - Timestamps stored as UTC ISO-8601 strings.
  - DB file: research/research_data.db (alongside this module).
"""

from __future__ import annotations

import pathlib
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Optional

from utils.logger import get_logger

logger = get_logger("research.collector")

# DB lives inside the research/ folder
_DB_PATH = pathlib.Path(__file__).resolve().parent / "research_data.db"

_SENSOR_INTERVAL_SEC = 600   # log_sensor fires at most once per 10 minutes

# Column-name lookup for verify_forecasts — avoids dynamic f-strings over user data
_HOUR_COLS: dict[int, tuple] = {
    1: ("temp_1h", "pressure_1h", "error_temp_1h", "error_pressure_1h", "verified_1h",
        "signed_error_temp_1h", "signed_error_pres_1h"),
    2: ("temp_2h", "pressure_2h", "error_temp_2h", "error_pressure_2h", "verified_2h",
        "signed_error_temp_2h", "signed_error_pres_2h"),
    3: ("temp_3h", "pressure_3h", "error_temp_3h", "error_pressure_3h", "verified_3h",
        "signed_error_temp_3h", "signed_error_pres_3h"),
}

_DDL = """
CREATE TABLE IF NOT EXISTS sensor_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    temperature  REAL,
    humidity     REAL,
    pressure     REAL,
    pressure_sl  REAL,
    gps_lat      REAL,
    gps_lon      REAL,
    gps_altitude REAL,
    gps_fix      INTEGER
);

CREATE TABLE IF NOT EXISTS forecast_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL,
    mode           TEXT,
    temp_1h        REAL,
    temp_2h        REAL,
    temp_3h        REAL,
    precip_1h      REAL,
    precip_2h      REAL,
    precip_3h      REAL,
    pressure_1h    REAL,
    pressure_2h    REAL,
    pressure_3h    REAL,
    pressure_trend REAL,
    forecast_text  TEXT,
    confidence     REAL
);

CREATE TABLE IF NOT EXISTS forecast_verification (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id       INTEGER NOT NULL REFERENCES forecast_log(id),
    verified_at       TEXT,
    actual_temp       REAL,
    actual_pressure   REAL,
    error_temp_1h     REAL,
    error_temp_2h     REAL,
    error_temp_3h     REAL,
    error_pressure_1h REAL,
    error_pressure_2h REAL,
    error_pressure_3h REAL,
    verified_1h       INTEGER DEFAULT 0,
    verified_2h       INTEGER DEFAULT 0,
    verified_3h       INTEGER DEFAULT 0,
    signed_error_temp_1h  REAL,
    signed_error_temp_2h  REAL,
    signed_error_temp_3h  REAL,
    signed_error_pres_1h  REAL,
    signed_error_pres_2h  REAL,
    signed_error_pres_3h  REAL
);

CREATE TABLE IF NOT EXISTS lstm_training_log (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp             TEXT    NOT NULL,
    readings_count        INTEGER,
    mae_temp              REAL,
    mae_pressure          REAL,
    rmse_temp             REAL,
    rmse_pressure         REAL,
    training_duration_sec REAL
);

CREATE TABLE IF NOT EXISTS cycle_timing (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    bme280_ms   REAL,
    gps_ms      REAL,
    forecast_ms REAL,
    total_ms    REAL,
    mode_used   TEXT
);
"""


class ResearchCollector:
    """Logs research data to research_data.db for diploma thesis.

    All public methods are no-throw — exceptions are caught and logged.
    Thread-safe via a single threading.Lock on the SQLite connection.

    Args:
        db_path: Override DB file path (default: research/research_data.db).
    """

    def __init__(self, db_path: Optional[pathlib.Path] = None) -> None:
        self._db_path  = str(db_path or _DB_PATH)
        self._lock     = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._last_sensor_ts: Optional[datetime] = None

        try:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_DDL)
            for _col in ["signed_error_temp_1h", "signed_error_temp_2h", "signed_error_temp_3h",
                         "signed_error_pres_1h", "signed_error_pres_2h", "signed_error_pres_3h"]:
                try:
                    conn.execute(f"ALTER TABLE forecast_verification ADD COLUMN {_col} REAL")
                except Exception:
                    pass  # already exists
            conn.commit()
            self._conn = conn
            logger.info("ResearchCollector ready → %s", self._db_path)
        except Exception as exc:
            logger.error("ResearchCollector init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_sensor(self, weather, gps) -> None:
        """Save one row to sensor_log — at most once per 10 minutes."""
        try:
            now = datetime.utcnow()
            if (self._last_sensor_ts is not None and
                    (now - self._last_sensor_ts).total_seconds() < _SENSOR_INTERVAL_SEC):
                return

            self._run(
                "INSERT INTO sensor_log "
                "(timestamp, temperature, humidity, pressure, pressure_sl, "
                " gps_lat, gps_lon, gps_altitude, gps_fix) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    now.isoformat(),
                    weather.temperature if weather else None,
                    weather.humidity    if weather else None,
                    weather.pressure    if weather else None,
                    getattr(weather, "pressure_sl", None) if weather else None,
                    gps.latitude  if gps else None,
                    gps.longitude if gps else None,
                    gps.altitude  if gps else None,
                    int(gps.fix)  if gps else None,
                ),
            )
            self._last_sensor_ts = now
        except Exception as exc:
            logger.warning("log_sensor error: %s", exc)

    def log_forecast(self, forecast, weather) -> int:
        """Insert one row to forecast_log.  Returns the new row id (0 on error)."""
        try:
            # HybridForecastResult has .mode; plain ForecastResult has only .method
            mode = getattr(forecast, "mode", forecast.method)
            return self._run_returning_id(
                "INSERT INTO forecast_log "
                "(timestamp, mode, temp_1h, temp_2h, temp_3h, "
                " precip_1h, precip_2h, precip_3h, "
                " pressure_1h, pressure_2h, pressure_3h, "
                " pressure_trend, forecast_text, confidence) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(),
                    mode,
                    forecast.temp_in_1h,
                    forecast.temp_in_2h,
                    forecast.temp_in_3h,
                    forecast.precip_prob_1h,
                    forecast.precip_prob_2h,
                    forecast.precip_prob_3h,
                    forecast.pressure_in_1h,
                    forecast.pressure_in_2h,
                    forecast.pressure_in_3h,
                    forecast.pressure_trend,
                    forecast.forecast_text,
                    forecast.confidence,
                ),
            )
        except Exception as exc:
            logger.warning("log_forecast error: %s", exc)
            return 0

    def verify_forecasts(self, current_weather) -> None:
        """Compare old forecast_log entries against current readings.

        For each hour H in {1, 2, 3}: finds forecast_log rows whose
        timestamp is within ±2 minutes of (now − H hours).  If not yet
        verified for that horizon, writes the absolute prediction errors
        to forecast_verification.
        """
        if current_weather is None or self._conn is None:
            return
        try:
            now      = datetime.utcnow()
            act_temp = current_weather.temperature
            act_pres = current_weather.pressure

            with self._lock:
                for hours, (tc, pc, etc, epc, vc, setc, sepc) in _HOUR_COLS.items():
                    target = now - timedelta(hours=hours)
                    lower  = (target - timedelta(minutes=2)).isoformat()
                    upper  = (target + timedelta(minutes=2)).isoformat()

                    fc_rows = self._conn.execute(
                        f"SELECT id, {tc}, {pc} FROM forecast_log "
                        "WHERE timestamp BETWEEN ? AND ?",
                        (lower, upper),
                    ).fetchall()

                    for fc in fc_rows:
                        fid = fc["id"]

                        # Check whether a verification row exists and whether
                        # this hour is already marked verified.
                        vrow = self._conn.execute(
                            "SELECT id, " + vc + " FROM forecast_verification "
                            "WHERE forecast_id = ? LIMIT 1",
                            (fid,),
                        ).fetchone()
                        if vrow and vrow[vc]:
                            continue  # already verified for this horizon

                        pred_t = fc[tc]
                        pred_p = fc[pc]
                        err_t  = (round(abs(pred_t - act_temp), 4)
                                  if pred_t is not None else None)
                        err_p  = (round(abs(pred_p - act_pres), 4)
                                  if pred_p is not None else None)
                        now_s  = now.isoformat()

                        # Create exactly ONE verification row (if none exists).
                        # Do NOT use INSERT OR IGNORE — without a UNIQUE
                        # constraint on forecast_id it always inserts, producing
                        # duplicate rows that break the per-horizon checks.
                        if vrow is None:
                            self._conn.execute(
                                "INSERT INTO forecast_verification "
                                "(forecast_id) VALUES (?)",
                                (fid,),
                            )
                        # Fill in this hour's columns on the single row.
                        self._conn.execute(
                            f"UPDATE forecast_verification SET "
                            f"  verified_at=?, actual_temp=?, actual_pressure=?, "
                            f"  {etc}=?, {epc}=?, {vc}=1, "
                            f"  {setc}=?, {sepc}=? "
                            f"WHERE forecast_id=?",
                            (now_s, act_temp, act_pres, err_t, err_p,
                             round(act_temp - pred_t, 4) if pred_t is not None else None,
                             round(act_pres - pred_p, 4) if pred_p is not None else None,
                             fid),
                        )

                self._conn.commit()
        except Exception as exc:
            logger.warning("verify_forecasts error: %s", exc)

    def log_lstm_training(self, metrics: dict) -> None:
        """Insert one row to lstm_training_log after a retrain cycle."""
        try:
            self._run(
                "INSERT INTO lstm_training_log "
                "(timestamp, readings_count, mae_temp, mae_pressure, "
                " rmse_temp, rmse_pressure, training_duration_sec) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(),
                    metrics.get("readings_count"),
                    metrics.get("mae_temp"),
                    metrics.get("mae_pressure"),
                    metrics.get("rmse_temp"),
                    metrics.get("rmse_pressure"),
                    metrics.get("duration_sec"),
                ),
            )
        except Exception as exc:
            logger.warning("log_lstm_training error: %s", exc)

    def log_timing(self, timings: dict) -> None:
        """Insert one row to cycle_timing."""
        try:
            self._run(
                "INSERT INTO cycle_timing "
                "(timestamp, bme280_ms, gps_ms, forecast_ms, total_ms, mode_used) "
                "VALUES (?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(),
                    timings.get("bme280_ms"),
                    timings.get("gps_ms"),
                    timings.get("forecast_ms"),
                    timings.get("total_ms"),
                    timings.get("mode"),
                ),
            )
        except Exception as exc:
            logger.warning("log_timing error: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run(self, sql: str, params: tuple = ()) -> None:
        """Execute a write statement under the lock."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _run_returning_id(self, sql: str, params: tuple = ()) -> int:
        """Execute an INSERT and return lastrowid."""
        if self._conn is None:
            return 0
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid or 0
