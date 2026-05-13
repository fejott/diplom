"""
ReportGenerator — reads research_data.db and produces formatted text
reports and CSV exports for the diploma thesis.

All queries use only the standard library (sqlite3 + math).
No pandas required — keeps the Pi dependency footprint small.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sqlite3
from datetime import datetime
from typing import Optional

_DB_PATH    = pathlib.Path(__file__).resolve().parent / "research_data.db"
_CSV_DIR    = pathlib.Path(__file__).resolve().parent
_SEP        = ";"   # semicolon — correct for Russian Excel locale
_MIN_VERIFY = 10    # skip mode in accuracy report if fewer verified forecasts


def _conn(db_path: pathlib.Path) -> sqlite3.Connection:
    """Open (and if necessary create) the research DB."""
    # Import DDL from data_collector so tables always exist, even when
    # the CLI is run before main.py has started.
    from research.data_collector import _DDL
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    c.executescript(_DDL)
    c.commit()
    return c


def _std(avg_sq: Optional[float], avg: Optional[float], n: int) -> Optional[float]:
    """Population std-dev: sqrt(E[x²] - E[x]²).  avg_sq = AVG(x*x) from SQL."""
    if n < 2 or avg_sq is None or avg is None:
        return None
    return math.sqrt(max(0.0, avg_sq - avg ** 2))


def _rmse(mse: Optional[float]) -> Optional[float]:
    """RMSE from MSE (AVG of squared errors) already computed by SQL."""
    if mse is None:
        return None
    return math.sqrt(mse)


class ReportGenerator:
    """Reads research_data.db and formats reports for the diploma thesis.

    Args:
        db_path: Override DB file path (default: research/research_data.db).
    """

    def __init__(self, db_path: Optional[pathlib.Path] = None) -> None:
        self._db_path  = db_path or _DB_PATH
        self._csv_dir  = self._db_path.parent

    # ── Public report methods ─────────────────────────────────────────────────

    def sensor_summary(self) -> str:
        """Period, total readings, temp/humidity/pressure stats, GPS fix %."""
        with _conn(self._db_path) as c:
            row = c.execute("""
                SELECT
                    MIN(timestamp)                          AS first_ts,
                    MAX(timestamp)                          AS last_ts,
                    COUNT(*)                                AS n,
                    MIN(temperature)  AS t_min,
                    AVG(temperature)  AS t_avg,
                    MAX(temperature)  AS t_max,
                    AVG(temperature*temperature)            AS t_sq,
                    MIN(humidity)     AS h_min,
                    AVG(humidity)     AS h_avg,
                    MAX(humidity)     AS h_max,
                    AVG(humidity*humidity)                  AS h_sq,
                    MIN(pressure)     AS p_min,
                    AVG(pressure)     AS p_avg,
                    MAX(pressure)     AS p_max,
                    AVG(pressure*pressure)                  AS p_sq,
                    SUM(CASE WHEN gps_fix=1 THEN 1 ELSE 0 END) * 100.0
                        / NULLIF(COUNT(*), 0)              AS gps_pct
                FROM sensor_log
            """).fetchone()

        if not row or not row["n"]:
            return "sensor_log: нет данных."

        n = row["n"]
        t_std = _std(row["t_sq"], row["t_avg"], n)
        h_std = _std(row["h_sq"], row["h_avg"], n)
        p_std = _std(row["p_sq"], row["p_avg"], n)

        lines = [
            "═" * 50,
            "  СВОДКА ПО СЕНСОРАМ",
            "═" * 50,
            f"  Период:    {row['first_ts'][:16]} → {row['last_ts'][:16]}",
            f"  Измерений: {n}",
            "",
            f"  {'Пар.':<12} {'Мин':>8} {'Ср.':>8} {'Макс':>8} {'СКО':>8}",
            "  " + "─" * 46,
            f"  {'Темп, °C':<12} "
            f"{row['t_min']:>8.2f} {row['t_avg']:>8.2f} {row['t_max']:>8.2f} "
            f"{t_std:>8.2f}",
            f"  {'Влаж., %':<12} "
            f"{row['h_min']:>8.1f} {row['h_avg']:>8.1f} {row['h_max']:>8.1f} "
            f"{h_std:>8.1f}",
            f"  {'Давл., hPa':<12} "
            f"{row['p_min']:>8.1f} {row['p_avg']:>8.1f} {row['p_max']:>8.1f} "
            f"{p_std:>8.1f}",
            "",
            f"  GPS-фикс:  {row['gps_pct']:.1f} % измерений",
            "═" * 50,
        ]
        return "\n".join(lines)

    def forecast_accuracy(self) -> str:
        """MAE/RMSE per mode and horizon; best mode summary at the bottom."""
        with _conn(self._db_path) as c:
            rows = c.execute("""
                SELECT
                    fl.mode                          AS mode,
                    COUNT(*)                         AS n,
                    AVG(fv.error_temp_1h)            AS mae_t1,
                    AVG(fv.error_temp_2h)            AS mae_t2,
                    AVG(fv.error_temp_3h)            AS mae_t3,
                    AVG(fv.error_temp_1h * fv.error_temp_1h)   AS mse_t1,
                    AVG(fv.error_pressure_1h)        AS mae_p1,
                    AVG(fv.error_pressure_2h)        AS mae_p2,
                    AVG(fv.error_pressure_3h)        AS mae_p3,
                    AVG(fv.error_pressure_1h * fv.error_pressure_1h) AS mse_p1
                FROM forecast_verification fv
                JOIN forecast_log fl ON fv.forecast_id = fl.id
                WHERE fv.verified_1h = 1
                GROUP BY fl.mode
            """).fetchall()

        if not rows:
            return "forecast_verification: нет данных."

        lines = [
            "═" * 60,
            "  ТОЧНОСТЬ ПРОГНОЗА (по верифицированным записям)",
            "═" * 60,
        ]

        best_t1: dict[str, tuple] = {}   # mode → (mae, label)

        for r in rows:
            n = r["n"]
            if n < _MIN_VERIFY:
                lines.append(
                    f"\n  {r['mode']:>12}:  только {n} верификаций "
                    f"(нужно ≥ {_MIN_VERIFY}) — пропускаем"
                )
                continue

            rmse_t1 = _rmse(r["mse_t1"])
            rmse_p1 = _rmse(r["mse_p1"])

            def _fmt_t(v): return f"{v:>9.3f}°" if v is not None else f"{'—':>10}"
            def _fmt_p(v): return f"{v:>9.2f}" if v is not None else f"{'—':>9}"

            lines += [
                "",
                f"  ▸ Режим: {r['mode']}   (N={n})",
                f"    {'Горизонт':<10} {'MAE Темп':>10} {'MAE Давл':>10} {'RMSE Темп':>12}",
                "    " + "─" * 44,
                f"    {'+1ч':<10} "
                f"{_fmt_t(r['mae_t1'])}  {_fmt_p(r['mae_p1'])}  {rmse_t1:>11.3f}°",
                f"    {'+2ч':<10} "
                f"{_fmt_t(r['mae_t2'])}  {_fmt_p(r['mae_p2'])}",
                f"    {'+3ч':<10} "
                f"{_fmt_t(r['mae_t3'])}  {_fmt_p(r['mae_p3'])}",
            ]

            if r["mae_t1"] is not None:
                best_t1[r["mode"]] = (r["mae_t1"], r["mae_p1"])

        # Best mode summary
        if best_t1:
            best_mode_t = min(best_t1, key=lambda m: best_t1[m][0])
            best_mode_p = min(best_t1, key=lambda m: best_t1[m][1])
            lines += [
                "",
                "─" * 60,
                f"  Лучший MAE темп +1ч:  {best_mode_t}  "
                f"({best_t1[best_mode_t][0]:.3f}°C)",
                f"  Лучший MAE давл +1ч:  {best_mode_p}  "
                f"({best_t1[best_mode_p][1]:.2f} hPa)",
            ]

        lines.append("═" * 60)
        return "\n".join(lines)

    def correction_impact(self) -> str:
        """Show before/after MAE from the correction model meta file."""
        try:
            import config
            meta_path = config.CORRECTION_META_PATH
        except Exception:
            return ""

        if not pathlib.Path(meta_path).exists():
            return "Модель коррекции не обучена"

        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
        except Exception as exc:
            return f"Ошибка чтения мета-файла коррекции: {exc}"

        trained_at = meta.get("trained_at", "?")[:16]
        n          = meta.get("n_samples", "?")
        mae_before = meta.get("mae_before")
        mae_after  = meta.get("mae_after")

        lines = [
            "═" * 50,
            "  МОДЕЛЬ КОРРЕКЦИИ",
            "═" * 50,
            f"  Обучена:        {trained_at} UTC",
            f"  Обучающих строк: {n}",
        ]
        if mae_before is not None:
            lines.append(f"  MAE до:         {mae_before:.4f}°C (темп +1ч)")
        if mae_after is not None:
            lines.append(f"  MAE после:      {mae_after:.4f}°C (темп +1ч)")
        if mae_before is not None and mae_after is not None and mae_before > 0:
            improvement = (mae_before - mae_after) / mae_before * 100
            lines.append(f"  Улучшение:      {improvement:+.1f}%")
        lines.append("═" * 50)
        return "\n".join(lines)

    def timing_summary(self) -> str:
        """Mean/max per sensor component; % cycles that exceeded UPDATE_INTERVAL."""
        try:
            import config
            interval_ms = config.UPDATE_INTERVAL * 1000
        except Exception:
            interval_ms = 5000

        with _conn(self._db_path) as c:
            r = c.execute("""
                SELECT
                    COUNT(*)            AS n,
                    AVG(bme280_ms)      AS avg_bme,  MAX(bme280_ms)  AS max_bme,
                    AVG(gps_ms)         AS avg_gps,  MAX(gps_ms)     AS max_gps,
                    AVG(forecast_ms)    AS avg_fc,   MAX(forecast_ms) AS max_fc,
                    AVG(total_ms)       AS avg_tot,  MAX(total_ms)    AS max_tot,
                    SUM(CASE WHEN total_ms > ? THEN 1 ELSE 0 END) * 100.0
                        / NULLIF(COUNT(*), 0)        AS pct_over
                FROM cycle_timing
            """, (interval_ms,)).fetchone()

        if not r or not r["n"]:
            return "cycle_timing: нет данных."

        lines = [
            "═" * 50,
            "  ВРЕМЯ ЦИКЛА",
            "═" * 50,
            f"  Циклов: {r['n']}",
            "",
            f"  {'Компонент':<16} {'Среднее':>10} {'Макс':>10}",
            "  " + "─" * 38,
            f"  {'BME280':<16} {r['avg_bme']:>9.1f}  {r['max_bme']:>9.1f}  мс",
            f"  {'GPS':<16} {r['avg_gps']:>9.1f}  {r['max_gps']:>9.1f}  мс",
            f"  {'Прогноз':<16} {r['avg_fc']:>9.1f}  {r['max_fc']:>9.1f}  мс",
            f"  {'Итого':<16} {r['avg_tot']:>9.1f}  {r['max_tot']:>9.1f}  мс",
            "",
            f"  Циклов > {interval_ms/1000:.0f}с: {r['pct_over']:.1f} %",
            "═" * 50,
        ]
        return "\n".join(lines)

    def lstm_progress(self) -> str:
        """Table of LSTM retraining events: date, readings, MAE temp/pressure."""
        with _conn(self._db_path) as c:
            rows = c.execute("""
                SELECT timestamp, readings_count, mae_temp, mae_pressure,
                       rmse_temp, rmse_pressure, training_duration_sec
                FROM lstm_training_log
                ORDER BY timestamp
            """).fetchall()

        if not rows:
            return "lstm_training_log: нет данных."

        lines = [
            "═" * 70,
            "  ПРОГРЕСС ОБУЧЕНИЯ LSTM",
            "═" * 70,
            f"  {'Дата':<20} {'Данных':>8} {'MAE T':>8} {'MAE P':>8} "
            f"{'RMSE T':>8} {'RMSE P':>8} {'Сек':>6}",
            "  " + "─" * 66,
        ]
        for r in rows:
            ts  = r["timestamp"][:16]
            n   = r["readings_count"] or 0
            mt  = f"{r['mae_temp']:.3f}"  if r["mae_temp"]  is not None else "   N/A"
            mp  = f"{r['mae_pressure']:.3f}" if r["mae_pressure"] is not None else "   N/A"
            rt  = f"{r['rmse_temp']:.3f}"  if r["rmse_temp"]  is not None else "   N/A"
            rp  = f"{r['rmse_pressure']:.3f}" if r["rmse_pressure"] is not None else "   N/A"
            sec = f"{r['training_duration_sec']:.0f}" if r["training_duration_sec"] is not None else " N/A"
            lines.append(
                f"  {ts:<20} {n:>8} {mt:>8} {mp:>8} {rt:>8} {rp:>8} {sec:>6}"
            )
        lines.append("═" * 70)
        return "\n".join(lines)

    def rp5_comparison_instructions(self) -> str:
        """Instructions for manual rp5.ru reference comparison."""
        with _conn(self._db_path) as c:
            rows = c.execute("""
                SELECT timestamp, temperature, humidity, pressure, pressure_sl,
                       gps_lat, gps_lon, gps_altitude, gps_fix
                FROM sensor_log
                ORDER BY timestamp DESC
                LIMIT 5
            """).fetchall()

        lines = [
            "═" * 60,
            "  СРАВНЕНИЕ С ЭТАЛОННЫМИ ДАННЫМИ (rp5.ru)",
            "═" * 60,
            "",
            "  Для сравнения с эталоном откройте rp5.ru и запишите",
            "  показания для вашего города в следующие моменты времени:",
            "",
            f"  {'Время (UTC)':<22} {'T, °C':>7} {'Вл, %':>7} {'Дав, hPa':>10}",
            "  " + "─" * 50,
        ]

        for r in rows:
            ts = r["timestamp"][:16]
            t  = f"{r['temperature']:.1f}" if r["temperature"] is not None else " N/A"
            h  = f"{r['humidity']:.1f}"    if r["humidity"]    is not None else " N/A"
            p  = f"{r['pressure']:.1f}"    if r["pressure"]    is not None else "    N/A"
            lines.append(f"  {ts:<22} {t:>7} {h:>7} {p:>10}")

        lines += [
            "",
            "  Внесите данные rp5.ru вручную в таблицу Excel",
            "  рядом с соответствующими строками sensor_log.csv",
            "═" * 60,
        ]
        return "\n".join(lines)

    def save_csv(self) -> None:
        """Export research tables to semicolon-separated CSV files."""
        self._export_sensor_log()
        self._export_forecast_accuracy()
        self._export_lstm_progress()
        print(f"CSV файлы сохранены в {self._csv_dir}/")

    # ── CSV helpers ───────────────────────────────────────────────────────────

    def _write_csv(self, filename: str, headers: list[str], rows) -> None:
        path = self._csv_dir / filename
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=_SEP)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"  Записан: {path.name}  ({len(list(rows)) if hasattr(rows, '__len__') else '?'} строк)")

    def _export_sensor_log(self) -> None:
        with _conn(self._db_path) as c:
            rows = c.execute("""
                SELECT timestamp, temperature, humidity, pressure, pressure_sl,
                       gps_lat, gps_lon, gps_altitude, gps_fix
                FROM sensor_log ORDER BY timestamp
            """).fetchall()
        path = self._csv_dir / "sensor_log.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=_SEP)
            w.writerow(["timestamp", "temperature", "humidity", "pressure",
                         "pressure_sl", "gps_lat", "gps_lon", "gps_altitude", "gps_fix"])
            for r in rows:
                w.writerow(list(r))
        print(f"  Записан: sensor_log.csv  ({len(rows)} строк)")

    def _export_forecast_accuracy(self) -> None:
        with _conn(self._db_path) as c:
            rows = c.execute("""
                SELECT
                    fl.mode, fl.timestamp AS fc_time,
                    fv.verified_at,
                    fv.actual_temp, fv.actual_pressure,
                    fl.temp_1h, fl.temp_2h, fl.temp_3h,
                    fv.error_temp_1h, fv.error_temp_2h, fv.error_temp_3h,
                    fl.pressure_1h, fl.pressure_2h, fl.pressure_3h,
                    fv.error_pressure_1h, fv.error_pressure_2h, fv.error_pressure_3h,
                    fl.confidence
                FROM forecast_verification fv
                JOIN forecast_log fl ON fv.forecast_id = fl.id
                WHERE fv.verified_1h = 1
                ORDER BY fl.timestamp
            """).fetchall()
        path = self._csv_dir / "forecast_accuracy.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=_SEP)
            w.writerow([
                "mode", "forecast_time", "verified_at",
                "actual_temp", "actual_pressure",
                "pred_temp_1h", "pred_temp_2h", "pred_temp_3h",
                "error_temp_1h", "error_temp_2h", "error_temp_3h",
                "pred_pres_1h", "pred_pres_2h", "pred_pres_3h",
                "error_pres_1h", "error_pres_2h", "error_pres_3h",
                "confidence",
            ])
            for r in rows:
                w.writerow(list(r))
        print(f"  Записан: forecast_accuracy.csv  ({len(rows)} строк)")

    def _export_lstm_progress(self) -> None:
        with _conn(self._db_path) as c:
            rows = c.execute("""
                SELECT timestamp, readings_count, mae_temp, mae_pressure,
                       rmse_temp, rmse_pressure, training_duration_sec
                FROM lstm_training_log ORDER BY timestamp
            """).fetchall()
        path = self._csv_dir / "lstm_progress.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=_SEP)
            w.writerow(["timestamp", "readings_count", "mae_temp", "mae_pressure",
                         "rmse_temp", "rmse_pressure", "training_duration_sec"])
            for r in rows:
                w.writerow(list(r))
        print(f"  Записан: lstm_progress.csv  ({len(rows)} строк)")
