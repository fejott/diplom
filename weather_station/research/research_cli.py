"""
Research CLI — quick access to diploma thesis data reports.

Usage (from weather_station/):
    python research/research_cli.py status
    python research/research_cli.py status --days 7
    python research/research_cli.py report
    python research/research_cli.py report --days 7
    python research/research_cli.py export
    python research/research_cli.py compare-rp5
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Optional

# Allow running from weather_station/ root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research.report_generator import ReportGenerator

_DB_PATH = pathlib.Path(__file__).resolve().parent / "research_data.db"


def _quick_status(days: Optional[int] = None) -> str:
    """One-screen summary of collected data."""
    if not _DB_PATH.exists():
        return (
            "═" * 50 + "\n"
            "  СТАТУС ИССЛЕДОВАНИЯ\n"
            "═" * 50 + "\n"
            "  Данных пока нет.\n"
            "  research_data.db создаётся автоматически при\n"
            "  первом запуске main.py.\n"
            "═" * 50
        )

    since_ts: Optional[str] = None
    if days is not None:
        since_ts = (datetime.utcnow() - timedelta(days=days)).isoformat()

    def _where(col: str = "fl.timestamp") -> str:
        return f"AND {col} >= ?" if since_ts else ""

    def _params(*extra) -> tuple:
        return ((since_ts, *extra) if since_ts else tuple(extra))

    with sqlite3.connect(str(_DB_PATH)) as c:
        c.row_factory = sqlite3.Row

        n_sensor   = c.execute("SELECT COUNT(*) FROM sensor_log").fetchone()[0]
        n_fc       = c.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0]
        n_verify   = c.execute(
            "SELECT COUNT(*) FROM forecast_verification WHERE verified_1h=1"
        ).fetchone()[0]
        n_corrections = c.execute("SELECT COUNT(*) FROM lstm_training_log").fetchone()[0]

        # Period
        first_ts = c.execute("SELECT MIN(timestamp) FROM sensor_log").fetchone()[0]
        last_ts  = c.execute("SELECT MAX(timestamp) FROM sensor_log").fetchone()[0]

        # Forecasts by mode
        modes = c.execute(
            "SELECT mode, COUNT(*) AS n FROM forecast_log GROUP BY mode"
        ).fetchall()
        mode_str = "  ".join(f"{r['mode']}: {r['n']}" for r in modes)

        # Best MAE per mode (1h temp) — filtered by days if given
        accuracy = c.execute(f"""
            SELECT fl.mode, AVG(fv.error_temp_1h) AS mae
            FROM forecast_verification fv
            JOIN forecast_log fl ON fv.forecast_id = fl.id
            WHERE fv.verified_1h = 1
            {_where()}
            GROUP BY fl.mode
            ORDER BY mae
        """, _params()).fetchall()

    period_label = (
        f"последние {days} дн. (с {since_ts[:10]})"
        if since_ts else
        f"{(first_ts or '?')[:16]} → {(last_ts or '?')[:16]}"
    )

    lines = [
        "═" * 50,
        "  СТАТУС ИССЛЕДОВАНИЯ",
        "═" * 50,
        f"  Период:         {period_label}",
        f"  Измерений:      {n_sensor}",
        f"  Прогнозов:      {n_fc}  ({mode_str})",
        f"  Верифицировано: {n_verify}",
        f"  LSTM модель:    заморожена (ERA5, обучена 1 раз)",
        f"  Коррекций:      {n_corrections}",
    ]

    if accuracy:
        label = f"последние {days} дн." if days else "всё время"
        lines.append("")
        lines.append(f"  MAE темп +1ч ({label}):")
        for r in accuracy:
            lines.append(f"    {r['mode']:>16}:  {r['mae']:.3f}°C")

    lines.append("═" * 50)
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> None:
    print(_quick_status(days=getattr(args, "days", None)))


def cmd_report(args: argparse.Namespace) -> None:
    days = getattr(args, "days", None)
    rg = ReportGenerator(days=days)
    print(rg.sensor_summary())
    print()
    print(rg.forecast_accuracy())
    print()
    print(rg.timing_summary())
    print()
    print(rg.lstm_progress())
    impact = rg.correction_impact()
    if impact:
        print()
        print(impact)


def cmd_export(_args: argparse.Namespace) -> None:
    ReportGenerator().save_csv()


def cmd_compare_rp5(_args: argparse.Namespace) -> None:
    print(ReportGenerator().rp5_comparison_instructions())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="research_cli",
        description="Diploma research data tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Quick one-screen summary")
    p_status.add_argument(
        "--days", type=int, default=None, metavar="N",
        help="Show MAE only for the last N days (default: all time)",
    )

    p_report = sub.add_parser("report", help="Full text report to terminal")
    p_report.add_argument(
        "--days", type=int, default=None, metavar="N",
        help="Filter forecast accuracy to last N days (default: all time)",
    )

    sub.add_parser("export",      help="Export CSV files for Excel")
    sub.add_parser("compare-rp5", help="Instructions for manual rp5.ru comparison")

    args = parser.parse_args()
    dispatch = {
        "status":      cmd_status,
        "report":      cmd_report,
        "export":      cmd_export,
        "compare-rp5": cmd_compare_rp5,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
