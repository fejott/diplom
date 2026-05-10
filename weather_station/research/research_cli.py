"""
Research CLI — quick access to diploma thesis data reports.

Usage (from weather_station/):
    python research/research_cli.py status
    python research/research_cli.py report
    python research/research_cli.py export
    python research/research_cli.py compare-rp5
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

# Allow running from weather_station/ root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research.report_generator import ReportGenerator

_DB_PATH = pathlib.Path(__file__).resolve().parent / "research_data.db"


def _quick_status() -> str:
    """One-screen summary of collected data."""
    if not _DB_PATH.exists():
        return "research_data.db не найдена — запустите main.py сначала."

    with sqlite3.connect(str(_DB_PATH)) as c:
        c.row_factory = sqlite3.Row

        n_sensor   = c.execute("SELECT COUNT(*) FROM sensor_log").fetchone()[0]
        n_fc       = c.execute("SELECT COUNT(*) FROM forecast_log").fetchone()[0]
        n_verify   = c.execute(
            "SELECT COUNT(*) FROM forecast_verification WHERE verified_1h=1"
        ).fetchone()[0]
        n_lstm     = c.execute("SELECT COUNT(*) FROM lstm_training_log").fetchone()[0]

        # Period
        first_ts = c.execute("SELECT MIN(timestamp) FROM sensor_log").fetchone()[0]
        last_ts  = c.execute("SELECT MAX(timestamp) FROM sensor_log").fetchone()[0]

        # Forecasts by mode
        modes = c.execute(
            "SELECT mode, COUNT(*) AS n FROM forecast_log GROUP BY mode"
        ).fetchall()
        mode_str = "  ".join(f"{r['mode']}: {r['n']}" for r in modes)

        # Best MAE per mode (1h temp)
        accuracy = c.execute("""
            SELECT fl.mode, AVG(fv.error_temp_1h) AS mae
            FROM forecast_verification fv
            JOIN forecast_log fl ON fv.forecast_id = fl.id
            WHERE fv.verified_1h = 1
            GROUP BY fl.mode
            ORDER BY mae
        """).fetchall()

    lines = [
        "═" * 50,
        "  СТАТУС ИССЛЕДОВАНИЯ",
        "═" * 50,
        f"  Период сбора:   "
        f"{(first_ts or '?')[:16]} → {(last_ts or '?')[:16]}",
        f"  Измерений:      {n_sensor}",
        f"  Прогнозов:      {n_fc}  ({mode_str})",
        f"  Верифицировано: {n_verify}",
        f"  LSTM обучений:  {n_lstm}",
    ]

    if accuracy:
        lines.append("")
        lines.append("  Лучший MAE темп +1ч:")
        for r in accuracy:
            lines.append(f"    {r['mode']:>12}:  {r['mae']:.3f}°C")

    lines.append("═" * 50)
    return "\n".join(lines)


def cmd_status(_args: argparse.Namespace) -> None:
    print(_quick_status())


def cmd_report(_args: argparse.Namespace) -> None:
    rg = ReportGenerator()
    print(rg.sensor_summary())
    print()
    print(rg.forecast_accuracy())
    print()
    print(rg.timing_summary())
    print()
    print(rg.lstm_progress())


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

    sub.add_parser("status",      help="Quick one-screen summary")
    sub.add_parser("report",      help="Full text report to terminal")
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
