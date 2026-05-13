"""
Calibration CLI — manage the residual correction model.

Usage (from weather_station/):
    python research/calibration_cli.py correction-status
    python research/calibration_cli.py train-correction
    python research/calibration_cli.py rollback-correction
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Allow running from weather_station/ root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from forecasting.correction_model import CorrectionModel

_RESEARCH_DB = pathlib.Path(__file__).resolve().parent / "research_data.db"


def _progress_bar(n: int, need: int) -> str:
    pct = min(100, n * 100 // need)
    filled = pct // 5
    return "█" * filled + "░" * (20 - filled)


def cmd_correction_status(_args: argparse.Namespace) -> None:
    """Show correction model status: verified count, progress, trained/not."""
    cm = CorrectionModel()
    ok, msg = cm.can_train(str(_RESEARCH_DB))

    # Count verified rows
    try:
        import sqlite3
        with sqlite3.connect(str(_RESEARCH_DB)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM forecast_verification fv "
                "JOIN forecast_log fl ON fv.forecast_id = fl.id "
                "WHERE fv.signed_error_temp_1h IS NOT NULL "
                "  AND fl.mode IN ('lstm', 'lstm_corrected')"
            ).fetchone()
        n = row[0] if row else 0
    except Exception:
        n = 0

    need = config.CORRECTION_MIN_VERIFIED
    bar  = _progress_bar(n, need)
    pct  = min(100, n * 100 // need)

    trained = cm.is_ready()
    trained_str = "Да" if trained else "Нет"

    # Read meta if exists
    meta_str = ""
    if os.path.exists(config.CORRECTION_META_PATH):
        try:
            with open(config.CORRECTION_META_PATH, "r") as fh:
                meta = json.load(fh)
            mae_b = meta.get("mae_before")
            mae_a = meta.get("mae_after")
            ts    = meta.get("trained_at", "?")[:16]
            meta_str = (
                f"\n  Обучено:        {ts} UTC"
                f"\n  MAE до:         {mae_b:.4f}°C" if mae_b is not None else ""
            )
            if mae_a is not None:
                meta_str += f"\n  MAE после:      {mae_a:.4f}°C"
        except Exception:
            pass

    lines = [
        "═" * 50,
        "  СТАТУС МОДЕЛИ КОРРЕКЦИИ",
        "═" * 50,
        f"  Верифицировано: {n} / {need}",
        f"  [{bar}] {pct}%",
        f"  Модель обучена: {trained_str}",
    ]
    if meta_str:
        lines.append(meta_str)
    lines.append("═" * 50)
    print("\n".join(lines))


def cmd_train_correction(_args: argparse.Namespace) -> None:
    """Train the correction model (asks for confirmation first)."""
    cm = CorrectionModel()
    ok, msg = cm.can_train(str(_RESEARCH_DB))

    print(f"  Проверка данных: {msg}")

    if not ok:
        print("  Обучение невозможно.")
        return

    answer = input("  Начать обучение модели коррекции? [y/N] ").strip().lower()
    if answer not in ("y", "yes", "д", "да"):
        print("  Отменено.")
        return

    print("  Обучение... (может занять несколько секунд)")
    result = cm.train(str(_RESEARCH_DB))

    print()
    if result.success:
        print("  Обучение завершено успешно!")
        print(f"  Образцов:      {result.n_samples}")
        if result.mae_before is not None:
            print(f"  MAE до:        {result.mae_before:.4f}°C")
        if result.mae_after is not None:
            print(f"  MAE после:     {result.mae_after:.4f}°C")
    else:
        print(f"  Ошибка: {result.message}")


def cmd_rollback_correction(_args: argparse.Namespace) -> None:
    """Delete all correction model files."""
    files = [
        config.CORRECTION_WEIGHTS_PATH,
        config.CORRECTION_WEIGHTS_PATH + ".npz"
        if not config.CORRECTION_WEIGHTS_PATH.endswith(".npz") else None,
        config.CORRECTION_SCALER_PATH,
        config.CORRECTION_META_PATH,
    ]

    deleted = []
    for f in files:
        if f is None:
            continue
        if os.path.exists(f):
            try:
                os.remove(f)
                deleted.append(f)
            except Exception as exc:
                print(f"  Не удалось удалить {f}: {exc}")

    if deleted:
        print("  Удалены файлы модели коррекции:")
        for f in deleted:
            print(f"    {f}")
    else:
        print("  Файлы модели коррекции не найдены — нечего удалять.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibration_cli",
        description="Correction model management tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("correction-status",  help="Show correction model status")
    sub.add_parser("train-correction",   help="Train the correction model")
    sub.add_parser("rollback-correction", help="Delete correction model files")

    args = parser.parse_args()
    dispatch = {
        "correction-status":   cmd_correction_status,
        "train-correction":    cmd_train_correction,
        "rollback-correction": cmd_rollback_correction,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
