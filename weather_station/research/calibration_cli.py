"""
Calibration CLI — manage the residual correction model.

Usage (from weather_station/):
    python research/calibration_cli.py correction-status
    python research/calibration_cli.py train-correction
    python research/calibration_cli.py rollback-correction
    python research/calibration_cli.py validate
"""

from __future__ import annotations

import argparse
import json
import math
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
    """Show correction model status: verified count (current LSTM only), progress, trained/not."""
    cm = CorrectionModel()
    ok, msg = cm.can_train(str(_RESEARCH_DB))
    since_ts = cm._get_lstm_trained_at()

    # Count verified rows — filtered to current LSTM deployment
    try:
        import sqlite3
        with sqlite3.connect(str(_RESEARCH_DB)) as conn:
            if since_ts:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM forecast_verification fv "
                    "JOIN forecast_log fl ON fv.forecast_id = fl.id "
                    "WHERE fv.signed_error_temp_1h IS NOT NULL "
                    "  AND fl.mode IN ('lstm', 'lstm_corrected') "
                    "  AND fl.timestamp >= ?",
                    (since_ts,),
                ).fetchone()
            else:
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

    since_line = f"  С момента деплоя LSTM: {since_ts[:16]}" if since_ts else ""
    lines = [
        "═" * 50,
        "  СТАТУС МОДЕЛИ КОРРЕКЦИИ",
        "═" * 50,
    ]
    if since_line:
        lines.append(since_line)
    lines += [
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


def cmd_validate(_args: argparse.Namespace) -> None:
    """Held-out validation: Base LSTM vs LSTM+Correction vs Online API."""
    import sqlite3
    from datetime import datetime as _dt2

    import numpy as np

    from forecasting.correction_model import CorrectionModel

    # ── 1. Load readings from weather_history.db ──────────────────────────────
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            all_rows = conn.execute(
                "SELECT timestamp, temperature, humidity, pressure "
                "FROM readings ORDER BY id"
            ).fetchall()
    except Exception as exc:
        print(f"  Ошибка чтения {config.DB_PATH}: {exc}")
        return

    n_total = len(all_rows)
    if n_total < config.FORECAST_MIN_READINGS:
        print(f"  Недостаточно данных: {n_total} < {config.FORECAST_MIN_READINGS}")
        return

    from datetime import timedelta

    # Time-based 70/30 split (mirrors lstm_forecast.py train() logic)
    t_start  = _dt2.fromisoformat(all_rows[0][0])
    t_end    = _dt2.fromisoformat(all_rows[-1][0])
    span_sec = (t_end - t_start).total_seconds()
    cutoff_dt = t_start + timedelta(seconds=span_sec * (1.0 - config.VALIDATION_SPLIT))
    cutoff_ts = cutoff_dt.isoformat()

    n_train_raw = sum(1 for r in all_rows if r[0] < cutoff_ts)
    n_held_raw  = n_total - n_train_raw

    seq     = config.SEQUENCE_LENGTH   # 5-min steps (12 = 1 h of context)
    steps   = config.FORECAST_STEPS   # [12, 24, 36] → +1h/+2h/+3h in 5-min units
    n_steps = len(steps)

    print("═" * 65)
    print("  ВАЛИДАЦИЯ НА ОТЛОЖЕННОЙ ВЫБОРКЕ")
    print("═" * 65)
    print(f"  Всего записей:   {n_total}")
    print(f"  Обучение (70%):  {n_train_raw} зап. (до {cutoff_ts[:16]})")
    print(f"  Отложено (30%):  {n_held_raw} зап. (с  {cutoff_ts[:16]})")

    # ── 2. Resample held-out rows to 5-minute buckets ─────────────────────────
    # The LSTM scaler and weights were trained on 5-min resampled data,
    # so validation sequences must also use 5-min resolution.
    from collections import defaultdict

    held_raw = [r for r in all_rows if r[0] >= cutoff_ts]

    def _resample_rows_to_5min(rows):
        """Average raw DB rows (ts, temp, hum, pres) into 5-minute buckets."""
        groups: dict = defaultdict(list)
        for r in rows:
            ts = _dt2.fromisoformat(r[0])
            minute_floor = (ts.minute // 5) * 5
            bucket_key   = ts.replace(minute=minute_floor, second=0, microsecond=0)
            groups[bucket_key].append(r)
        out = []
        for bk in sorted(groups.keys()):
            grp = groups[bk]
            n   = len(grp)
            out.append((
                bk.isoformat(),
                sum(r[1] for r in grp) / n,
                sum(r[2] for r in grp) / n,
                sum(r[3] for r in grp) / n,
            ))
        return out

    held_5min   = _resample_rows_to_5min(held_raw)
    n_held_5min = len(held_5min)

    print(f"  Отложено (5-мин): {n_held_5min} бакетов")

    min_needed = seq + max(steps) + 10
    if n_held_5min < min_needed:
        print(f"\n  Отложенная выборка слишком мала: {n_held_5min} < {min_needed}")
        return

    data_raw = np.array(
        [[r[1], r[2], r[3]] for r in held_5min], dtype=np.float32
    )  # (n_held_5min, 3): [temp, hum, pres]

    # ── 3. Load scaler ────────────────────────────────────────────────────────
    if not os.path.exists(config.SCALER_PATH):
        print("  Scaler не найден — LSTM ещё не обучен.")
        return
    with open(config.SCALER_PATH) as fh:
        sc = json.load(fh)
    s_min = np.array(sc["min"], dtype=np.float32)
    s_max = np.array(sc["max"], dtype=np.float32)
    rng   = s_max - s_min
    rng[rng == 0] = 1.0
    norm  = (data_raw - s_min) / rng

    # ── 4. Load LSTM model ────────────────────────────────────────────────────
    weights_file = config.WEIGHTS_PATH
    if not weights_file.endswith(".npz"):
        weights_file += ".npz"
    if not os.path.exists(weights_file):
        print("  Веса LSTM не найдены — модель ещё не обучена.")
        return

    print("\n  Загрузка LSTM модели...")
    try:
        import tensorflow as tf

        n_out = 3 * n_steps
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(seq, 3)),
            tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(config.LSTM_UNITS),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_out),
        ])
        w_data = np.load(weights_file, allow_pickle=False)
        model.set_weights([w_data[f"arr_{i}"] for i in range(len(w_data.files))])
    except Exception as exc:
        print(f"  Ошибка загрузки модели: {exc}")
        return

    # ── 5. Build sequences from 5-min held-out data ───────────────────────────
    n_seqs = n_held_5min - seq - max(steps)
    X_list, y_list, ts_list = [], [], []
    for i in range(n_seqs):
        X_list.append(norm[i: i + seq])
        targets = []
        for st in steps:
            targets.extend(data_raw[i + seq + st - 1])
        y_list.append(targets)
        ts_list.append(held_5min[i + seq][0])  # forecast timestamp

    X      = np.array(X_list, dtype=np.float32)   # (n_seqs, seq, 3)
    y_true = np.array(y_list, dtype=np.float32)   # (n_seqs, n_steps*3)

    print(f"  Последовательностей для оценки: {n_seqs}")
    print("  Вычисление предсказаний LSTM (может занять ~1 мин)...")

    y_pred_norm = model.predict(X, batch_size=64, verbose=0)

    # Denormalise predictions
    rng_tiled = np.tile(rng, n_steps)
    min_tiled = np.tile(s_min, n_steps)
    y_pred    = y_pred_norm * rng_tiled + min_tiled  # (n_seqs, n_steps*3)

    # MAE — Base LSTM
    mae_base_temp = [
        float(np.mean(np.abs(y_pred[:, i * 3]     - y_true[:, i * 3])))
        for i in range(n_steps)
    ]
    mae_base_pres = [
        float(np.mean(np.abs(y_pred[:, i * 3 + 2] - y_true[:, i * 3 + 2])))
        for i in range(n_steps)
    ]

    # ── 6. Apply correction model (vectorised batch) ──────────────────────────
    cm = CorrectionModel()
    correction_available = cm.is_ready() and n_steps == 3

    if correction_available:
        print("  Применение модели коррекции...")
        from datetime import datetime as _dt

        # Parse timestamps and compute cyclic time features (vectorised)
        dt_list  = []
        for ts_str in ts_list:
            try:
                dt_list.append(_dt.fromisoformat(ts_str))
            except Exception:
                dt_list.append(_dt.utcnow())
        hours    = np.array([dt.hour      for dt in dt_list], dtype=np.float32)
        weekdays = np.array([dt.weekday() for dt in dt_list], dtype=np.float32)
        sin_h    = np.sin(2 * np.pi * hours    / 24).astype(np.float32)
        cos_h    = np.cos(2 * np.pi * hours    / 24).astype(np.float32)
        sin_dow  = np.sin(2 * np.pi * weekdays /  7).astype(np.float32)
        cos_dow  = np.cos(2 * np.pi * weekdays /  7).astype(np.float32)

        # Current pressure at the time of each forecast (last reading of context)
        cur_pres   = data_raw[seq - 1: seq - 1 + n_seqs, 2]        # (n_seqs,)
        pres_trend = y_pred[:, (n_steps - 1) * 3 + 2] - cur_pres   # (n_seqs,)

        X_corr = np.column_stack([
            y_pred[:, 0],   # temp_1h
            y_pred[:, 3],   # temp_2h
            y_pred[:, 6],   # temp_3h
            y_pred[:, 2],   # pres_1h
            y_pred[:, 5],   # pres_2h
            y_pred[:, 8],   # pres_3h
            pres_trend,
            sin_h, cos_h, sin_dow, cos_dow,
        ]).astype(np.float32)  # (n_seqs, 11)

        deltas = cm.predict_correction_batch(X_corr)  # (n_seqs, 6)

        y_corr = y_pred.copy()
        for i in range(n_steps):
            y_corr[:, i * 3]     += deltas[:, i]           # Δtemp
            y_corr[:, i * 3 + 2] += deltas[:, n_steps + i] # Δpres

        mae_corr_temp = [
            float(np.mean(np.abs(y_corr[:, i * 3]     - y_true[:, i * 3])))
            for i in range(n_steps)
        ]
        mae_corr_pres = [
            float(np.mean(np.abs(y_corr[:, i * 3 + 2] - y_true[:, i * 3 + 2])))
            for i in range(n_steps)
        ]
    else:
        mae_corr_temp = [None] * n_steps
        mae_corr_pres = [None] * n_steps
        if not cm.is_ready():
            print("  Модель коррекции не обучена — колонка недоступна.")

    # ── 7. Online API MAE from research_data.db ───────────────────────────────
    mae_api_temp = [None] * n_steps
    mae_api_pres = [None] * n_steps
    horizon_labels = ["1h", "2h", "3h"]
    try:
        with sqlite3.connect(str(_RESEARCH_DB)) as conn:
            conn.row_factory = sqlite3.Row
            for i, label in enumerate(horizon_labels[:n_steps]):
                col_t = f"signed_error_temp_{label}"
                col_p = f"signed_error_pres_{label}"
                r = conn.execute(
                    f"SELECT AVG(ABS(fv.{col_t})) AS mt, "
                    f"       AVG(ABS(fv.{col_p})) AS mp "
                    "FROM forecast_verification fv "
                    "JOIN forecast_log fl ON fv.forecast_id = fl.id "
                    "WHERE fl.mode = 'online_api' "
                    "  AND fl.timestamp >= ? "
                    f"  AND fv.{col_t} IS NOT NULL",
                    (cutoff_ts,),
                ).fetchone()
                if r and r["mt"] is not None:
                    mae_api_temp[i] = float(r["mt"])
                    if r["mp"] is not None:
                        mae_api_pres[i] = float(r["mp"])
    except Exception as exc:
        print(f"  Предупреждение: данные API недоступны: {exc}")

    # ── 8. Print results ──────────────────────────────────────────────────────
    def _ft(v):
        return f"{v:.4f}°C" if v is not None else "—"

    def _fp(v):
        return f"{v:.4f} hPa" if v is not None else "—"

    print()
    print("═" * 65)
    print(f"  Период отложенной выборки: {cutoff_ts[:16]}  →  {held_5min[-1][0][:16]}")
    print(f"  Образцов (5-мин): {n_seqs}")
    print("─" * 65)
    print(f"  {'Горизонт':<10} {'Базовый LSTM':>16} {'LSTM+Корр.':>16} {'Online API':>16}")
    print(f"  {'MAE (°C)':<10} {'─'*16} {'─'*16} {'─'*16}")
    for i, h in enumerate(horizon_labels[:n_steps]):
        print(f"  +{h:<9} {_ft(mae_base_temp[i]):>16} "
              f"{_ft(mae_corr_temp[i]):>16} {_ft(mae_api_temp[i]):>16}")
    print()
    print(f"  {'Горизонт':<10} {'Базовый LSTM':>16} {'LSTM+Корр.':>16} {'Online API':>16}")
    print(f"  {'MAE (hPa)':<10} {'─'*16} {'─'*16} {'─'*16}")
    for i, h in enumerate(horizon_labels[:n_steps]):
        print(f"  +{h:<9} {_fp(mae_base_pres[i]):>16} "
              f"{_fp(mae_corr_pres[i]):>16} {_fp(mae_api_pres[i]):>16}")
    print("═" * 65)


def cmd_backfill_signed_errors(_args: argparse.Namespace) -> None:
    """Backfill signed_error_* columns for historical LSTM verifications.

    Old forecast_verification rows were written before the signed_error_*
    columns existed in the schema.  They have verified_Nh=1 and absolute
    error_temp_Nh stored, but signed_error_temp_Nh=NULL — so the correction
    model can't use them.

    This command reconstructs the signed errors by looking up the actual
    temperature and pressure from sensor_log (same research_data.db, same
    UTC clock) at forecast_timestamp + H hours for each horizon H in {1,2,3}.
    A reading within ±10 minutes is accepted as the verification actual.
    """
    import bisect
    import sqlite3
    from datetime import datetime as _dt, timedelta

    print("═" * 60)
    print("  БЭКФИЛЛ SIGNED ERRORS ДЛЯ КОРРЕКЦИОННОЙ МОДЕЛИ")
    print("═" * 60)

    with sqlite3.connect(str(_RESEARCH_DB)) as conn:
        conn.row_factory = sqlite3.Row

        # ── 1. Load sensor_log as reference actual values ──────────────────
        sl_rows = conn.execute(
            "SELECT timestamp, temperature, pressure "
            "FROM sensor_log ORDER BY timestamp"
        ).fetchall()

    if not sl_rows:
        print("  sensor_log пуст — нечего использовать для бэкфилла.")
        return

    sl_ts   = [r["timestamp"]   for r in sl_rows]  # sorted ISO UTC strings
    sl_temp = [r["temperature"] for r in sl_rows]
    sl_pres = [r["pressure"]    for r in sl_rows]

    _MAX_GAP_SEC = 600  # accept sensor_log reading within ±10 min

    def _lookup_actual(forecast_ts: str, hours: int):
        """Find nearest sensor_log (temp, pressure) at forecast_ts + hours."""
        target = (_dt.fromisoformat(forecast_ts)
                  + timedelta(hours=hours)).isoformat()
        idx = bisect.bisect_left(sl_ts, target)
        best_i, best_gap = None, float("inf")
        for i in (idx - 1, idx):
            if 0 <= i < len(sl_ts):
                gap = abs((_dt.fromisoformat(sl_ts[i])
                           - _dt.fromisoformat(target)).total_seconds())
                if gap < best_gap:
                    best_gap, best_i = gap, i
        if best_i is None or best_gap > _MAX_GAP_SEC:
            return None, None
        return sl_temp[best_i], sl_pres[best_i]

    with sqlite3.connect(str(_RESEARCH_DB)) as conn:
        conn.row_factory = sqlite3.Row

        # ── 2. Load rows that need backfilling ─────────────────────────────
        need_rows = conn.execute(
            "SELECT fv.id AS fv_id, fl.timestamp, "
            "  fl.temp_1h, fl.temp_2h, fl.temp_3h, "
            "  fl.pressure_1h, fl.pressure_2h, fl.pressure_3h, "
            "  fv.verified_1h, fv.verified_2h, fv.verified_3h "
            "FROM forecast_verification fv "
            "JOIN forecast_log fl ON fv.forecast_id = fl.id "
            "WHERE fv.signed_error_temp_1h IS NULL "
            "  AND fv.verified_1h = 1 "
            "  AND fl.mode IN ('lstm', 'lstm_corrected')"
        ).fetchall()

    print(f"  Записей для бэкфилла: {len(need_rows)}")
    if not need_rows:
        print("  Нечего бэкфиллить.")
        return

    # ── 3. Compute signed errors ───────────────────────────────────────────
    updates = []
    n_partial = 0
    for row in need_rows:
        fts = row["timestamp"]
        signed: dict = {}
        for h in (1, 2, 3):
            if not row[f"verified_{h}h"]:
                continue  # horizon not yet verified — skip
            a_temp, a_pres = _lookup_actual(fts, h)
            if a_temp is None:
                continue
            pred_t = row[f"temp_{h}h"]
            pred_p = row[f"pressure_{h}h"]
            if pred_t is not None:
                signed[f"st{h}"] = round(a_temp - pred_t, 4)
            if pred_p is not None and a_pres is not None:
                signed[f"sp{h}"] = round(a_pres - pred_p, 4)

        if "st1" not in signed:
            n_partial += 1
            continue  # need at minimum the +1h signed error

        updates.append((
            signed.get("st1"),
            signed.get("st2"),
            signed.get("st3"),
            signed.get("sp1"),
            signed.get("sp2"),
            signed.get("sp3"),
            row["fv_id"],
        ))

    print(f"  Совпадений в sensor_log:  {len(updates)}")
    print(f"  Без покрытия (sensor_log слишком редкий): {n_partial}")

    if not updates:
        print("  Нет данных для записи.")
        return

    # ── 4. Write back to forecast_verification ─────────────────────────────
    with sqlite3.connect(str(_RESEARCH_DB)) as conn:
        conn.executemany(
            "UPDATE forecast_verification SET "
            "  signed_error_temp_1h=?, signed_error_temp_2h=?, signed_error_temp_3h=?, "
            "  signed_error_pres_1h=?, signed_error_pres_2h=?, signed_error_pres_3h=? "
            "WHERE id=?",
            updates,
        )
        conn.commit()

    print(f"  Обновлено строк:          {len(updates)}")
    print("  Готово. Запустите correction-status, чтобы проверить счётчик.")
    print("═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibration_cli",
        description="Correction model management tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("correction-status",      help="Show correction model status")
    sub.add_parser("train-correction",       help="Train the correction model")
    sub.add_parser("rollback-correction",    help="Delete correction model files")
    sub.add_parser("validate",               help="Held-out validation: LSTM vs correction vs API")
    sub.add_parser("backfill-signed-errors", help="Backfill signed errors for old LSTM verifications")

    args = parser.parse_args()
    dispatch = {
        "correction-status":      cmd_correction_status,
        "train-correction":       cmd_train_correction,
        "rollback-correction":    cmd_rollback_correction,
        "validate":               cmd_validate,
        "backfill-signed-errors": cmd_backfill_signed_errors,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
