#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attack injector for household water-usage dataset (minute-level).
- Converts m^3 -> liters (if needed)
- Injects ONE attack per run for each type:
    (1) Additive (+Δ), anywhere, 4 consecutive days, same hour block each day
    (2) Deductive (–Δ), ONLY when usage>0 and ONLY in high-usage hours (morning/evening),
        4 consecutive days, same hour block each day
- Δ in [5, 30] L/min
- Prints summaries and writes two CSVs with liters_before/liters_after/attack_flag.

Paths:
- Input:  combined_minute_ALL.csv
- Output: combined_minute_ALL_additive_attacked.csv
          combined_minute_ALL_deductive_attacked.csv
"""

import sys
import random
from typing import Tuple, List
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG (adjust if needed)
# -----------------------------
CONFIG = {
    "INPUT_CSV": "combined_minute_ALL.csv",
    "OUT_ADD": "combined_minute_ALL_additive_attacked.csv",
    "OUT_DED": "combined_minute_ALL_deductive_attacked.csv",

    # If your file uses custom names, set these:
    "TIMESTAMP_COL": None,            # e.g., "timestamp"
    "CONSUMPTION_LITERS_COL": None,   # e.g., "liters"
    "CONSUMPTION_M3_COL": None,       # e.g., "m3"

    "RANDOM_SEED": 42,
    "DELTA_MIN": 5.0,      # L/min
    "DELTA_MAX": 30.0,     # L/min
    "DURATION_HOURS": 2,   # consecutive hours per attacked day
    "WINDOW_DAYS": 4,      # consecutive days

    # Deductive allowed hours (24h)
    "HIGH_USAGE_HOURS": list(range(6,10)) + list(range(18,22)),  # 6–9 & 18–21

    "USAGE_EPSILON": 1e-6  # treat <= as no usage
}

random.seed(CONFIG["RANDOM_SEED"])
np.random.seed(CONFIG["RANDOM_SEED"])


# -----------------------------
# Helpers
# -----------------------------
def autodetect_columns(df: pd.DataFrame,
                       ts_hint: str = None,
                       liters_hint: str = None,
                       m3_hint: str = None) -> Tuple[str, str]:
    """
    Return (timestamp_col, liters_col). If liters not found, convert from m3.
    """
    # Timestamp column
    ts_col = ts_hint
    if ts_col is None:
        candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if not candidates:
            # fallback: first column that can be parsed as datetime
            for c in df.columns:
                try:
                    pd.to_datetime(df[c])
                    candidates = [c]
                    break
                except Exception:
                    continue
        if not candidates:
            raise ValueError("Could not auto-detect timestamp column. Set CONFIG['TIMESTAMP_COL'].")

        ts_col = candidates[0]

    # parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().all():
        raise ValueError(f"Timestamp column '{ts_col}' could not be parsed as datetime.")

    # Liters vs m3 columns
    liters_col = liters_hint if (liters_hint and liters_hint in df.columns) else None
    if liters_col is not None:
        return ts_col, liters_col

    m3_col = m3_hint if (m3_hint and m3_hint in df.columns) else None

    if liters_col is None:
        # obvious liters names
        for c in df.columns:
            if "liter" in c.lower():
                liters_col = c
                break

    if liters_col is None:
        # detect m3 column if not provided
        if m3_col is None:
            for c in df.columns:
                cl = c.lower()
                if "m3" in cl or "m^3" in cl or "m³" in cl or cl.endswith("_m3") or cl.startswith("m3"):
                    m3_col = c
                    break

        if m3_col is not None:
            df["liters"] = pd.to_numeric(df[m3_col], errors="coerce").astype(float) * 1000.0
            liters_col = "liters"
        else:
            # fallback: choose the numeric column with highest variance
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found for consumption.")
            variances = df[numeric_cols].var().sort_values(ascending=False)
            liters_col = variances.index[0]  # assume already liters-scale

    return ts_col, liters_col


def choose_four_day_window(dates: List[pd.Timestamp]) -> pd.Timestamp:
    """
    Choose a start date such that start .. start+3 are consecutive calendar days in data.
    """
    uniq_days = sorted(pd.to_datetime(pd.Series(dates).dt.floor("D")).dropna().unique())
    if len(uniq_days) < CONFIG["WINDOW_DAYS"]:
        raise ValueError(f"Need at least {CONFIG['WINDOW_DAYS']} unique days in the dataset.")

    for i in range(len(uniq_days) - CONFIG["WINDOW_DAYS"] + 1):
        ok = True
        for k in range(CONFIG["WINDOW_DAYS"] - 1):
            if (uniq_days[i + k + 1] - uniq_days[i + k]).days != 1:
                ok = False
                break
        if ok:
            return uniq_days[i]
    # fallback: first day (tolerate gaps)
    return uniq_days[0]


def pick_hour_block(candidate_hours: List[int], block_len: int) -> List[int]:
    """
    Pick a contiguous block of 'block_len' hours from candidate_hours.
    If none exists, sample distinct hours.
    """
    candidate_hours = sorted(set(candidate_hours))
    if block_len <= 1:
        return [random.choice(candidate_hours)]

    # Find contiguous runs
    runs = []
    run = [candidate_hours[0]]
    for h in candidate_hours[1:]:
        if h == run[-1] + 1:
            run.append(h)
        else:
            runs.append(run)
            run = [h]
    runs.append(run)

    blocks = []
    for r in runs:
        if len(r) >= block_len:
            for i in range(len(r) - block_len + 1):
                blocks.append(r[i:i+block_len])

    if not blocks:
        return sorted(random.sample(candidate_hours, min(block_len, len(candidate_hours))))
    return random.choice(blocks)


def inject_attack(df: pd.DataFrame,
                  ts_col: str,
                  liters_col: str,
                  attack_type: str):
    """
    Inject a single attack into a copy of df.
    attack_type: "additive" or "deductive"
    Returns (df_attacked, summary_dict)
    """
    dfa = df.copy()
    dfa = dfa.sort_values(ts_col).reset_index(drop=True)

    dfa["date"] = dfa[ts_col].dt.floor("D")
    dfa["hour"] = dfa[ts_col].dt.hour

    # pick 4-day window
    start_day = choose_four_day_window(dfa["date"].tolist())
    days = [start_day + pd.Timedelta(days=k) for k in range(CONFIG["WINDOW_DAYS"])]

    # decide the hour block
    if attack_type == "additive":
        attack_hours = pick_hour_block(list(range(24)), CONFIG["DURATION_HOURS"])
    elif attack_type == "deductive":
        attack_hours = pick_hour_block(CONFIG["HIGH_USAGE_HOURS"], CONFIG["DURATION_HOURS"])
        if not attack_hours:
            raise ValueError("No valid hour block within HIGH_USAGE_HOURS for deductive attack.")
    else:
        raise ValueError("attack_type must be 'additive' or 'deductive'.")

    # Δ in [5, 30] L/min
    delta = round(float(np.random.uniform(CONFIG["DELTA_MIN"], CONFIG["DELTA_MAX"])), 2)

    # mark attack window
    dfa["attack_flag"] = 0
    affected_minutes = 0
    any_mask = False

    for d in days:
        day_mask = dfa["date"].eq(d)
        hour_mask = dfa["hour"].isin(attack_hours)
        this_mask = day_mask & hour_mask

        if attack_type == "deductive":
            # only when there is actual usage
            this_mask = this_mask & (pd.to_numeric(dfa[liters_col], errors="coerce").fillna(0.0) > CONFIG["USAGE_EPSILON"])

        dfa.loc[this_mask, "attack_flag"] = 1
        affected_minutes += int(this_mask.sum())
        any_mask = any_mask or bool(this_mask.any())

    if not any_mask:
        raise RuntimeError(
            f"No minutes matched the attack mask for {attack_type}. "
            f"Try different HIGH_USAGE_HOURS/DURATION_HOURS or check data coverage."
        )

    # before/after
    dfa["liters_before"] = pd.to_numeric(dfa[liters_col], errors="coerce").astype(float)
    if attack_type == "additive":
        dfa["liters_after"] = dfa["liters_before"] + (dfa["attack_flag"] * delta)
    else:  # deductive
        dfa["liters_after"] = dfa["liters_before"] - (dfa["attack_flag"] * delta)
        dfa.loc[dfa["liters_after"] < 0, "liters_after"] = 0.0

    dfa[f"{liters_col}_attacked"] = dfa["liters_after"]

    hrs_str = ", ".join([f"{h:02d}:00–{h:02d}:59" for h in attack_hours])
    day_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in days]
    summary = {
        "attack_type": attack_type,
        "delta_L_per_min": delta,
        "window_days": CONFIG["WINDOW_DAYS"],
        "days": day_strs,
        "hours": attack_hours,
        "hours_human": hrs_str,
        "affected_minutes": affected_minutes,
        "total_minutes": int(dfa.shape[0]),
        "start_day": day_strs[0],
        "end_day": day_strs[-1]
    }
    return dfa, summary


def print_summary(summary: dict, dfa: pd.DataFrame):
    atype = summary["attack_type"].upper()
    print(f"[Attack Summary]  type={atype}")
    print(f"  window_days: {summary['window_days']}")
    print(f"  start:       {summary['start_day']}  end: {summary['end_day']}")
    print(f"  hours:       {summary['hours_human']}  (local 24h)")
    print(f"  delta:       {summary['delta_L_per_min']} L/min")
    print(f"  affected minutes: {summary['affected_minutes']} / {summary['total_minutes']} total")

    before_sum = dfa.loc[dfa["attack_flag"]==1, "liters_before"].sum()
    after_sum  = dfa.loc[dfa["attack_flag"]==1, "liters_after"].sum()
    change = after_sum - before_sum
    sign = "+" if change >= 0 else "-"
    print(f"  sum_before (attacked window): {before_sum:.2f} L")
    print(f"  sum_after  (attacked window): {after_sum:.2f} L")
    print(f"  net change over attacked window: {sign}{abs(change):.2f} L")
    print("  days & timeframes attacked:")
    for d in summary["days"]:
        print(f"    - {d}: {summary['hours_human']}")


def main():
    df = pd.read_csv(CONFIG["INPUT_CSV"])

    # Detect timestamp & liters (convert m3 if needed)
    ts_col, liters_col = autodetect_columns(
        df,
        ts_hint=CONFIG["TIMESTAMP_COL"],
        liters_hint=CONFIG["CONSUMPTION_LITERS_COL"],
        m3_hint=CONFIG["CONSUMPTION_M3_COL"]
    )
    df[liters_col] = pd.to_numeric(df[liters_col], errors="coerce").astype(float)

    # ADDITIVE attack
    add_df, add_summary = inject_attack(df, ts_col, liters_col, attack_type="additive")
    add_df.to_csv(CONFIG["OUT_ADD"], index=False)
    print("\n==================== ADDITIVE ATTACK ====================")
    print_summary(add_summary, add_df)

    # DEDUCTIVE attack
    ded_df, ded_summary = inject_attack(df, ts_col, liters_col, attack_type="deductive")
    ded_df.to_csv(CONFIG["OUT_DED"], index=False)
    print("\n==================== DEDUCTIVE ATTACK ===================")
    print_summary(ded_summary, ded_df)

    # Dataset totals
    def totals(label, dfa):
        before = dfa["liters_before"].sum()
        after = dfa["liters_after"].sum()
        return f"{label}: total_before={before:.2f} L  total_after={after:.2f} L  delta={(after-before):.2f} L"

    print("\n==================== DATASET TOTALS =====================")
    print(totals("ADDITIVE", add_df))
    print(totals("DEDUCTIVE", ded_df))

    print("\n[OK] Saved attacked datasets:")
    print(f"  - {CONFIG['OUT_ADD']}")
    print(f"  - {CONFIG['OUT_DED']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", str(e))
        sys.exit(1)
