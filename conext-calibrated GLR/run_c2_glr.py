
"""
run_c2_glr.py
-------------
Runner script for C²-GLR on a household water dataset.

Adds defensive cleaning to eliminate NaNs before training.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from c2_glr import (
    DetectorParams, run_c2_glr, inject_attacks, add_time_features
)


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure timestamp exists and is parseable
    if "timestamp_utc" not in df.columns:
        raise ValueError("CSV must include 'timestamp_utc'")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    # Consumption to L/min
    if "consumption_L_per_min" not in df.columns:
        if "consumption_m3_per_min" in df.columns:
            df["consumption_L_per_min"] = df["consumption_m3_per_min"].astype(float) * 1000.0
        else:
            raise ValueError("Expected consumption_m3_per_min or consumption_L_per_min in CSV")

    # Clip negatives and interpolate small gaps in consumption
    df["consumption_L_per_min"] = df["consumption_L_per_min"].astype(float)
    df["consumption_L_per_min"] = df["consumption_L_per_min"].clip(lower=0)

    # Binary columns: fill NaN -> 0 and cast to int
    for b in ["occupied", "has_activity", "dishwasher_running", "washing_machine_running"]:
        if b not in df.columns:
            df[b] = 0
        df[b] = df[b].fillna(0).astype(int)

    # Optional: interpolate short missing runs in consumption (time-aware)
    if df["consumption_L_per_min"].isna().any():
        df = df.set_index("timestamp_utc")
        df["consumption_L_per_min"] = df["consumption_L_per_min"].interpolate(method="time", limit=60).fillna(0.0)
        df = df.reset_index()

    return df


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to combined_minute_ALL.csv")
    ap.add_argument("--train-days", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--inject", action="store_true", help="Inject synthetic attacks for evaluation")
    ap.add_argument("--add-delta", type=float, default=0.5, help="L/min additive pulse size if --inject")
    ap.add_argument("--ded-delta", type=float, default=0.5, help="L/min deductive pulse size if --inject")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_prefix = csv_path.with_suffix("")
    out_dir = csv_path.parent
    fig_dir = out_dir / "figures"

    df = load_and_prepare(csv_path)

    if args.inject:
        from c2_glr import AttackSpec
        specs = [
            AttackSpec(mode="additive", delta_L_per_min=args.add_delta, prob_per_min=0.004),
            AttackSpec(mode="deductive", delta_L_per_min=args.ded_delta, prob_per_min=0.004),
        ]
        df_attacked, labels = inject_attacks(df, specs, seed=args.seed)
        df_to_use = df_attacked
        labels_path = Path(f"{out_prefix}_labels.csv")
        attacked_path = Path(f"{out_prefix}_attacked.csv")
        df_attacked.to_csv(attacked_path, index=False)
        labels.to_csv(labels_path, index=False, header=["label"])
        print(f"[OK] Saved injected series -> {attacked_path}")
        print(f"[OK] Saved labels         -> {labels_path}")
    else:
        df_to_use = df

    params = DetectorParams(train_days=args.train_days, alpha=args.alpha)
    det = run_c2_glr(df_to_use, params)
    det_path = Path(f"{out_prefix}_detections.csv")
    det.to_csv(det_path, index=False)
    print(f"[OK] Saved detections -> {det_path}")

    # Figures
    df_aug = df_to_use.copy()
    df_aug["timestamp_utc"] = pd.to_datetime(df_aug["timestamp_utc"], utc=True)
    det["timestamp_utc"] = pd.to_datetime(det["timestamp_utc"], utc=True)

    t0 = df_aug["timestamp_utc"].min()
    plot_start = t0 + pd.Timedelta(days=params.train_days)
    plot_end = plot_start + pd.Timedelta(days=7)

    m = (det["timestamp_utc"] >= plot_start) & (det["timestamp_utc"] <= plot_end)
    if m.any():
        fig1 = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.plot(det.loc[m, "timestamp_utc"], det.loc[m, "y"], label="y (L/min)")
        ax.plot(det.loc[m, "timestamp_utc"], det.loc[m, "yhat"], label="yhat (L/min)")
        ax.set_title("Consumption vs Forecast (sample week)")
        ax.set_xlabel("Time")
        ax.set_ylabel("L/min")
        ax.legend()
        save_fig(fig_dir / "consumption_vs_forecast.png")

        fig2 = plt.figure(figsize=(12, 3))
        ax2 = plt.gca()
        ax2.plot(det.loc[m, "timestamp_utc"], det.loc[m, "p_value"], label="Conformal p-value")
        ax2.axhline(params.alpha, linestyle="--")
        ax2.set_title("Conformal p-values (sample week)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("p-value")
        ax2.legend()
        save_fig(fig_dir / "pvalues.png")

        fig3 = plt.figure(figsize=(12, 3))
        ax3 = plt.gca()
        ax3.plot(det.loc[m, "timestamp_utc"], det.loc[m, "S_pos"], label="S_pos")
        ax3.plot(det.loc[m, "timestamp_utc"], det.loc[m, "S_neg"], label="S_neg")
        ax3.set_title("GLR-CUSUM stats (sample week)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("stat")
        ax3.legend()
        save_fig(fig_dir / "glr_stats.png")

    if args.inject:
        labels_path = Path(f"{out_prefix}_labels.csv")
        labels = pd.read_csv(labels_path)["label"].values
        L = min(len(labels), len(det))
        labels = labels[:L]
        pred = np.zeros(L, dtype=int)
        pred[det["alarm_additive"].values[:L] == 1] = 1
        pred[det["alarm_deductive"].values[:L] == 1] = -1

        tp_add = int(((pred == 1) & (labels == 1)).sum())
        tp_ded = int(((pred == -1) & (labels == -1)).sum())
        fp = int(((pred != 0) & (labels == 0)).sum())
        fn_add = int(((pred != 1) & (labels == 1)).sum())
        fn_ded = int(((pred != -1) & (labels == -1)).sum())
        print("[Eval] additive: TP={}, FN={}".format(tp_add, fn_add))
        print("[Eval] deductive: TP={}, FN={}".format(tp_ded, fn_ded))
        print("[Eval] false positives:", fp)

    print("[DONE] Figures saved under:", fig_dir)


if __name__ == "__main__":
    main()
