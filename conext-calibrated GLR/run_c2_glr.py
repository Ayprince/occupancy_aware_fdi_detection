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

    # Injection / evaluation
    ap.add_argument("--inject", action="store_true", help="Inject synthetic attacks for evaluation")
    ap.add_argument("--add-delta", type=float, default=0.5, help="L/min additive pulse size if --inject")
    ap.add_argument("--ded-delta", type=float, default=0.5, help="L/min deductive pulse size if --inject")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject-anywhere", action="store_true", help="Inject attacks regardless of context gate (for testing)")
    ap.add_argument("--inject-in-train", action="store_true", help="Allow attacks in the training window; default injects only after training cutoff")

    # GLR / fusion / debounce
    ap.add_argument("--eta-pos", type=float, default=60.0, help="GLR threshold for positive (additive) shifts")
    ap.add_argument("--eta-neg", type=float, default=60.0, help="GLR threshold for negative (deductive) shifts")
    ap.add_argument("--debounce-K", type=int, default=3, help="Require at least K hits in last M to raise alarm")
    ap.add_argument("--debounce-M", type=int, default=10, help="Window M for debounce counting")
    ap.add_argument("--fusion", type=str, choices=["and","or"], default="and", help="Fuse conformal and GLR via AND (default) or OR")
    ap.add_argument("--no-glr", action="store_true", help="Disable GLR for debugging (conformal only)")
    ap.add_argument("--no-conformal", action="store_true", help="Disable conformal for debugging (GLR only)")

    # Conformal / sigma
    ap.add_argument("--sigma-scale", type=float, default=1.0, help="Multiply sigma by this factor (e.g., 0.7 more sensitive, 1.3 less)")
    ap.add_argument("--two-sided", action="store_true", help="Use two-sided conformal instead of one-sided")
    ap.add_argument("--min-conf-n", type=int, default=200, help="Fallback to global conformal if stratum buffer smaller than this")

    # Multiscale GLR and Bernoulli CUSUM
    ap.add_argument("--no-multiscale", action="store_true", help="Disable multiscale GLR detectors")
    ap.add_argument("--no-bernoulli", action="store_true", help="Disable Bernoulli exceedance CUSUM")
    ap.add_argument("--exceed-k", type=float, default=0.75, help="z=1 if one-sided residual > k*sigma")
    ap.add_argument("--no-bern-auto", action="store_true", help="Disable auto baseline for Bernoulli CUSUM; use p0/p1 flags")
    ap.add_argument("--bern-p0", type=float, default=0.1, help="Bernoulli baseline p0 (used if --no-bern-auto)")
    ap.add_argument("--bern-p1", type=float, default=0.35, help="Bernoulli attack p1 (used if --no-bern-auto)")
    ap.add_argument("--bern-delta", type=float, default=0.15, help="If auto, p1 = min(0.9, p0 + bern_delta)")
    ap.add_argument("--bern-h", type=float, default=8.0, help="Bernoulli CUSUM threshold h")
    # Injection / evaluation (add these just after your existing injection flags)
    ap.add_argument("--structured", action="store_true",
                    help="Use structured 4-day, same-hour attacks")
    ap.add_argument("--attack-mode", type=str, choices=["additive","deductive","both"], default="both")
    ap.add_argument("--delta-min", type=float, default=5.0, help="Min attack magnitude for structured attacks (L/min)")
    ap.add_argument("--delta-max", type=float, default=30.0, help="Max attack magnitude for structured attacks (L/min)")
    ap.add_argument("--hours", type=str, default="",
                    help="Comma-separated UTC hours to target (e.g., '6,7,18,19'); if empty, chosen automatically")
    ap.add_argument("--span-hours", type=int, default=2,
                    help="If --hours empty, target this many consecutive hours starting from a random base")
    ap.add_argument("--start-day", type=str, default="",
                    help="UTC date YYYY-MM-DD to start the 4-day window; if empty, chosen automatically")
        # Reporting
    ap.add_argument("--report", action="store_true", help="Print diagnostic detection rates in post-train window")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_prefix = csv_path.with_suffix("")
    out_dir = csv_path.parent
    fig_dir = out_dir / "figures"

    df = load_and_prepare(csv_path)
    df = add_time_features(df, "timestamp_utc")

    t0 = pd.to_datetime(df["timestamp_utc"], utc=True).min()
    train_cutoff = t0 + pd.Timedelta(days=args.train_days)

    if args.inject:
        labels_path = Path(f"{out_prefix}_labels.csv")
        attacked_path = Path(f"{out_prefix}_attacked.csv")

        if args.structured:
            from c2_glr import inject_structured_attacks

            modes = ([args.attack_mode] if args.attack_mode in ("additive","deductive")
                    else ["additive","deductive"])
            hours_list = [int(h.strip()) for h in args.hours.split(",") if h.strip()!=""] if args.hours else None
            start_day = pd.to_datetime(args.start_day, utc=True) if args.start_day else None

            df_attacked, labels, summary = inject_structured_attacks(
                df,
                modes=modes,
                delta_min=args.delta_min,
                delta_max=args.delta_max,
                hours=hours_list,
                span_hours=args.span_hours,
                days=4,
                start_day_utc=start_day,
                seed=args.seed,
            )
            df_to_use = df_attacked
            df_attacked.to_csv(attacked_path, index=False)
            labels.to_csv(labels_path, index=False, header=["label"])
            print(f"[OK] Saved injected series -> {attacked_path}")
            print(f"[OK] Saved labels         -> {labels_path}")

            # ---- Pretty summary ----
            print("[Attack Summary]")
            print(f"  window_days: {summary['window_days']}")
            print(f"  start UTC:   {summary['start_day_utc']}  end UTC: {summary['end_day_utc']}")
            for s in summary["attacks"]:
                print(f"  - mode={s['mode']:<9} delta={s['delta']:.2f} L/min  hours={s['hours']}")
                print(f"    targeted minutes={s['minutes_targeted']}  affected minutes={s['minutes_affected']}")
                print(f"    mean_before={s['mean_before']:.3f}  mean_after={s['mean_after']:.3f}")

        else:
            # keep your existing stochastic injector for baseline comparisons
            from c2_glr import AttackSpec, inject_attacks
            specs = [
                AttackSpec(mode="additive",  delta_L_per_min=args.add_delta, prob_per_min=0.004),
                AttackSpec(mode="deductive", delta_L_per_min=args.ded_delta, prob_per_min=0.004),
            ]
            df_attacked, labels = inject_attacks(
                df,
                specs,
                seed=args.seed,
                respect_gate=(not args.inject_anywhere),
                only_after_ts=(None if args.inject_in_train else train_cutoff),
            )
            df_to_use = df_attacked
            df_attacked.to_csv(attacked_path, index=False)
            labels.to_csv(labels_path, index=False, header=["label"])
            print(f"[OK] Saved injected series -> {attacked_path}")
            print(f"[OK] Saved labels         -> {labels_path}")
            try:
                import numpy as np
                uniq, cnts = np.unique(labels.values, return_counts=True)
                print("[Diag] label counts:", dict(zip(uniq.tolist(), cnts.tolist())))
            except Exception:
                pass
    else:
        df_to_use = df

    params = DetectorParams(
        train_days=args.train_days,
        alpha=args.alpha,
        eta_pos=args.eta_pos,
        eta_neg=args.eta_neg,
        debounce_K=args.debounce_K,
        debounce_M=args.debounce_M,
        fusion=args.fusion,
        use_glr=(not args.no_glr),
        use_conformal=(not args.no_conformal),
        sigma_scale=args.sigma_scale,
        one_sided=(not args.two_sided),
        use_multiscale=(not args.no_multiscale),
        use_bernoulli=(not args.no_bernoulli),
        exceed_k=args.exceed_k,
        min_conf_n=args.min_conf_n,
        bern_auto=(not args.no_bern_auto),
        bern_p0=args.bern_p0,
        bern_p1=args.bern_p1,
        bern_delta=args.bern_delta,
        bern_h=args.bern_h,
    )
    det = run_c2_glr(df_to_use, params)
    det_path = Path(f"{out_prefix}_detections.csv")
    det.to_csv(det_path, index=False)
    print(f"[OK] Saved detections -> {det_path}")

    # Figures
    df_aug = df_to_use.copy()
    df_aug["timestamp_utc"] = pd.to_datetime(df_aug["timestamp_utc"], utc=True)
    det["timestamp_utc"] = pd.to_datetime(det["timestamp_utc"], utc=True)

    plot_start = train_cutoff
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
        det_ts = pd.to_datetime(det["timestamp_utc"], utc=True)
        eval_mask = (det_ts >= train_cutoff).values

        L = min(len(labels), len(det))
        labels = labels[:L]
        eval_mask = eval_mask[:L]

        pred = np.zeros(L, dtype=int)
        pred[det["alarm_additive"].values[:L] == 1] = 1
        pred[det["alarm_deductive"].values[:L] == 1] = -1

        # Evaluate only on post-train minutes
        idx = np.where(eval_mask)[0]
        lab_sub = labels[idx]
        pred_sub = pred[idx]

        tp_add = int(((pred_sub == 1) & (lab_sub == 1)).sum())
        tp_ded = int(((pred_sub == -1) & (lab_sub == -1)).sum())
        fp = int(((pred_sub != 0) & (lab_sub == 0)).sum())
        fn_add = int(((pred_sub != 1) & (lab_sub == 1)).sum())
        fn_ded = int(((pred_sub != -1) & (lab_sub == -1)).sum())
        print(f"[Eval/post-train] additive: TP={tp_add}, FN={fn_add}")
        print(f"[Eval/post-train] deductive: TP={tp_ded}, FN={fn_ded}")
        print("[Eval/post-train] false positives:", fp)

        # Diagnostics: how many labels in post-train region
        try:
            import numpy as np
            uniq, cnts = np.unique(lab_sub, return_counts=True)
            print("[Diag/post-train] label counts:", dict(zip(uniq.tolist(), cnts.tolist())))
        except Exception:
            pass

        if args.report:
            # Diagnostic rates: GLR-only and conformal-only hit rates by label
            det_sub = det.iloc[idx].copy()
            glr_add_rate = float((det_sub["glr_add_raw"] == 1).mean()) if "glr_add_raw" in det_sub else float("nan")
            glr_ded_rate = float((det_sub["glr_ded_raw"] == 1).mean()) if "glr_ded_raw" in det_sub else float("nan")
            conf_add_rate = float((det_sub.get("p_pos", det_sub["p_value"]) < args.alpha).mean())
            conf_ded_rate = float((det_sub.get("p_neg", det_sub["p_value"]) < args.alpha).mean())
            print(f"[Report] GLR add raw rate (post-train): {glr_add_rate:.4f}")
            print(f"[Report] GLR ded raw rate (post-train): {glr_ded_rate:.4f}")
            print(f"[Report] Conformal add p<alpha rate (post-train): {conf_add_rate:.4f}")
            print(f"[Report] Conformal ded p<alpha rate (post-train): {conf_ded_rate:.4f}")

            # Conditional hit rates inside labeled windows
            lab_series = pd.Series(lab_sub, index=det_sub.index)
            def rate(mask, col):
                m = (lab_series == mask)
                return float((det_sub.loc[m, col] == 1).mean()) if m.any() else float('nan')
            print(f"[Report] GLR add hit | label=+1: {rate(1, 'glr_add_raw'):.4f}")
            print(f"[Report] GLR ded hit | label=-1: {rate(-1, 'glr_ded_raw'):.4f}")
            print(f"[Report] MS add hit  | label=+1: {rate(1, 'ms_pos_hit'):.4f}")
            print(f"[Report] MS ded hit  | label=-1: {rate(-1, 'ms_neg_hit'):.4f}")
            print(f"[Report] Bern add hit| label=+1: {rate(1, 'bern_pos_hit'):.4f}")
            print(f"[Report] Bern ded hit| label=-1: {rate(-1, 'bern_neg_hit'):.4f}")
            # Conformal one-sided rates inside labels
            if 'p_pos' in det_sub and 'p_neg' in det_sub:
                conf_add_in = float((det_sub.loc[lab_series==1, 'p_pos'] < args.alpha).mean()) if (lab_series==1).any() else float('nan')
                conf_ded_in = float((det_sub.loc[lab_series==-1, 'p_neg'] < args.alpha).mean()) if (lab_series==-1).any() else float('nan')
                print(f"[Report] Conformal add p<alpha | label=+1: {conf_add_in:.4f}")
                print(f"[Report] Conformal ded p<alpha | label=-1: {conf_ded_in:.4f}")

    print("[DONE] Figures saved under:", fig_dir)


if __name__ == "__main__":
    main()
