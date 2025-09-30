#!/usr/bin/env python3
"""
SWM Attack/Activity-Aware Detection Pipeline

Run from terminal:
  python swm_pipeline.py --smartmeter smartmeter.csv --activities activities.csv --info info.json --out out/

Inputs
  --smartmeter : CSV with timestamp + cumulative meter index (any header names)
  --activities : CSV with activity annotations; can have start/end or single timestamps and free-text labels
  --info       : (optional) JSON with "frame_from"/"frame_to" if activities have no timestamps
  --out        : output folder (created if missing)

Outputs (written to --out)
  swm_pipeline_scores.csv   : per-timestamp features and scores
  swm_pipeline_events.csv   : consolidated events (start, end, type)
  plots/*.png               : quick-look figures

Notes
- Designed to be robust to different column names and CSV delimiters.
- No external services required. Pure Python + pandas + matplotlib.

Author: (Your Name)
License: MIT
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smart Water Meter attack/activity-aware detection")
    p.add_argument("--smartmeter", required=True, help="Path to smartmeter CSV (cumulative index)")
    p.add_argument("--activities", required=True, help="Path to activities CSV")
    p.add_argument("--info", default=None, help="Optional info.json with frame bounds")
    p.add_argument("--out", default="out", help="Output directory")
    p.add_argument("--roll-var-w", type=int, default=6, help="Window for rolling variance (samples)")
    p.add_argument("--cusum-w", type=int, default=24, help="Window horizon for WL-like CUSUM (samples)")
    p.add_argument("--small-eps-frac", type=float, default=0.10, help="Tiny-flow cutoff as frac of 95th pct of nonzero increments")
    p.add_argument("--cal-frac", type=float, default=0.5, help="Fraction of earliest data for threshold calibration")
    p.add_argument("--alpha-q", type=float, default=0.999, help="Quantile for thresholds (e.g., 0.999)")
    p.add_argument("--min-event-gap", type=int, default=4, help="Merge detections with gaps ≤ this many samples")
    return p.parse_args()

# ----------------------- IO Utils -----------------------
def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Try common delimiters; return DataFrame."""
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)

def infer_time_value_columns(df: pd.DataFrame):
    """Return (df_renamed, time_col, value_col)."""
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={old:new for old,new in zip(df.columns, cols)})
    dtc = [c for c in df.columns if "time" in c or "date" in c or c in {"ts","timestamp"}]
    if not dtc:
        dtc = [df.columns[0]]
    tcol = dtc[0]
    # choose first non-time column as value
    vcol = next((c for c in df.columns if c != tcol), df.columns[1] if len(df.columns)>1 else None)
    return df, tcol, vcol

# ----------------------- Activities projection -----------------------
APPLIANCE_COLS = ["dishwasher","washing_machine","shower","toilet","tap","irrigation"]

def flags_from_text(text: str) -> dict:
    t = (str(text) if text is not None else "").lower()
    return {
        "dishwasher": ("dishwasher" in t) or ("dish " in t),
        "washing_machine": any(k in t for k in ["washing","laundry","washer"]),
        "shower": "shower" in t,
        "toilet": ("toilet" in t) or ("flush" in t),
        "tap": any(k in t for k in ["tap","faucet","sink"]),
        "irrigation": any(k in t for k in ["irrigation","sprinkler","garden"]),
    }

def project_activities(idx: pd.DatetimeIndex, acts: pd.DataFrame, info: dict|None) -> pd.DataFrame:
    proj = pd.DataFrame(index=idx)
    proj["activity_any"] = False
    for c in APPLIANCE_COLS:
        proj[c] = False

    acts = acts.copy()
    acts.columns = [c.strip().lower().replace(" ", "_") for c in acts.columns]
    dt_cols = [c for c in acts.columns if "time" in c or "date" in c or c in {"ts","timestamp","start","end","start_time","end_time"}]
    for c in dt_cols:
        acts[c] = pd.to_datetime(acts[c], errors="coerce", infer_datetime_format=True)

    start_col = next((c for c in acts.columns if c.startswith("start")), dt_cols[0] if dt_cols else None)
    end_col   = next((c for c in acts.columns if c.startswith("end")), None)

    def row_flags(row):
        txt = " ".join(str(row.get(c,"")) for c in acts.columns)
        return flags_from_text(txt)

    if start_col is not None and end_col is not None and acts[[start_col,end_col]].notna().any().all():
        # Interval events
        acts = acts.dropna(subset=[start_col, end_col])
        for _, r in acts.iterrows():
            s, e = r[start_col], r[end_col]
            if pd.isna(s) or pd.isna(e): 
                continue
            mask = (proj.index >= s) & (proj.index <= e)
            proj.loc[mask, "activity_any"] = True
            ff = row_flags(r)
            for k,v in ff.items():
                if v: proj.loc[mask, k] = True
    elif start_col is not None and end_col is None and start_col in acts.columns and acts[start_col].notna().any():
        # Point events → expand ±2 samples
        for _, r in acts.dropna(subset=[start_col]).iterrows():
            t = r[start_col]
            idxpos = proj.index.get_indexer([t], method="nearest")[0]
            lo = max(0, idxpos-2); hi = min(len(proj), idxpos+3)
            proj.iloc[lo:hi, proj.columns.get_loc("activity_any")] = True
            ff = row_flags(r)
            for k,v in ff.items():
                if v: proj.iloc[lo:hi, proj.columns.get_loc(k)] = True
    else:
        # No timestamps → use info.json frame bounds if present
        if info:
            fr = pd.to_datetime(info.get("frame_from") or info.get("frame_information", {}).get("frame_from"), errors="coerce")
            to = pd.to_datetime(info.get("frame_to")   or info.get("frame_information", {}).get("frame_to"),   errors="coerce")
            if pd.notna(fr) and pd.notna(to):
                mask = (proj.index >= fr) & (proj.index <= to)
                proj.loc[mask, "activity_any"] = True
                agg = {c: False for c in APPLIANCE_COLS}
                for _, r in acts.iterrows():
                    ff = row_flags(r)
                    for k,v in ff.items():
                        agg[k] = agg[k] or v
                for k,v in agg.items():
                    if v: proj.loc[mask, k] = True

    return proj

# ----------------------- Stats & detectors -----------------------
def window_var(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).var()

def cusum_windowed(series: pd.Series, mode: str, W: int) -> pd.Series:
    """WL-like CUSUM over sliding window: accumulate positive evidence in given direction."""
    x = series.astype(float).copy()
    x = x.replace([np.inf,-np.inf], np.nan).fillna(1.0)
    s = (x - 1.0) if mode=="up" else (1.0 - x)
    acc = s.rolling(W, min_periods=1).sum()
    return acc.clip(lower=0)

def events_from_score(score: pd.Series, thresh: float, min_gap: int):
    mask = score >= thresh
    idx = np.where(mask.values)[0]
    events = []
    if len(idx)==0:
        return events
    start = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            events.append((start, prev))
            start = i
        prev = i
    events.append((start, prev))
    ts = score.index
    return [(ts[s], ts[e], float(score.iloc[s:e+1].max())) for s,e in events]

# ----------------------- Main -----------------------
def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)

    # 1) Smartmeter ingest
    smart_raw = read_csv_flexible(Path(args.smartmeter))
    smart_raw, tcol, vcol = infer_time_value_columns(smart_raw)
    smart_raw[tcol] = pd.to_datetime(smart_raw[tcol], errors="coerce", infer_datetime_format=True)
    smart_raw[vcol] = pd.to_numeric(smart_raw[vcol], errors="coerce")
    smart = (smart_raw
             .dropna(subset=[tcol, vcol])
             .sort_values(tcol)
             .rename(columns={tcol:"timestamp", vcol:"cumulative"}))
    smart["delta_sec"] = smart["timestamp"].diff().dt.total_seconds()
    smart["increment"] = smart["cumulative"].diff()
    smart.loc[smart["increment"] < 0, "increment"] = np.nan
    smart["increment"] = smart["increment"].fillna(0.0)
    smart["rate_m3ph"] = smart.apply(lambda r: (r["increment"] / (r["delta_sec"]/3600.0)) if (pd.notnull(r["delta_sec"]) and r["delta_sec"]>0) else np.nan, axis=1)
    smart = smart.set_index("timestamp")

    # 2) Activities projection
    acts = read_csv_flexible(Path(args.activities))
    info = None
    if args.info:
        try:
            info = json.loads(Path(args.info).read_text())
        except Exception:
            info = None
    proj = project_activities(smart.index, acts, info)
    df = smart.join(proj, how="left").fillna({"activity_any": False})
    for c in APPLIANCE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(False)

    # 3) Baselines
    df["hour"]  = df.index.hour
    df["state"] = np.where(df["activity_any"], "act", "quiet")

    base_med = df.groupby(["state","hour"])["rate_m3ph"].median()
    df["baseline_rate_med"] = [base_med.get((s,h), np.nan) for s,h in zip(df["state"], df["hour"])]

    inc = df["increment"].fillna(0.0)
    hist_var_by_hour = inc.groupby(df["hour"]).var()
    df["roll_var_inc"] = window_var(inc, args.roll_var_w)
    df["hist_var_hour"] = df["hour"].map(hist_var_by_hour)

    nonzero = inc[inc>0]
    nonzero_p95 = float(np.nanpercentile(nonzero, 95)) if len(nonzero) else 0.0
    eps = args.small_eps_frac * nonzero_p95 if np.isfinite(nonzero_p95) else 0.0
    df["is_small_flow"] = (inc > 0) & (inc <= eps)
    small_frac_by_hour = df.groupby("hour")["is_small_flow"].mean()
    df["roll_small_frac"] = df["is_small_flow"].rolling(args.roll_var_w, min_periods=args.roll_var_w).mean()
    df["small_frac_hour"] = df["hour"].map(small_frac_by_hour)

    # 4) Residuals
    df["level_ratio"] = df["rate_m3ph"] / df["baseline_rate_med"].replace(0, np.nan)
    df["var_ratio"]   = (df["roll_var_inc"] / df["hist_var_hour"]).replace([np.inf,-np.inf], np.nan)
    df["small_flow_ratio"] = (df["roll_small_frac"] / df["small_frac_hour"].replace(0,np.nan)).replace([np.inf,-np.inf], np.nan)
    for col in ["level_ratio","var_ratio","small_flow_ratio"]:
        df[col] = df[col].fillna(1.0).clip(lower=1e-6)

    # 5) Scores (WL-like)
    df["score_level_up"]   = cusum_windowed(df["level_ratio"], mode="up",   W=args.cusum_w)
    df["score_level_down"] = cusum_windowed(df["level_ratio"], mode="down", W=args.cusum_w)
    df["score_var_up"]     = cusum_windowed(df["var_ratio"],   mode="up",   W=args.cusum_w)
    df["score_var_down"]   = cusum_windowed(df["var_ratio"],   mode="down", W=args.cusum_w)
    df["score_small_up"]   = cusum_windowed(df["small_flow_ratio"], mode="up",   W=args.cusum_w)
    df["score_small_down"] = cusum_windowed(df["small_flow_ratio"], mode="down", W=args.cusum_w)

    # 6) Thresholds (data-driven)
    n = len(df); cut = int(max(args.roll_var_w*10, args.cal_frac*n))
    train = df.iloc[:cut]
    def q(col): 
        return float(np.nanquantile(train[col], args.alpha_q))
    thresh = {
        "level_up":   q("score_level_up"),
        "level_down": q("score_level_down"),
        "var_up":     q("score_var_up"),
        "var_down":   q("score_var_down"),
        "small_up":   q("score_small_up"),
        "small_down": q("score_small_down"),
    }

    # 7) Events
    def events_from(colname, thr):
        return events_from_score(df[colname], thr, args.min_event_gap)

    over_up = events_from("score_level_up", thresh["level_up"])
    # annotate with small-tail inflation boolean
    over_up_annot = []
    for s,e,pk in over_up:
        tail_peak = float(df.loc[s:e, "score_small_up"].max())
        over_up_annot.append((s,e,pk, tail_peak >= thresh["small_up"]))

    events = {
        "over_billing_additive_like": over_up_annot,
        "under_billing_downscale_like": events_from("score_level_down", thresh["level_down"]),
        "smoothing_like": events_from("score_var_down", thresh["var_down"]),
        "party_like":     events_from("score_var_up",   thresh["var_up"]),
    }

    rows = []
    for etype, lst in events.items():
        for tup in lst:
            if len(tup)==3:
                s,e,pk = tup; tailflag = None
            else:
                s,e,pk,tailflag = tup
            rows.append({"event_type": etype, "start": s, "end": e, "peak_score": pk, "small_tail_excess": tailflag})
    events_df = pd.DataFrame(rows).sort_values(["start","event_type"])

    # 8) Save CSVs
    scores_csv = outdir / "swm_pipeline_scores.csv"
    events_csv = outdir / "swm_pipeline_events.csv"

    keep_cols = [
        "cumulative","delta_sec","increment","rate_m3ph","activity_any",
        "dishwasher","washing_machine","shower","toilet","tap","irrigation",
        "hour","state","baseline_rate_med","roll_var_inc","hist_var_hour",
        "is_small_flow","roll_small_frac","small_frac_hour",
        "level_ratio","var_ratio","small_flow_ratio",
        "score_level_up","score_level_down","score_var_up","score_var_down",
        "score_small_up","score_small_down"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[keep_cols].reset_index().to_csv(scores_csv, index=False)
    events_df.to_csv(events_csv, index=False)

    # 9) Plots
    def plot_series(ts, y, title, filename, over=None):
        plt.figure(figsize=(11,4))
        plt.plot(ts, y)
        if over is not None:
            plt.step(ts, over, where="post")
        plt.title(title); plt.xlabel("time"); plt.ylabel("value")
        plt.tight_layout()
        fp = plots/filename
        plt.savefig(fp); plt.close()
        return fp

    rate = df["rate_m3ph"]
    try:
        p95 = np.nanpercentile(rate, 95)
        lvl = p95 if np.isfinite(p95) else 1.0
    except Exception:
        lvl = 1.0
    act_overlay = df["activity_any"].astype(int) * (0.2*lvl)

    _ = plot_series(df.index, df["rate_m3ph"], "Rate (m^3/h) with activity overlay", "rate_activity.png", over=act_overlay)
    _ = plot_series(df.index, df["level_ratio"], "Level ratio (rate / baseline)", "level_ratio.png")
    _ = plot_series(df.index, df["score_level_up"], "Score: level UP (over-billing/additive-like)", "score_level_up.png")
    _ = plot_series(df.index, df["score_level_down"], "Score: level DOWN (under-billing/downscale-like)", "score_level_down.png")
    _ = plot_series(df.index, df["var_ratio"], "Variance ratio (window var / hourly hist var)", "var_ratio.png")
    _ = plot_series(df.index, df["score_var_down"], "Score: variance DOWN (smoothing-like)", "score_var_down.png")
    _ = plot_series(df.index, df["score_var_up"], "Score: variance UP (party-like)", "score_var_up.png")
    _ = plot_series(df.index, df["small_flow_ratio"], "Small-flow ratio (window vs hourly baseline)", "small_flow_ratio.png")
    _ = plot_series(df.index, df["score_small_up"], "Score: small-flow UP (tiny-flow inflation)", "score_small_up.png")

    # 10) Write thresholds + coverage
    thresh_json = outdir / "thresholds.json"
    thresh_json.write_text(json.dumps(thresh, indent=2, default=float))

    coverage = {
        "n_rows": int(len(df)),
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "median_sampling_seconds": float(np.nanmedian(df["delta_sec"])),
        "pct_activity_intervals": float(100.0 * df["activity_any"].mean() if "activity_any" in df.columns else 0.0),
        "calibration_rows": int(len(train)),
        "alpha_quantile": args.alpha_q,
    }
    (outdir/"coverage.json").write_text(json.dumps(coverage, indent=2, default=float))

    print(f"[OK] Wrote: {scores_csv}, {events_csv}, plots/, thresholds.json, coverage.json")

if __name__ == "__main__":
    main()
