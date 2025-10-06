"""
c2_glr.py
---------
Context-Calibrated GLR (C²-GLR) with Online Conformal Calibration and Change-Point–Aware Resets.

Now includes robust NaN handling:
 - Columns are imputed with training medians inside QuantileForecaster.
 - y with NaN is dropped from the forecaster's training portion.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Utility / Feature Engineering
# -----------------------------

def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["ts"] = ts
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    return df


def safe_rolling_features(df: pd.DataFrame, y_col: str, windows=(5, 15, 60)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{y_col}_rollmean_{w}"] = (
            df[y_col].rolling(w, min_periods=max(1, w//3)).mean().bfill().astype(float)
        )
        df[f"{y_col}_rollstd_{w}"] = (
            df[y_col].rolling(w, min_periods=max(1, w//3)).std().fillna(0.0).astype(float)
        )
    return df


def add_lags(df: pd.DataFrame, y_col: str, lags=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    df = df.copy()
    for L in lags:
        df[f"{y_col}_lag_{L}"] = df[y_col].shift(L).bfill().astype(float)
    return df


def default_gate(row: pd.Series) -> int:
    """
    High-usage context gate: (occupied OR has_activity) AND (evening OR appliance_on OR has_activity)
    - Robust to missing columns; defaults to 0.
    """
    occupied = int(row.get("occupied", 0) or 0)
    has_act  = int(row.get("has_activity", 0) or 0)
    hour     = int(row.get("hour", 0) or 0)
    app_on   = int(row.get("dishwasher_running", 0) or 0) or int(row.get("washing_machine_running", 0) or 0)
    is_even  = 18 <= hour < 23
    # Gate opens if there is explicit activity, or if occupied and (evening or an appliance is on)
    return int((has_act == 1) or (occupied == 1 and (is_even or app_on)))

# --------- Parameterized gate policy ---------
def gate_from_policy(row: pd.Series, policy: str = "strict") -> int:
    """Return gate according to policy.
    strict  : existing default_gate
    lenient : open if (occupied OR has_activity OR evening 6-22 OR any appliance on)
    always  : gate always open (1)
    """
    policy = (policy or "strict").lower()
    if policy == "always":
        return 1
    hour = int(row.get("hour", 0) or 0)
    app_on = int(row.get("dishwasher_running", 0) or 0) or int(row.get("washing_machine_running", 0) or 0)
    if policy == "lenient":
        occ = int(row.get("occupied", 0) or 0)
        act = int(row.get("has_activity", 0) or 0)
        is_evening = 6 <= hour < 23
        return int((occ == 1) or (act == 1) or is_evening or app_on)
    # default
    return default_gate(row)


def stratum_id(row: pd.Series) -> Tuple[int, int, int]:
    """
    Define a context stratum for conformal calibration.
    (occupied, hour_bucket(=hour), appliance_any)
    """
    occ = int(row.get("occupied", 0) or 0)
    hour = int(row.get("hour", 0) or 0)
    app = int((row.get("dishwasher_running", 0) or 0) or (row.get("washing_machine_running", 0) or 0))
    return (occ, hour, int(app))


# -----------------------------
# Quantile Forecaster (with NaN-safe imputation)
# -----------------------------

@dataclass
class QuantileForecaster:
    """
    Fast quantile forecaster using scikit-learn's GradientBoostingRegressor.
    Trains three models for q10, q50, q90 to derive a robust spread estimate.
    Now includes median imputation learned from training data.
    """
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    max_depth: int = 3
    n_estimators: int = 150
    learning_rate: float = 0.05
    random_state: int = 0

    models: Dict[float, GradientBoostingRegressor] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    feature_names_: Optional[List[str]] = None
    medians_: Optional[pd.Series] = None  # per-column medians for imputation

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.feature_names_:
            if c not in X.columns:
                X[c] = self.medians_.get(c, 0.0)
        return X.fillna(self.medians_)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)

        # Drop rows where y is NaN
        mask = ~pd.isna(y.values)
        X_train = X.loc[mask].copy()
        y_train = y.loc[mask].astype(float)

        # Compute column medians from training features
        self.medians_ = X_train.median(numeric_only=True)
        X_train = self._impute(X_train)

        # Scale + fit quantile models
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train.values)
        for q in self.quantiles:
            m = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            m.fit(Xs, y_train.values.astype(float))
            self.models[q] = m
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None and self.medians_ is not None
        Xi = self._impute(X[self.feature_names_])
        return self.scaler.transform(Xi.values)

    def predict_quants(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs = self._transform(X)
        q10 = self.models[self.quantiles[0]].predict(Xs)
        q50 = self.models[self.quantiles[1]].predict(Xs)
        q90 = self.models[self.quantiles[2]].predict(Xs)
        return q10, q50, q90

    def predict_point_and_sigma(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        q10, q50, q90 = self.predict_quants(X)
        sigma = np.maximum(q90 - q50, q50 - q10)
        # avoid sigma=0
        if np.any(sigma > 0):
            floor = max(1e-6, 0.05 * np.median(sigma[sigma>0]))
        else:
            floor = 1e-3
        sigma = np.maximum(sigma, floor)
        return q50, sigma


# -----------------------------
# Online Conformal Calibration
# -----------------------------

@dataclass
class ConformalCalibrator:
    window_per_stratum: int = 2000
    buffers: Dict[Tuple[int,int,int], deque] = field(default_factory=lambda: defaultdict(deque))

    def reset_all(self):
        self.buffers = defaultdict(deque)

    def update_and_pvalue(self, stratum: Tuple[int,int,int], score: float) -> float:
        buf = self.buffers[stratum]
        if len(buf) == 0:
            p = 1.0
        else:
            ge = sum(1 for s in buf if s >= score)
            p = (1 + ge) / (1 + len(buf))
        buf.append(float(score))
        if len(buf) > self.window_per_stratum:
            buf.popleft()
        self.buffers[stratum] = buf
        return float(p)

    def size(self, stratum: Tuple[int,int,int]) -> int:
        return len(self.buffers[stratum])


# -----------------------------
# Residual CPD (CUSUM variant)
# -----------------------------

@dataclass
class ResidualCUSUM:
    k: float = 0.0
    h: float = 30.0
    drift: float = 0.0
    sum_pos: float = 0.0
    sum_neg: float = 0.0

    def update(self, r: float) -> bool:
        x = r - self.k - self.drift
        self.sum_pos = max(0.0, self.sum_pos + x)
        self.sum_neg = min(0.0, self.sum_neg + x)
        if self.sum_pos > self.h or abs(self.sum_neg) > self.h:
            self.sum_pos = 0.0
            self.sum_neg = 0.0
            return True
        return False


# -----------------------------
# State-Gated GLR-CUSUM
# -----------------------------

@dataclass
class GLRCUSUM:
    eta_pos: float = 50.0
    eta_neg: float = 50.0
    lam: float = 0.05
    sigma_floor: float = 1e-3

    S_pos: float = 0.0
    S_neg: float = 0.0
    mu_pos: float = 0.0
    mu_neg: float = 0.0

    def update(self, r: float, sigma: float, gate: int) -> Tuple[bool, bool]:
        if not gate:
            self.S_pos = max(0.0, self.S_pos * 0.9)
            self.S_neg = max(0.0, self.S_neg * 0.9)
            return (False, False)

        s2 = max(sigma**2, self.sigma_floor**2)

        # Positive shift branch (on r)
        self.mu_pos = (1 - self.lam) * self.mu_pos + self.lam * r
        ll_pos = 0.5 * ((r**2 - (r - self.mu_pos)**2) / s2)
        self.S_pos = max(0.0, self.S_pos + ll_pos)
        alarm_pos = self.S_pos > self.eta_pos
        if alarm_pos:
            self.S_pos = 0.0

        # Negative shift branch: operate on r_neg = -r, with its own mean
        r_neg = -r
        self.mu_neg = (1 - self.lam) * self.mu_neg + self.lam * r_neg
        ll_neg = 0.5 * ((r_neg**2 - (r_neg - self.mu_neg)**2) / s2)
        self.S_neg = max(0.0, self.S_neg + ll_neg)
        alarm_neg = self.S_neg > self.eta_neg
        if alarm_neg:
            self.S_neg = 0.0

        return (alarm_pos, alarm_neg)


# -----------------------------
# End-to-end runner
# -----------------------------

@dataclass
class DetectorParams:
    train_days: int = 14
    conf_window: int = 2000
    alpha: float = 0.005
    debounce_K: int = 3
    debounce_M: int = 10

    eta_pos: float = 60.0
    eta_neg: float = 60.0
    lam: float = 0.05
    cpd_h: float = 40.0

    roll_windows: Tuple[int,...] = (5, 15, 60)
    lags: Tuple[int,...] = (1, 2, 3, 5, 10)

    # how to fuse conformal and GLR decisions: "and" or "or"
    fusion: str = "and"
    # enable/disable components for debugging
    use_conformal: bool = True
    use_glr: bool = True

    # scale factor for sigma used in scoring ( >1.0 more permissive )
    sigma_scale: float = 1.0
    # use one-sided conformal (separate p for + and - residuals)
    one_sided: bool = True

    # Conformal fallback: if stratum count < min_conf_n, fall back to global buffer
    min_conf_n: int = 200

    # Multiscale GLR settings
    use_multiscale: bool = True
    ms_windows: Tuple[int, ...] = (5, 15)
    ms_eta_pos: Tuple[float, ...] = (12.0, 8.0)
    ms_eta_neg: Tuple[float, ...] = (12.0, 8.0)

    # Bernoulli exceedance CUSUM settings
    use_bernoulli: bool = True
    exceed_k: float = 0.75
    bern_auto: bool = True
    bern_delta: float = 0.15  # target uplift p1 = min(0.9, p0 + bern_delta)
    bern_p0: float = 0.1
    bern_p1: float = 0.35
    bern_h: float = 8.0
    # Gate policy: "strict" (default), "lenient", or "always"
    gate_mode: str = "strict"
class BernoulliCUSUM:
    """
    One-sided Bernoulli CUSUM on exceedance indicators z_t in {0,1}.
    Under H0: P(z=1)=p0; Under H1: P(z=1)=p1 (>p0 for additive, and define z accordingly for deductive).
    Update: S_t = max(0, S_{t-1} + log(p1/p0)*z + log((1-p1)/(1-p0))*(1-z)).
    Alarm when S_t > h; then reset.
    """
    def __init__(self, p0: float = 0.25, p1: float = 0.55, h: float = 8.0):
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.h = float(h)
        self.S = 0.0
        # Precompute increments
        self.a1 = np.log(self.p1 / self.p0)
        self.a0 = np.log((1 - self.p1) / (1 - self.p0))

    def update(self, z: int, gate: int) -> bool:
        if not gate:
            # light decay when gate is off
            self.S = max(0.0, self.S * 0.9)
            return False
        inc = self.a1 if z == 1 else self.a0
        self.S = max(0.0, self.S + inc)
        if self.S > self.h:
            self.S = 0.0
            return True
        return False


def make_features(df: pd.DataFrame, y_col: str = "consumption_L_per_min") -> Tuple[pd.DataFrame, List[str]]:
    df = add_time_features(df, "timestamp_utc")
    df = safe_rolling_features(df, y_col, windows=(5,15,60))
    df = add_lags(df, y_col, lags=(1,2,3,5,10))

    feats = [
        "hour", "dow", "is_weekend", "hour_of_week",
        "occupied", "has_activity",
        "dishwasher_running", "washing_machine_running",
        f"{y_col}_rollmean_5", f"{y_col}_rollmean_15", f"{y_col}_rollmean_60",
        f"{y_col}_rollstd_5", f"{y_col}_rollstd_15", f"{y_col}_rollstd_60",
        f"{y_col}_lag_1", f"{y_col}_lag_2", f"{y_col}_lag_3", f"{y_col}_lag_5", f"{y_col}_lag_10",
    ]

    for c in feats:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feats].astype(float)
    return X, feats


def run_c2_glr(df: pd.DataFrame, params: DetectorParams) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp_utc")

    assert "consumption_L_per_min" in df.columns
    # Guard negatives & NaNs in target
    df["consumption_L_per_min"] = df["consumption_L_per_min"].astype(float).clip(lower=0)

    for b in ["occupied", "has_activity", "dishwasher_running", "washing_machine_running"]:
        if b not in df.columns:
            df[b] = 0
        df[b] = df[b].fillna(0).astype(int)

    X, feat_names = make_features(df, "consumption_L_per_min")
    y = df["consumption_L_per_min"].astype(float).values

    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    t0 = ts.min()
    cutoff = t0 + pd.Timedelta(days=params.train_days)
    train_idx = ts < cutoff

    f = QuantileForecaster()
    f.fit(X.loc[train_idx], pd.Series(y[train_idx]))

    cal = ConformalCalibrator(window_per_stratum=params.conf_window)
    cpd = ResidualCUSUM(h=params.cpd_h)
    glr = GLRCUSUM(eta_pos=params.eta_pos, eta_neg=params.eta_neg, lam=params.lam)

    # Optional multiscale GLR detectors (on smoothed residuals)
    glr_ms = []
    if params.use_multiscale and len(params.ms_windows) > 0:
        for w, epos, eneg in zip(params.ms_windows, params.ms_eta_pos, params.ms_eta_neg):
            glr_ms.append((w, GLRCUSUM(eta_pos=epos, eta_neg=eneg, lam=params.lam)))

    out = []
    from collections import deque
    deb_add_q = deque(maxlen=params.debounce_M)
    deb_ded_q = deque(maxlen=params.debounce_M)

    yhat_all, sigma_all = f.predict_point_and_sigma(X)
    resid_all = y - yhat_all

    # Optional sigma scaling (smaller sigma -> more sensitive; larger -> less sensitive)
    sigma_all = sigma_all * float(params.sigma_scale)

    # Precompute multiscale smoothed residuals/sigmas (centered moving averages)
    ms_data = {}
    if params.use_multiscale and len(glr_ms) > 0:
        for (w, _g) in glr_ms:
            r_sm = pd.Series(resid_all).rolling(w, min_periods=w).mean().fillna(0.0).values
            s2_sm = pd.Series(sigma_all**2).rolling(w, min_periods=w).mean().values
            # Backfill NaNs with median variance
            if np.isnan(s2_sm).any():
                med = np.nanmedian(s2_sm)
                if not np.isfinite(med) or med <= 0:
                    med = np.median(sigma_all**2)
                s2_sm = np.where(np.isnan(s2_sm), med, s2_sm)
            sig_sm = np.sqrt(np.maximum(s2_sm, 1e-6))
            ms_data[w] = (r_sm, sig_sm)

    # Global and stratified conformal calibrators (one-sided)
    cal_pos = ConformalCalibrator(window_per_stratum=params.conf_window)
    cal_neg = ConformalCalibrator(window_per_stratum=params.conf_window)
    cal_pos_global = ConformalCalibrator(window_per_stratum=params.conf_window)
    cal_neg_global = ConformalCalibrator(window_per_stratum=params.conf_window)
    STRAT_GLOBAL_KEY = (-1, -1, -1)

    # Auto-tune Bernoulli baseline from training window
    if params.use_bernoulli:
        zpos_train = (np.maximum(0.0, resid_all[train_idx]) > params.exceed_k * sigma_all[train_idx]).astype(int)
        zneg_train = (np.maximum(0.0, -resid_all[train_idx]) > params.exceed_k * sigma_all[train_idx]).astype(int)
        p0_pos = float(np.clip(zpos_train.mean() if zpos_train.size else 0.01, 0.01, 0.3))
        p0_neg = float(np.clip(zneg_train.mean() if zneg_train.size else 0.01, 0.01, 0.3))
        if not params.bern_auto:
            p0_pos = params.bern_p0
            p0_neg = params.bern_p0
        p1_pos = float(min(0.9, p0_pos + params.bern_delta)) if params.bern_auto else params.bern_p1
        p1_neg = float(min(0.9, p0_neg + params.bern_delta)) if params.bern_auto else params.bern_p1
        bern_add = BernoulliCUSUM(p0=p0_pos, p1=p1_pos, h=params.bern_h)
        bern_ded = BernoulliCUSUM(p0=p0_neg, p1=p1_neg, h=params.bern_h)
    else:
        bern_add = None
        bern_ded = None

    for i in range(len(df)):
        row = df.iloc[i]
        y_t = float(y[i])
        yhat_t = float(yhat_all[i])
        sigma_t = float(sigma_all[i])
        r_t = float(resid_all[i])

        cpd_trig = cpd.update(r_t)
        if cpd_trig:
            cal.reset_all()
            cal_pos.reset_all()
            cal_neg.reset_all()
            cal_pos_global.reset_all()
            cal_neg_global.reset_all()

        # Conformal scores with hierarchical fallback (one-sided if enabled)
        s_id = stratum_id(row)
        if params.one_sided:
            score_pos = max(0.0, r_t) / max(sigma_t, 1e-6)
            score_neg = max(0.0, -r_t) / max(sigma_t, 1e-6)
            # local p-values
            p_pos_local = cal_pos.update_and_pvalue(s_id, score_pos)
            p_neg_local = cal_neg.update_and_pvalue(s_id, score_neg)
            # global p-values (single global stratum)
            p_pos_global_v = cal_pos_global.update_and_pvalue(STRAT_GLOBAL_KEY, score_pos)
            p_neg_global_v = cal_neg_global.update_and_pvalue(STRAT_GLOBAL_KEY, score_neg)
            # fallback if local buffers are small
            use_local_pos = cal_pos.size(s_id) >= params.min_conf_n
            use_local_neg = cal_neg.size(s_id) >= params.min_conf_n
            p_pos = p_pos_local if use_local_pos else p_pos_global_v
            p_neg = p_neg_local if use_local_neg else p_neg_global_v
        else:
            score = abs(r_t) / max(sigma_t, 1e-6)
            p_local = cal.update_and_pvalue(s_id, score)
            p_global = cal_pos_global.update_and_pvalue(STRAT_GLOBAL_KEY, score)
            use_local = cal.size(s_id) >= params.min_conf_n
            p = p_local if use_local else p_global
            p_pos = p_neg = p

        # Gate and GLR-CUSUM
        gate = gate_from_policy(row, getattr(params, "gate_mode", "strict"))
        a_pos, a_neg = glr.update(r_t, sigma_t, gate)

        # Multiscale GLR updates
        ms_pos_hit = False
        ms_neg_hit = False
        if params.use_multiscale and len(glr_ms) > 0:
            for (w, gdet) in glr_ms:
                r_sm, sig_sm = ms_data[w]
                ap, an = gdet.update(float(r_sm[i]), float(sig_sm[i]), gate)
                ms_pos_hit = ms_pos_hit or ap
                ms_neg_hit = ms_neg_hit or an

        # Bernoulli exceedance CUSUM updates (one-sided)
        bern_pos_hit = False
        bern_neg_hit = False
        if params.use_bernoulli:
            z_pos = int(r_t > params.exceed_k * sigma_t)
            z_neg = int((-r_t) > params.exceed_k * sigma_t)
            bern_pos_hit = bern_add.update(z_pos, gate)
            bern_neg_hit = bern_ded.update(z_neg, gate)

        # Composite raw GLR-like hits
        glr_add_raw = a_pos or ms_pos_hit or bern_pos_hit
        glr_ded_raw = a_neg  or ms_neg_hit or bern_neg_hit

        # Combine evidence with configurable fusion
        glr_add = bool(glr_add_raw) if params.use_glr else False
        glr_ded = bool(glr_ded_raw) if params.use_glr else False
        conf_add = (p_pos < params.alpha) if params.use_conformal else False
        conf_ded = (p_neg < params.alpha) if params.use_conformal else False

        if params.fusion.lower() == "or":
            add_hit = glr_add or conf_add
            ded_hit = glr_ded or conf_ded
        else:  # default "and"
            add_hit = glr_add and conf_add
            ded_hit = glr_ded and conf_ded

        deb_add_q.append(1 if add_hit else 0)
        deb_ded_q.append(1 if ded_hit else 0)
        alarm_add = int(sum(deb_add_q) >= params.debounce_K)
        alarm_ded = int(sum(deb_ded_q) >= params.debounce_K)

        out.append({
            "timestamp_utc": row["timestamp_utc"],
            "y": y_t,
            "yhat": yhat_t,
            "resid": r_t,
            "sigma": sigma_t,
            "p_value": (p_pos if params.one_sided else p_pos),
            "gate": int(gate),
            "S_pos": glr.S_pos,
            "S_neg": glr.S_neg,
            "alarm_additive": alarm_add,
            "alarm_deductive": alarm_ded,
            "cpd_reset": int(cpd_trig),
            "glr_add_raw": int(glr_add),
            "glr_ded_raw": int(glr_ded),
            "p_below_alpha": int((p_pos if params.one_sided else p_pos) < params.alpha),
            "p_pos": float(p_pos),
            "p_neg": float(p_neg),
            "ms_pos_hit": int(ms_pos_hit),
            "ms_neg_hit": int(ms_neg_hit),
            "bern_pos_hit": int(bern_pos_hit),
            "bern_neg_hit": int(bern_neg_hit),
        })

    return pd.DataFrame(out)


# -----------------------------
# Attack Injection (for evaluation)
# -----------------------------

@dataclass
class AttackSpec:
    mode: str            # "additive" or "deductive"
    delta_L_per_min: float
    min_len: int = 5
    max_len: int = 60
    prob_per_min: float = 0.005

# -----------------------------
# Structured Attack Injection (4-day, same-hour pattern)
# -----------------------------

def _find_consecutive_4day_block(
    ts: pd.Series,
    rng: np.random.Generator,
    min_start_ts: Optional[pd.Timestamp] = None
) -> Optional[pd.Timestamp]:
    """Return the UTC midnight timestamp of a random 4-consecutive-day block start, optionally constrained to start >= min_start_ts."""
    days = pd.to_datetime(ts, utc=True).dt.floor('D')
    uniq = days.drop_duplicates().sort_values().reset_index(drop=True)
    if min_start_ts is not None:
        min_day = pd.to_datetime(min_start_ts, utc=True).floor('D')
        uniq = uniq[uniq >= min_day]
        uniq = uniq.reset_index(drop=True)
    candidates = []
    for i in range(len(uniq) - 3):
        d0, d1, d2, d3 = uniq.iloc[i:i+4]
        if (d1 - d0 == pd.Timedelta(days=1)
            and d2 - d1 == pd.Timedelta(days=1)
            and d3 - d2 == pd.Timedelta(days=1)):
            candidates.append(d0)
    if not candidates:
        return None
    return rng.choice(candidates)

def inject_structured_attacks(
    df: pd.DataFrame,
    modes: List[str] = ("additive", "deductive"),
    delta_min: float = 5.0,
    delta_max: float = 30.0,
    hours: Optional[List[int]] = None,
    span_hours: int = 2,
    days: int = 4,
    start_day_utc: Optional[pd.Timestamp] = None,
    seed: int = 0,
    only_after_ts: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Inject structured attacks over a 4-day window at the *same hours each day*.

    - additive: can affect anywhere
    - deductive: only subtract where there is usage (>0), and by default we choose hours
                 from early morning or evening.

    A single magnitude (delta in [delta_min, delta_max]) is drawn per attack.
    Returns: attacked_df, labels_series, summary_dict
    """
    rng = np.random.default_rng(seed)
    attacked = df.copy().sort_values("timestamp_utc").reset_index(drop=True)

    # Ensure hour-of-day exists
    if "hour" not in attacked.columns:
        from c2_glr import add_time_features
        attacked = add_time_features(attacked, "timestamp_utc")

    ts = pd.to_datetime(attacked["timestamp_utc"], utc=True)
    # Choose a 4-day consecutive block, honoring only_after_ts if provided
    if start_day_utc is None:
        start_day_utc = _find_consecutive_4day_block(ts, rng, min_start_ts=only_after_ts)
    if start_day_utc is None:
        # Fallback: first 4 unique days at/after cutoff (if provided), else anywhere
        uniq = ts.dt.floor('D').drop_duplicates().sort_values()
        if only_after_ts is not None:
            uniq = uniq[uniq >= pd.to_datetime(only_after_ts, utc=True).floor('D')]
        if len(uniq) >= days:
            start_day_utc = uniq.iloc[0]
        else:
            start_day_utc = ts.dt.floor('D').min()
    end_day_utc = start_day_utc + pd.Timedelta(days=days)
    in_block = (ts >= start_day_utc) & (ts < end_day_utc)
    if only_after_ts is not None:
        in_block = in_block & (ts >= pd.to_datetime(only_after_ts, utc=True))

    hours_all = list(range(24))

    def choose_hours_for(mode: str) -> List[int]:
        # If user provided explicit hours, use them
        if hours is not None and len(hours) > 0:
            return [int(h) % 24 for h in hours]
        # Otherwise pick a contiguous span starting from a base hour
        if mode == "deductive":
            # Early and evening candidates (UTC)
            cand = [5,6,7,8, 18,19,20,21,22]
        else:
            cand = hours_all
        base = int(rng.choice(cand))
        return [ (base + k) % 24 for k in range(max(1, int(span_hours))) ]

    labels = pd.Series(np.zeros(len(attacked), dtype=int), index=attacked.index)
    summaries: List[Dict[str, Any]] = []

    def apply_attack(mode: str):
        hset = set(choose_hours_for(mode))
        hour_mask = attacked["hour"].astype(int).isin(list(hset))
        mask = in_block & hour_mask
        if not mask.any():
            return
        y_col = "consumption_L_per_min"
        attacked[y_col] = attacked[y_col].astype(float).clip(lower=0.0)

        # One magnitude for this attack
        delta = float(rng.uniform(delta_min, delta_max))

        before = attacked.loc[mask, y_col].to_numpy(copy=True)

        if mode == "additive":
            attacked.loc[mask, y_col] = attacked.loc[mask, y_col] + delta
            # label only untouched positions to avoid overwriting if both modes run
            labels.loc[mask] = np.where(labels.loc[mask] == 0, 1, labels.loc[mask])
            affected_mask = mask
        elif mode == "deductive":
            # Only subtract where there is water usage
            use_mask = mask & (attacked[y_col] > 0.0)
            if use_mask.any():
                attacked.loc[use_mask, y_col] = np.maximum(0.0, attacked.loc[use_mask, y_col] - delta)
                labels.loc[use_mask] = np.where(labels.loc[use_mask] == 0, -1, labels.loc[use_mask])
            affected_mask = use_mask
        else:
            return

        after = attacked.loc[mask, y_col].to_numpy(copy=True)

        summaries.append({
            "mode": mode,
            "delta": delta,
            "start_day_utc": pd.to_datetime(start_day_utc),
            "end_day_utc": pd.to_datetime(end_day_utc),
            "hours": sorted(list(hset)),
            "minutes_targeted": int(mask.sum()),
            "minutes_affected": int(affected_mask.sum()),
            "mean_before": float(np.mean(before)) if before.size else float("nan"),
            "mean_after": float(np.mean(after)) if after.size else float("nan"),
        })

    for m in modes:
        if m in ("additive", "deductive"):
            apply_attack(m)

    summary = {
        "window_days": days,
        "start_day_utc": pd.to_datetime(start_day_utc),
        "end_day_utc": pd.to_datetime(end_day_utc),
        "attacks": summaries,
        "only_after_ts": (pd.to_datetime(only_after_ts) if only_after_ts is not None else None),
    }
    return attacked, labels, summary

def inject_attacks(
    df: pd.DataFrame,
    specs: List[AttackSpec],
    seed: int = 0,
    respect_gate: bool = True,
    only_after_ts: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    attacked = df.copy()
    labels = pd.Series(np.zeros(len(attacked), dtype=int), index=attacked.index)

    # Gate logic (or bypass)
    gate = (df.apply(default_gate, axis=1).values if respect_gate else np.ones(len(df), dtype=int))

    # Optional time cutoff: restrict injections to timestamps >= only_after_ts
    allowed = np.ones(len(attacked), dtype=bool)
    if only_after_ts is not None:
        ts = pd.to_datetime(attacked["timestamp_utc"], utc=True)
        cutoff_ts = pd.to_datetime(only_after_ts, utc=True)
        allowed = (ts >= cutoff_ts).values

    for spec in specs:
        i = 0
        N = len(attacked)
        while i < N:
            if allowed[i] and gate[i] == 1 and rng.random() < spec.prob_per_min:
                L = int(rng.integers(spec.min_len, spec.max_len + 1))
                j_end = min(N, i + L)
                if spec.mode == "additive":
                    attacked.loc[attacked.index[i:j_end], "consumption_L_per_min"] += spec.delta_L_per_min
                    labels.iloc[i:j_end] = 1
                elif spec.mode == "deductive":
                    attacked.loc[attacked.index[i:j_end], "consumption_L_per_min"] = np.maximum(
                        0.0, attacked.loc[attacked.index[i:j_end], "consumption_L_per_min"] - spec.delta_L_per_min
                    )
                    labels.iloc[i:j_end] = -1
                i = j_end
            else:
                i += 1

    return attacked, labels
