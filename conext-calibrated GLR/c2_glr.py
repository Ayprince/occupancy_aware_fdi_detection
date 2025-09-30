
"""
c2_glr.py
---------
Context-Calibrated GLR (C²-GLR) with Online Conformal Calibration and Change-Point–Aware Resets.

Implements a two-track detector:
  Track A: Contextual forecaster -> residuals -> online conformal p-values (state-stratified)
  Track B: State-gated GLR-CUSUM on residuals for additive (+) and deductive (–) pulses
Plus a lightweight residual CUSUM CPD to reset calibration after regime shifts (e.g., travel).

Dependencies: numpy, pandas, scikit-learn
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
    Default high-usage context gate: occupied AND (evening hour OR any appliance running)
    """
    occupied = int(row.get("occupied", 0))
    hour = int(row.get("hour", 0))
    appliance_on = int(row.get("dishwasher_running", 0)) or int(row.get("washing_machine_running", 0))
    is_evening = int(18 <= hour < 23)
    return int(occupied == 1 and (is_evening or appliance_on))


def stratum_id(row: pd.Series) -> Tuple[int, int, int]:
    """
    Define a context stratum for conformal calibration.
    (occupied, hour_bucket(=hour), appliance_any)
    """
    occ = int(row.get("occupied", 0))
    hour = int(row.get("hour", 0))
    app = int(row.get("dishwasher_running", 0)) or int(row.get("washing_machine_running", 0))
    return (occ, hour, int(app))


# -----------------------------
# Quantile Forecaster
# -----------------------------

@dataclass
class QuantileForecaster:
    """
    Fast quantile forecaster using scikit-learn's GradientBoostingRegressor.
    Trains three models for q10, q50, q90 to derive a robust spread estimate.
    """
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    max_depth: int = 3
    n_estimators: int = 150
    learning_rate: float = 0.05
    random_state: int = 0

    models: Dict[float, GradientBoostingRegressor] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X.values)
        for q in self.quantiles:
            m = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            m.fit(Xs, y.values.astype(float))
            self.models[q] = m
        return self

    def predict_quants(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.scaler is not None and len(self.models) == 3, "Model not fitted."
        Xs = self.scaler.transform(X.values)
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

from collections import deque, defaultdict

@dataclass
class ConformalCalibrator:
    """
    Maintains sliding windows of nonconformity scores per context stratum
    and emits conformal p-values.
    """
    window_per_stratum: int = 2000

    buffers: Dict[Tuple[int,int,int], deque] = field(default_factory=lambda: defaultdict(deque))

    def reset_all(self):
        self.buffers = defaultdict(deque)

    def update_and_pvalue(self, stratum: Tuple[int,int,int], score: float) -> float:
        buf = self.buffers[stratum]
        # compute p-value BEFORE inserting score (leave-one-out style)
        if len(buf) == 0:
            p = 1.0  # no evidence yet
        else:
            ge = sum(1 for s in buf if s >= score)
            p = (1 + ge) / (1 + len(buf))
        # push new score
        buf.append(float(score))
        if len(buf) > self.window_per_stratum:
            buf.popleft()
        self.buffers[stratum] = buf
        return float(p)


# -----------------------------
# Residual CPD (CUSUM variant)
# -----------------------------

@dataclass
class ResidualCUSUM:
    """
    Lightweight mean-shift CUSUM on residuals to detect regime changes.
    On detection, caller should reset conformal buffers.
    """
    k: float = 0.0           # reference mean (0 on residuals)
    h: float = 30.0          # threshold
    drift: float = 0.0       # small drift to avoid false resets
    sum_pos: float = 0.0
    sum_neg: float = 0.0

    def update(self, r: float) -> bool:
        x = r - self.k - self.drift
        self.sum_pos = max(0.0, self.sum_pos + x)
        self.sum_neg = min(0.0, self.sum_neg + x)
        if self.sum_pos > self.h or abs(self.sum_neg) > self.h:
            # reset after detection
            self.sum_pos = 0.0
            self.sum_neg = 0.0
            return True
        return False


# -----------------------------
# State-Gated GLR-CUSUM
# -----------------------------

@dataclass
class GLRCUSUM:
    """
    Two-sided GLR-CUSUM on residuals r_t with known/estimated sigma.
    Updates only when gate==1.
    """
    eta_pos: float = 50.0
    eta_neg: float = 50.0
    lam: float = 0.05               # EWMA for mean estimate under H1
    sigma_floor: float = 1e-3

    S_pos: float = 0.0
    S_neg: float = 0.0
    mu_pos: float = 0.0
    mu_neg: float = 0.0

    def update(self, r: float, sigma: float, gate: int) -> Tuple[bool, bool]:
        if not gate:
            # decay stats slowly when gate is off
            self.S_pos = max(0.0, self.S_pos * 0.9)
            self.S_neg = max(0.0, self.S_neg * 0.9)
            return (False, False)

        s2 = max(sigma**2, self.sigma_floor**2)

        # Positive shift branch
        self.mu_pos = (1 - self.lam) * self.mu_pos + self.lam * r
        ll_pos = 0.5 * ((r**2 - (r - self.mu_pos)**2) / s2)
        self.S_pos = max(0.0, self.S_pos + ll_pos)
        alarm_pos = self.S_pos > self.eta_pos
        if alarm_pos:
            self.S_pos = 0.0  # reset after alarm

        # Negative shift branch
        self.mu_neg = (1 - self.lam) * self.mu_neg + self.lam * r
        # For negative shift, use mirrored statistic by flipping sign
        r_neg = -r
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
    # training/calibration
    train_days: int = 14
    conf_window: int = 2000
    alpha: float = 0.005
    debounce_K: int = 3
    debounce_M: int = 10

    # GLR/CPD
    eta_pos: float = 60.0
    eta_neg: float = 60.0
    lam: float = 0.05
    cpd_h: float = 40.0

    # feature windows
    roll_windows: Tuple[int,...] = (5, 15, 60)
    lags: Tuple[int,...] = (1, 2, 3, 5, 10)


def make_features(df: pd.DataFrame, y_col: str = "consumption_L_per_min") -> Tuple[pd.DataFrame, List[str]]:
    df = add_time_features(df, "timestamp_utc")
    df = safe_rolling_features(df, y_col, windows=(5,15,60))
    df = add_lags(df, y_col, lags=(1,2,3,5,10))

    # Basic numeric + binary features
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
    # Ensure required cols
    assert "consumption_L_per_min" in df.columns
    for b in ["occupied", "has_activity", "dishwasher_running", "washing_machine_running"]:
        if b not in df.columns:
            df[b] = 0

    # Build features
    X, feat_names = make_features(df, "consumption_L_per_min")
    y = df["consumption_L_per_min"].astype(float).values

    # Train/Online split
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    t0 = ts.min()
    cutoff = t0 + pd.Timedelta(days=params.train_days)
    train_idx = ts < cutoff
    test_idx = ~train_idx

    # Forecaster
    f = QuantileForecaster()
    f.fit(X.loc[train_idx], pd.Series(y[train_idx]))

    # Calibrator / CPD / GLR
    cal = ConformalCalibrator(window_per_stratum=params.conf_window)
    cpd = ResidualCUSUM(h=params.cpd_h)
    glr = GLRCUSUM(eta_pos=params.eta_pos, eta_neg=params.eta_neg, lam=params.lam)

    # Storage
    out = []
    from collections import deque
    deb_add_q = deque(maxlen=params.debounce_M)
    deb_ded_q = deque(maxlen=params.debounce_M)

    # Iterate
    yhat_all, sigma_all = f.predict_point_and_sigma(X)
    resid_all = y - yhat_all

    for i in range(len(df)):
        row = df.iloc[i]
        y_t = float(y[i])
        yhat_t = float(yhat_all[i])
        sigma_t = float(sigma_all[i])
        r_t = float(resid_all[i])

        # CPD to reset calibration
        cpd_trig = cpd.update(r_t)
        if cpd_trig:
            cal.reset_all()

        # Conformal score & p-value
        s_id = stratum_id(row)
        score = abs(r_t) / max(sigma_t, 1e-6)
        pval = cal.update_and_pvalue(s_id, score)

        # Gate and GLR-CUSUM
        gate = default_gate(row)
        a_pos, a_neg = glr.update(r_t, sigma_t, gate)

        # Debounce final alarms (AND rule: p small and GLR fires)
        add_hit = (pval < params.alpha) and a_pos
        ded_hit = (pval < params.alpha) and a_neg

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
            "p_value": pval,
            "gate": int(gate),
            "S_pos": glr.S_pos,
            "S_neg": glr.S_neg,
            "alarm_additive": alarm_add,
            "alarm_deductive": alarm_ded,
            "cpd_reset": int(cpd_trig),
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
    prob_per_min: float = 0.005   # attempt start probability within gate-on windows


def inject_attacks(df: pd.DataFrame, specs: List[AttackSpec], seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return (df_attacked, labels), where labels in {0=benign, 1=additive, -1=deductive}
    Attacks only occur in gate-on windows.
    """
    rng = np.random.default_rng(seed)
    attacked = df.copy()
    labels = pd.Series(np.zeros(len(attacked), dtype=int), index=attacked.index)
    gate = attacked.apply(default_gate, axis=1).values

    for spec in specs:
        i = 0
        N = len(attacked)
        while i < N:
            if gate[i] == 1 and rng.random() < spec.prob_per_min:
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
