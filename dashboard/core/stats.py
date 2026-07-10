"""
stats.py

Time-window hypothesis tests and trend fits for the Statistical Tests page.

_sign_test_p and _stouffer are ported verbatim from
ambient_analysis/analyze_ambient.py (private helpers of a script whose
module-level config is ambient-specific, so importing the module would drag in
irrelevant assumptions). _twosided_p / _significance_label are imported from
analyze_sample, which is a clean library-style import.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config

from analyze_sample import _significance_label, _twosided_p
from calibration import measured_sigma_v

MIN_READINGS_PER_WINDOW = 3


# ported from ambient_analysis/analyze_ambient.py ---------------------------- #

def _sign_test_p(n_up: int, n_down: int) -> float:
    """Two-sided exact binomial (sign) test that n_up vs n_down is consistent
    with a fair 50/50 coin. Ties are excluded by the caller."""
    n = n_up + n_down
    if n == 0:
        return float("nan")
    k = max(n_up, n_down)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _stouffer(z_values: list[float]) -> tuple[float, float]:
    """Stouffer's combined z (Z = sum z_i / sqrt(N)) and its two-sided p."""
    zs = [z for z in z_values if np.isfinite(z)]
    if not zs:
        return float("nan"), float("nan")
    z_comb = sum(zs) / math.sqrt(len(zs))
    return z_comb, _twosided_p(z_comb)


# ---------------------------------------------------------------------------- #

def shift_analysis(valid_df: pd.DataFrame,
                   before: tuple[pd.Timestamp, pd.Timestamp],
                   after: tuple[pd.Timestamp, pd.Timestamp],
                   sigma_coverage: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Per-sensor before/after delta_v shift with z = shift / (sigma_V *
    sqrt(1/n_b + 1/n_a)), plus the collective sign test and Stouffer Z
    (the analyze_ambient methodology on arbitrary user-chosen windows)."""
    rows = []
    for group, ch in config.ALL_SENSORS:
        sub = valid_df[(valid_df["sensor_group"] == group) & (valid_df["channel"] == ch)]
        b = sub[(sub["timestamp"] >= before[0]) & (sub["timestamp"] <= before[1])]["delta_v"]
        a = sub[(sub["timestamp"] >= after[0]) & (sub["timestamp"] <= after[1])]["delta_v"]
        n_b, n_a = len(b), len(a)
        sigma_v = measured_sigma_v(group, ch) * sigma_coverage
        row = {"sensor": config.sensor_id(group, ch), "sensor_group": group,
               "channel": ch, "n_before": n_b, "n_after": n_a, "sigma_v": sigma_v}
        if n_b < MIN_READINGS_PER_WINDOW or n_a < MIN_READINGS_PER_WINDOW:
            row.update({"before_v": np.nan, "after_v": np.nan, "shift_v": np.nan,
                        "sigma_shift_v": np.nan, "z": np.nan, "p": np.nan,
                        "direction": "n/a", "significance": "n/a"})
        else:
            bm, am = float(b.mean()), float(a.mean())
            shift = am - bm
            sigma_shift = sigma_v * math.sqrt(1.0 / n_b + 1.0 / n_a)
            z = shift / sigma_shift if sigma_shift > 0 else float("nan")
            row.update({"before_v": bm, "after_v": am, "shift_v": shift,
                        "sigma_shift_v": sigma_shift, "z": z, "p": _twosided_p(z),
                        "direction": "up" if shift > 0 else ("down" if shift < 0 else "flat"),
                        "significance": _significance_label(z)})
        rows.append(row)

    table = pd.DataFrame(rows)
    usable = table[np.isfinite(table["z"])]
    n_up = int((usable["shift_v"] > 0).sum())
    n_down = int((usable["shift_v"] < 0).sum())
    z_comb, p_comb = _stouffer(usable["z"].tolist())
    collective = {
        "n_usable": len(usable), "n_up": n_up, "n_down": n_down,
        "n_sig2": int((usable["z"].abs() >= config.Z_95).sum()),
        "n_sig3": int((usable["z"].abs() >= config.Z_997).sum()),
        "sign_p": _sign_test_p(n_up, n_down),
        "stouffer_z": z_comb, "stouffer_p": p_comb,
    }
    return table, collective


def wls_trend(t_sec: np.ndarray, dose: np.ndarray, sigma: np.ndarray) -> dict:
    """Weighted least-squares line dose = a + b*t on a binned cumulative-dose
    series (weights 1/sigma_meas^2, closed form - no scipy). Returns the slope
    in Rad/day with its 1-sigma, and chi^2/dof as a linearity diagnostic:
    chi^2/dof >> 1 means the rate is NOT constant (SAA banding, SPE)."""
    ok = np.isfinite(dose) & np.isfinite(sigma) & (sigma > 0)
    t, y, s = t_sec[ok], dose[ok], sigma[ok]
    if len(t) < 3:
        return {"n": int(len(t)), "slope_rad_day": np.nan, "slope_sigma_rad_day": np.nan,
                "intercept_rad": np.nan, "chi2_dof": np.nan}
    w = 1.0 / s**2
    sw, swx, swy = w.sum(), (w * t).sum(), (w * y).sum()
    swxx, swxy = (w * t * t).sum(), (w * t * y).sum()
    delta = sw * swxx - swx**2
    b = (sw * swxy - swx * swy) / delta          # Rad/s
    a = (swy * swxx - swx * swxy) / delta
    sigma_b = math.sqrt(sw / delta)
    resid = y - (a + b * t)
    dof = len(t) - 2
    chi2 = float((w * resid**2).sum())
    return {"n": int(len(t)), "slope_rad_day": b * 86400.0,
            "slope_sigma_rad_day": sigma_b * 86400.0,
            "intercept_rad": a, "chi2_dof": chi2 / dof if dof > 0 else np.nan}


def periodogram(series: pd.DataFrame, value_col: str = "rate_rad_h") -> pd.DataFrame:
    """Welch-averaged power spectrum of a binned series (Hann window, 50%
    overlap, numpy only). Segment averaging suppresses the white measurement
    noise so coherent periodicities (the ~92.9 min orbital comb, the ~24 h SAA
    cluster cycle) stand out. Gaps are linearly interpolated onto the median
    cadence. Returns period [minutes] vs power normalized to the median."""
    from .pipeline import epoch_seconds

    s = series.dropna(subset=[value_col]).sort_values("timestamp")
    if len(s) < 64:
        return pd.DataFrame(columns=["period_min", "power"])
    t = epoch_seconds(s["timestamp"])
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    y = np.interp(grid, t, s[value_col].to_numpy())

    seg_len = min(len(grid) // 2, max(64, int(4 * 86400 / dt)))
    step = seg_len // 2
    win = np.hanning(seg_len)
    acc = np.zeros(seg_len // 2 + 1)
    n_seg = 0
    for i in range(0, len(y) - seg_len + 1, step):
        seg = y[i:i + seg_len]
        acc += np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2
        n_seg += 1
    if n_seg == 0:
        return pd.DataFrame(columns=["period_min", "power"])
    freq = np.fft.rfftfreq(seg_len, d=dt)
    keep = freq > 0
    power = acc[keep] / n_seg
    out = pd.DataFrame({"period_min": 1.0 / freq[keep] / 60.0,
                        "power": power / np.median(power)})
    return out[out["period_min"] <= 48 * 60]  # up to 2 days
