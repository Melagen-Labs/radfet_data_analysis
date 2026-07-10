"""
pipeline.py

Time-series dose computations on top of the canonical delta_v:

  * binned_series      - resample delta_v per sensor (median), convert to dose
  * combined_series    - inverse-variance R1+R2 combination per time bin
  * add_dose_rate      - dose-rate from the binned cumulative dose
  * end_of_window_summary - adapter feeding the imported analyze_sample stats
                            (combine_trials, z-tests, Monte Carlo CIs) with an
                            arbitrary time slice

Dose conversion per bin uses a VECTORIZED equivalent of
calibration.select_curve_and_dose + calibration.dose_uncertainty (verified to
match the scalar originals; see the cross-check in the repo history). Curve
coefficients are read from calibration.CALIBRATION_CURVES - never re-derived.

Noise control: per-reading dose at low dVt is dominated by the ~5 mV sensor
noise floor, so we smooth in VOLTAGE space first (median per bin) and convert
once per bin. The bin sigma is sigma_V * 1.2533 / sqrt(n): 1.2533 = sqrt(pi/2),
the efficiency penalty of the median vs the mean under Gaussian noise.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from . import config

from calibration import (
    CALIBRATION_CURVES,
    dose_uncertainty,
    measured_sigma_v,
    select_curve_and_dose,
)
from analyze_sample import combine_trials

MEDIAN_EFFICIENCY = 1.2533  # sqrt(pi/2)


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Datetime series -> seconds since epoch, independent of the underlying
    datetime64 resolution (pandas 3.0 defaults to microseconds, so
    .astype('int64') / 1e9 silently gives a 1000x error)."""
    return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()

_CURVE_MAX = np.array([c.dose_max_rad for c in CALIBRATION_CURVES])
_CURVE_A = np.array([c.A for c in CALIBRATION_CURVES])
_CURVE_B = np.array([c.B for c in CALIBRATION_CURVES])
_CURVE_SA = np.array([c.sigma_A for c in CALIBRATION_CURVES])
_CURVE_SB = np.array([c.sigma_B for c in CALIBRATION_CURVES])
_CURVE_NAMES = np.array([c.name for c in CALIBRATION_CURVES])


def vector_dose(dvt: np.ndarray, sigma_dvt: np.ndarray) -> pd.DataFrame:
    """Vectorized select_curve_and_dose + dose_uncertainty.

    Walks the Varadis fits narrowest->widest and keeps the first whose own dose
    estimate lies inside its validity range (identical logic to the scalar
    original). Returns dose, full sigma, measurement-only sigma, curve name,
    extrapolation flag. Non-positive / non-finite dvt -> NaN dose.
    """
    dvt = np.asarray(dvt, dtype=float)
    sigma_dvt = np.asarray(sigma_dvt, dtype=float)
    n = dvt.shape[0]

    dose = np.full(n, np.nan)
    curve_idx = np.full(n, -1)
    ok = np.isfinite(dvt) & (dvt > 0)
    remaining = ok.copy()
    safe = np.where(ok, dvt, 1.0)

    for i in range(len(CALIBRATION_CURVES)):
        cand = (safe / _CURVE_A[i]) ** (1.0 / _CURVE_B[i])
        take = remaining & (cand <= _CURVE_MAX[i])
        dose[take] = cand[take]
        curve_idx[take] = i
        remaining &= ~take

    # beyond the widest fit -> extrapolate with it, flagged
    i = len(CALIBRATION_CURVES) - 1
    cand = (safe / _CURVE_A[i]) ** (1.0 / _CURVE_B[i])
    dose[remaining] = cand[remaining]
    curve_idx[remaining] = i
    extrapolated = remaining & ok

    # dose_uncertainty(), vectorized over the selected curve's coefficients
    sel = np.clip(curve_idx, 0, None)
    A, B = _CURVE_A[sel], _CURVE_B[sel]
    sA, sB = _CURVE_SA[sel], _CURVE_SB[sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        logd = np.log(np.where(dose > 0, dose, np.nan))
        term_meas = (sigma_dvt / (B * dvt)) ** 2
        term_a = (sA / (B * A)) ** 2
        term_b = (logd * sB / B) ** 2
        sigma_full = np.sqrt(term_meas + term_a + term_b) * dose
        sigma_meas = np.sqrt(term_meas) * dose

    names = np.where(curve_idx >= 0, _CURVE_NAMES[sel], "n/a")
    return pd.DataFrame({
        "dose_rad": dose,
        "dose_sigma_full": sigma_full,
        "dose_sigma_meas": sigma_meas,
        "curve_name": names,
        "extrapolated": extrapolated,
    })


@st.cache_data(show_spinner="Binning and converting to dose...")
def binned_series(valid_df: pd.DataFrame, freq: str,
                  sigma_coverage: float = 1.0) -> pd.DataFrame:
    """Per-sensor time bins: median delta_v, bin sigma, dose (+/- sigma)."""
    g = valid_df.groupby(
        ["sensor_group", "channel", pd.Grouper(key="timestamp", freq=freq)],
        observed=True,
    )["delta_v"]
    out = g.agg(dvt_med="median", n="count").reset_index()
    out = out[out["n"] > 0].copy()

    sigma_v = np.array([
        measured_sigma_v(grp, ch) * sigma_coverage
        for grp, ch in zip(out["sensor_group"], out["channel"])
    ])
    out["sigma_bin"] = sigma_v * MEDIAN_EFFICIENCY / np.sqrt(out["n"].to_numpy())

    conv = vector_dose(out["dvt_med"].to_numpy(), out["sigma_bin"].to_numpy())
    out = pd.concat([out.reset_index(drop=True), conv], axis=1)
    return out.sort_values(["sensor_group", "channel", "timestamp"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def combined_series(binned: pd.DataFrame) -> pd.DataFrame:
    """R1+R2 per (channel, bin): inverse-variance mean in voltage space, then
    convert - the time-resolved analogue of analyze_sample.combine_trials.
    Bins present in only one group degrade gracefully to that group."""
    b = binned.copy()
    b["w"] = 1.0 / b["sigma_bin"] ** 2
    b["wx"] = b["w"] * b["dvt_med"]
    agg = (
        b.groupby(["channel", "timestamp"], observed=True)
        .agg(w_sum=("w", "sum"), wx_sum=("wx", "sum"),
             n=("n", "sum"), n_groups=("sensor_group", "nunique"))
        .reset_index()
    )
    agg["dvt_med"] = agg["wx_sum"] / agg["w_sum"]
    agg["sigma_bin"] = np.sqrt(1.0 / agg["w_sum"])
    conv = vector_dose(agg["dvt_med"].to_numpy(), agg["sigma_bin"].to_numpy())
    out = pd.concat(
        [agg[["channel", "timestamp", "dvt_med", "sigma_bin", "n", "n_groups"]], conv],
        axis=1,
    )
    out["sensor_group"] = "R1+R2"
    return out.sort_values(["channel", "timestamp"]).reset_index(drop=True)


def add_dose_rate(series: pd.DataFrame) -> pd.DataFrame:
    """Dose-rate [Rad/h] via np.gradient on each sensor's binned dose series.

    Rate uncertainty uses the MEASUREMENT-ONLY dose sigma: calibration error is
    common-mode across neighbouring bins and cancels in the difference (same
    rationale as the repo's voltage-space significance tests)."""
    parts = []
    for _, sub in series.groupby(["sensor_group", "channel"], observed=True):
        sub = sub.sort_values("timestamp").copy()
        good = sub["dose_rad"].notna()
        sub["rate_rad_h"] = np.nan
        sub["rate_sigma_rad_h"] = np.nan
        if good.sum() >= 3:
            s = sub[good]
            t = epoch_seconds(s["timestamp"])
            dose = s["dose_rad"].to_numpy()
            sig = s["dose_sigma_meas"].to_numpy()
            rate = np.gradient(dose, t) * 3600.0
            sig_rate = np.full_like(rate, np.nan)
            if len(t) > 2:
                dt = t[2:] - t[:-2]
                sig_rate[1:-1] = np.sqrt(sig[2:] ** 2 + sig[:-2] ** 2) / dt * 3600.0
            sub.loc[good, "rate_rad_h"] = rate
            sub.loc[good, "rate_sigma_rad_h"] = sig_rate
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def downsample_for_plot(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Stride-thin a single trace's frame so Plotly never gets >max_points."""
    if len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride]


@st.cache_data(show_spinner="Aggregating window...")
def end_of_window_summary(valid_df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp,
                          sigma_coverage: float = 1.0) -> tuple[dict, dict]:
    """Aggregate an arbitrary time slice exactly like analyze_sample.process_trial
    (mean dVt over valid readings; sigma = lead-brick sigma_V), then combine
    R1/R2 with the imported combine_trials.

    Returns (trial_results, per_channel) shaped for analyze_sample's
    significance_vs_reference / pairwise_z_matrix / monte_carlo_shielding_ci.
    """
    window = valid_df[(valid_df["timestamp"] >= t0) & (valid_df["timestamp"] <= t1)]
    trial_results: dict = {}
    for group in config.TRIAL_GROUPS:
        for ch in config.EXPECTED_CHANNELS:
            sub = window[(window["sensor_group"] == group) & (window["channel"] == ch)]
            if sub.empty:
                continue
            dvt_mean = float(sub["delta_v"].mean())
            sigma_v = measured_sigma_v(group, ch) * sigma_coverage
            curve, dose, extrapolated = select_curve_and_dose(dvt_mean)
            dose_sigma = dose_uncertainty(curve, dvt_mean, sigma_v) if curve else math.nan
            trial_results[(group, ch)] = {
                "group": group, "channel": ch,
                "n_total": len(sub), "n_valid": len(sub),
                "dvt_mean": dvt_mean,
                "dvt_std": float(sub["delta_v"].std(ddof=1)) if len(sub) > 1 else math.nan,
                "sigma_v": sigma_v,
                "curve": curve.name if curve else "n/a",
                "dose_rad": dose, "dose_sigma_rad": dose_sigma,
                "extrapolated": extrapolated,
            }

    per_channel = {
        ch: combine_trials([trial_results[(g, ch)] for g in config.TRIAL_GROUPS
                            if (g, ch) in trial_results])
        for ch in config.EXPECTED_CHANNELS
    }
    return trial_results, per_channel
