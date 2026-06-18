"""
analyze_ambient.py

Long-duration ("ambient") voltage-drift analysis for the 10 RADFET dosimeter
sensors (sensor_group R1/R2 x channel 1-5). The question this answers is:

    Over the multi-day ambient log, has the voltage on any sensor MEANINGFULLY
    shifted -- and even if no single sensor moved significantly, did the sensors
    as a group drift consistently in one direction?

Pipeline
--------
  1. Parse the per-sensor measurement uncertainty sigma_V [V] (the characterized
     noise floor) from ../lead_brick_analysis/analysis_report.txt. These are the
     "std dev (sample)" figures, one per sensor.
  2. Load every daily CSV under raw_data/historic_data, clean bad timestamps
     (e.g. the 1969 epoch) and flag out-of-band / saturated voltages. The end of
     the span ("present day") is DETECTED as the latest valid timestamp -- it is
     never hard-coded.
  3. For each sensor, average voltage over a window of length W within the
     "before" period and within the "after" period, giving a before-voltage and
     an after-voltage. The uncertainty on each window mean is sigma_V/sqrt(n)
     (standard error of n readings whose per-reading noise is the lead-brick
     sigma_V).
  4. PER-SENSOR significance: shift = after - before, tested against the sensor's
     own uncertainty:
         sigma_shift = sigma_V * sqrt(1/n_before + 1/n_after)
         z = shift / sigma_shift,  two-sided p = erfc(|z|/sqrt2)
     Report which sensors shifted, in which direction, and to what degree.
  5. COLLECTIVE significance: small per-sensor shifts that all point the SAME way
     are themselves unlikely by chance. Two complementary tests:
         - Sign / exact binomial test on n_up vs n_down (null p=0.5 each).
         - Stouffer's combined z: Z = sum(z_i)/sqrt(N), two-sided p.
  6. WINDOW-LENGTH SENSITIVITY: the averaging window can reasonably range from
     ~30 min to 2 hr+. Every window length in WINDOW_LENGTHS_MIN is trialed and
     the results (per-sensor significance counts + both collective tests) are
     tabulated, so the window choice is surfaced BEFORE a default is committed to.

The before/after comparison periods are CONFIGURABLE parameters (BEFORE_START /
BEFORE_END / AFTER_START / AFTER_END). Leave them None to auto-detect: "before"
anchors at the earliest data, "after" anchors at the most recent data.

Outputs (under output/):
    ambient_analysis_report.txt  -- full text report (analogous to the lead-brick
                                     analysis_report.txt)
    ambient_shift_summary.csv    -- tidy per-sensor table at the default window
    plots/*.png                  -- optional, only if matplotlib is installed

Run:
    python analyze_ambient.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from sigma_v import parse_sigma_v_from_report

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Paths resolved relative to this script so it runs from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data" / "historic_data"
LEAD_BRICK_REPORT = PROJECT_ROOT / "lead_brick_analysis" / "analysis_report.txt"

OUTPUT_DIR = SCRIPT_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "ambient_analysis_report.txt"
SUMMARY_CSV = OUTPUT_DIR / "ambient_shift_summary.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Source column names (same schema as the lead-brick / sample data).
GROUP_COL = "sensor_group"
CHANNEL_COL = "channel"
ADC_COL = "raw_adc"
TIMESTAMP_COL = "timestamp"

# The CSV's own "voltage" column is IGNORED: it was mislabeled (it is really a
# delta_v) and the baseline that was subtracted to produce it changed over time,
# so it is not comparable across days. We instead recompute delta_v ourselves
# from the untouched raw ADC count:
#
#     raw_voltage = raw_adc * (ADC_VREF_V / ADC_FULL_SCALE)
#     delta_v     = raw_voltage - baseline[group]
#
# The per-group baselines below are the agreed reference voltages.
ADC_VREF_V = 5.0
ADC_FULL_SCALE = 4095
DV_BASELINE_BY_GROUP = {"R1": 1.71, "R2": 1.73}
DV_BASELINE_DEFAULT = 1.71

RAW_VOLTAGE_COL = "raw_voltage"   # derived: raw_adc -> volts
VOLTAGE_COL = "delta_v"           # derived: the delta_v we actually analyze

# The 10 sensors we expect: every (group, channel) pairing.
EXPECTED_GROUPS = ["R1", "R2"]
EXPECTED_CHANNELS = [1, 2, 3, 4, 5]
EXPECTED_SENSORS = [(g, c) for g in EXPECTED_GROUPS for c in EXPECTED_CHANNELS]

# Plausibility bounds (a real baseline reading sits ~0.15-0.45 V; >1 V is a
# saturation/glitch). Identical band to the lead-brick stability analysis.
VOLTAGE_VALID_MIN = 0.0
VOLTAGE_VALID_MAX = 1.0
SATURATION_ADC = 4090
TIMESTAMP_VALID_MIN = pd.Timestamp("2000-01-01")
TIMESTAMP_VALID_MAX = pd.Timestamp("2100-01-01")

# Only consider data on/after this date. The first ~3 days (Jun 10-13) are a
# one-time sensor warm-up/settling transition (plus a Jun-12 glitch and the
# ch4 Jun-13 step), so the stable regime begins Jun 14. The END of the span is
# detected from the data, never hard-coded.
SPAN_START = pd.Timestamp("2026-06-14")

# --------------------------------------------------------------------------- #
# Comparison periods (CONFIGURABLE -- set the exact times you want to compare).
#
# Each is an ISO-8601 string or None. None => auto-detect:
#   * BEFORE_START defaults to the earliest valid timestamp in the span,
#   * AFTER_END   defaults to the latest  valid timestamp ("present day").
# BEFORE_END / AFTER_START are optional hard caps on where a window may fall.
#
# The averaging window of length W is placed WITHIN each period, anchored at:
#   BEFORE_ANCHOR = "start" -> the earliest W of the before period,
#   AFTER_ANCHOR  = "end"   -> the most-recent W of the after period,
# which compares the earliest-available data against the most-recent data.
# --------------------------------------------------------------------------- #
BEFORE_START: str | None = "2026-06-14 00:00:00"
BEFORE_END: str | None = None
AFTER_START: str | None = None
AFTER_END: str | None = None      # None => latest valid timestamp ("present day")

BEFORE_ANCHOR = "start"   # "start" or "end"
AFTER_ANCHOR = "end"      # "start" or "end"

# Averaging-window lengths to trial, in minutes. The default (committed) window
# is chosen from this list after the sensitivity comparison is shown. The list
# extends past 3 hr up to multi-day windows so the trend can be seen to peak and
# then fall off (once the before/after windows approach each other the shift
# shrinks; see report note).
WINDOW_LENGTHS_MIN = [30, 60, 120, 180, 240, 360, 480, 720, 1080,
                      1440, 2160, 2880, 4320, 5760]
DEFAULT_WINDOW_MIN = 120

# Coverage factor on every sigma_V (1.0 = raw 1-sigma from the report).
SIGMA_V_COVERAGE = 1.0
SIGMA_V_DEFAULT = 0.005

# Significance thresholds on |z|.
Z_95 = 1.96       # ~2 sigma, 95% two-sided
Z_997 = 3.0       # ~3 sigma, 99.7% two-sided

# Minimum readings required in a window for a usable mean.
MIN_READINGS_PER_WINDOW = 2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sensor_label(group: str, channel: int) -> str:
    """Filesystem-friendly identifier for a sensor, e.g. 'R1_ch3'."""
    return f"{group}_ch{channel}"


def _twosided_p(z: float) -> float:
    """Two-sided p-value for a standard-normal z."""
    if not np.isfinite(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))


def _significance_label(z: float) -> str:
    if not np.isfinite(z):
        return "n/a"
    az = abs(z)
    if az >= Z_997:
        return ">3 sigma"
    if az >= Z_95:
        return ">2 sigma"
    return "n.s."


def _sign_test_p(n_up: int, n_down: int) -> float:
    """
    Two-sided exact binomial (sign) test that n_up vs n_down is consistent with
    a fair 50/50 coin. p = min(1, 2 * P(X >= max(n_up, n_down))) under p=0.5.
    Ties (shift == 0) are excluded by the caller, as is standard for a sign test.
    """
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


# --------------------------------------------------------------------------- #
# Load & quality control
# --------------------------------------------------------------------------- #

def load_data(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and concatenate every daily CSV under raw_dir. Parses timestamps,
    nulls implausible ones, restricts to >= SPAN_START, and adds a boolean
    'valid' column. Returns (dataframe, list of source filenames).
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw-data directory not found: {raw_dir}")

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {raw_dir}")

    frames = []
    for f in files:
        part = pd.read_csv(f)
        part["__source_file"] = f.name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    # Recompute delta_v from the raw ADC count (see config note). The CSV's own
    # "voltage" column is deliberately not used.
    adc = pd.to_numeric(df[ADC_COL], errors="coerce")
    df[RAW_VOLTAGE_COL] = adc * (ADC_VREF_V / ADC_FULL_SCALE)
    baseline = df[GROUP_COL].map(DV_BASELINE_BY_GROUP).fillna(DV_BASELINE_DEFAULT)
    df[VOLTAGE_COL] = df[RAW_VOLTAGE_COL] - baseline

    # Parse timestamps; out-of-window values (e.g. 1969 epoch) -> NaT.
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    out_of_range = df[TIMESTAMP_COL].notna() & (
        (df[TIMESTAMP_COL] < TIMESTAMP_VALID_MIN)
        | (df[TIMESTAMP_COL] > TIMESTAMP_VALID_MAX)
    )
    df.loc[out_of_range, TIMESTAMP_COL] = pd.NaT

    # Restrict to the relevant span (drop anything before SPAN_START). Rows with
    # NaT timestamps are kept here but excluded later by the window selection.
    df = df[(df[TIMESTAMP_COL].isna()) | (df[TIMESTAMP_COL] >= SPAN_START)]
    df = df.reset_index(drop=True)

    df["valid"] = flag_valid(df)
    return df, [f.name for f in files]


def flag_valid(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: finite, in-band voltage, not ADC-saturated."""
    voltage = pd.to_numeric(df[VOLTAGE_COL], errors="coerce")
    adc = pd.to_numeric(df[ADC_COL], errors="coerce")
    return (
        np.isfinite(voltage)
        & (voltage > VOLTAGE_VALID_MIN)
        & (voltage <= VOLTAGE_VALID_MAX)
        & (adc < SATURATION_ADC)
    )


# --------------------------------------------------------------------------- #
# Period / window resolution
# --------------------------------------------------------------------------- #

def _coerce_ts(value, default: pd.Timestamp) -> pd.Timestamp:
    return default if value is None else pd.Timestamp(value)


def resolve_periods(df_valid: pd.DataFrame) -> dict:
    """
    Resolve the before/after period bounds, filling None with auto-detected
    edges of the valid-timestamp span. Returns a dict of pd.Timestamps and the
    detected span edges (for reporting).
    """
    ts = df_valid[TIMESTAMP_COL].dropna()
    span_min, span_max = ts.min(), ts.max()
    return {
        "span_min": span_min,
        "span_max": span_max,
        "before_start": _coerce_ts(BEFORE_START, span_min),
        "before_end": _coerce_ts(BEFORE_END, span_max),
        "after_start": _coerce_ts(AFTER_START, span_min),
        "after_end": _coerce_ts(AFTER_END, span_max),
    }


def window_bounds(period_start, period_end, anchor: str,
                  length_min: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Place a window of `length_min` minutes inside [period_start, period_end].
      anchor == "start": [period_start, min(period_start + W, period_end)]
      anchor == "end"  : [max(period_end - W, period_start), period_end]
    """
    w = pd.Timedelta(minutes=length_min)
    if anchor == "start":
        return period_start, min(period_start + w, period_end)
    return max(period_end - w, period_start), period_end


def window_voltages(sensor_valid: pd.DataFrame,
                    win_start: pd.Timestamp,
                    win_end: pd.Timestamp) -> pd.Series:
    """Valid voltages for one sensor falling within [win_start, win_end]."""
    ts = sensor_valid[TIMESTAMP_COL]
    mask = ts.notna() & (ts >= win_start) & (ts <= win_end)
    return pd.to_numeric(sensor_valid.loc[mask, VOLTAGE_COL], errors="coerce").dropna()


# --------------------------------------------------------------------------- #
# Per-sensor shift for a given window length
# --------------------------------------------------------------------------- #

def sensor_shift(sensor_valid: pd.DataFrame, sigma_v: float, periods: dict,
                 length_min: float) -> dict:
    """
    Compute the before/after means and the significance of their difference for
    a single sensor at one averaging-window length.
    """
    b_start, b_end = window_bounds(
        periods["before_start"], periods["before_end"], BEFORE_ANCHOR, length_min
    )
    a_start, a_end = window_bounds(
        periods["after_start"], periods["after_end"], AFTER_ANCHOR, length_min
    )
    before = window_voltages(sensor_valid, b_start, b_end)
    after = window_voltages(sensor_valid, a_start, a_end)

    n_b, n_a = int(before.count()), int(after.count())
    out = {
        "n_before": n_b, "n_after": n_a,
        "before_window": (b_start, b_end), "after_window": (a_start, a_end),
        "sigma_v": sigma_v,
    }
    if n_b < MIN_READINGS_PER_WINDOW or n_a < MIN_READINGS_PER_WINDOW:
        out.update({"before_mean": float("nan"), "after_mean": float("nan"),
                    "shift": float("nan"), "sigma_shift": float("nan"),
                    "z": float("nan"), "p": float("nan"),
                    "direction": "n/a", "label": "n/a",
                    "before_std": float("nan"), "after_std": float("nan")})
        return out

    before_mean, after_mean = float(before.mean()), float(after.mean())
    shift = after_mean - before_mean
    # Uncertainty on each window mean is sigma_V/sqrt(n); they add in quadrature.
    sigma_shift = sigma_v * math.sqrt(1.0 / n_b + 1.0 / n_a)
    z = shift / sigma_shift if sigma_shift > 0 else float("nan")

    out.update({
        "before_mean": before_mean, "after_mean": after_mean,
        "before_std": float(before.std(ddof=1)) if n_b > 1 else float("nan"),
        "after_std": float(after.std(ddof=1)) if n_a > 1 else float("nan"),
        "shift": shift, "sigma_shift": sigma_shift,
        "z": z, "p": _twosided_p(z),
        "direction": "up" if shift > 0 else ("down" if shift < 0 else "flat"),
        "label": _significance_label(z),
    })
    return out


def analyze_window(df: pd.DataFrame, sigma_v_by_sensor: dict, periods: dict,
                   length_min: float) -> dict:
    """Per-sensor shifts + collective tests at one averaging-window length."""
    valid = df[df["valid"]]
    per_sensor: dict = {}
    for group, channel in EXPECTED_SENSORS:
        mask = (valid[GROUP_COL] == group) & (valid[CHANNEL_COL] == channel)
        sigma_v = sigma_v_by_sensor.get((group, channel), SIGMA_V_DEFAULT) * SIGMA_V_COVERAGE
        per_sensor[(group, channel)] = sensor_shift(
            valid[mask], sigma_v, periods, length_min
        )

    usable = [r for r in per_sensor.values() if np.isfinite(r["z"])]
    n_up = sum(1 for r in usable if r["shift"] > 0)
    n_down = sum(1 for r in usable if r["shift"] < 0)
    n_sig2 = sum(1 for r in usable if abs(r["z"]) >= Z_95)
    n_sig3 = sum(1 for r in usable if abs(r["z"]) >= Z_997)
    z_comb, p_comb = _stouffer([r["z"] for r in usable])

    collective = {
        "n_usable": len(usable),
        "n_up": n_up, "n_down": n_down,
        "n_sig2": n_sig2, "n_sig3": n_sig3,
        "sign_p": _sign_test_p(n_up, n_down),
        "stouffer_z": z_comb, "stouffer_p": p_comb,
        "consensus_direction": "up" if n_up > n_down else ("down" if n_down > n_up else "split"),
    }
    return {"per_sensor": per_sensor, "collective": collective}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt_ts(ts: pd.Timestamp) -> str:
    return "n/a" if pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M:%S")


def build_report(results_by_window: dict, default_min: int, periods: dict,
                 sigma_source: str, qc: dict, source_files: list[str]) -> str:
    lines: list[str] = []
    sep = "=" * 78
    sub = "-" * 78

    lines.append(sep)
    lines.append("RADFET AMBIENT LOG - LONG-DURATION VOLTAGE-SHIFT ANALYSIS")
    lines.append(sep)
    lines.append(f"Raw-data directory : {RAW_DATA_DIR}")
    lines.append(f"Daily files        : {len(source_files)} "
                 f"({source_files[0]} .. {source_files[-1]})")
    lines.append(f"Output directory   : {OUTPUT_DIR}")
    lines.append(f"Sensors            : {len(EXPECTED_SENSORS)} "
                 f"({', '.join(EXPECTED_GROUPS)} x ch{EXPECTED_CHANNELS[0]}-{EXPECTED_CHANNELS[-1]})")
    lines.append(f"Measurement sigma  : per-sensor lead-brick sigma_V "
                 f"(coverage factor {SIGMA_V_COVERAGE:g})")
    lines.append(f"  sigma_V source   : {sigma_source}")
    base_str = ", ".join(f"{g}:{b:g}V" for g, b in DV_BASELINE_BY_GROUP.items())
    lines.append(f"delta_v definition : raw_adc * ({ADC_VREF_V:g}/{ADC_FULL_SCALE}) "
                 f"- baseline ({base_str})")
    lines.append("  (CSV 'voltage' column ignored: mislabeled, inconsistent baseline over time)")
    lines.append(f"Valid delta_v band : ({VOLTAGE_VALID_MIN}, {VOLTAGE_VALID_MAX}] V; "
                 f"saturation raw_adc >= {SATURATION_ADC}")
    lines.append("")
    lines.append(f"Span (detected)    : {_fmt_ts(periods['span_min'])} "
                 f"-> {_fmt_ts(periods['span_max'])}  "
                 f"({(periods['span_max'] - periods['span_min']).total_seconds() / 86400.0:.2f} days)")
    lines.append("  ('after' end is the latest valid timestamp = present day; not hard-coded)")
    lines.append(f"Before period      : {_fmt_ts(periods['before_start'])} "
                 f"-> {_fmt_ts(periods['before_end'])}  (window anchored at {BEFORE_ANCHOR})")
    lines.append(f"After  period      : {_fmt_ts(periods['after_start'])} "
                 f"-> {_fmt_ts(periods['after_end'])}  (window anchored at {AFTER_ANCHOR})")
    lines.append("")

    # --- Data quality ------------------------------------------------------ #
    lines.append(sub)
    lines.append("DATA QUALITY")
    lines.append(sub)
    lines.append(f"Total readings       : {qc['n_total']}")
    lines.append(f"Valid readings       : {qc['n_valid']}")
    lines.append(f"Rejected (band/NaN)  : {qc['n_band']}")
    lines.append(f"Rejected (saturation): {qc['n_sat']}")
    lines.append(f"Bad/missing timestamp: {qc['n_bad_ts']}")
    lines.append("")

    # --- Window-length sensitivity (shown BEFORE committing to a default) -- #
    lines.append(sub)
    lines.append("WINDOW-LENGTH SENSITIVITY (how the result depends on averaging window)")
    lines.append(sub)
    lines.append("  For each window length W, both edges average W minutes of data. "
                 "n_up/n_down")
    lines.append("  is the per-sensor shift direction count; sign_p is the two-sided exact")
    lines.append("  binomial test; Stouffer Z combines the per-sensor z's "
                 "(Z=sum z_i/sqrt N).")
    lines.append("")
    lines.append(f"{'W[min]':>7} {'n_use':>6} {'up':>3} {'dn':>3} "
                 f"{'>2s':>4} {'>3s':>4} {'sign_p':>10} {'Stouffer_Z':>11} "
                 f"{'Stouffer_p':>11} {'consensus':>10}")
    for w in WINDOW_LENGTHS_MIN:
        c = results_by_window[w]["collective"]
        star = " *" if w == default_min else ""
        lines.append(
            f"{w:>7} {c['n_usable']:>6} {c['n_up']:>3} {c['n_down']:>3} "
            f"{c['n_sig2']:>4} {c['n_sig3']:>4} {c['sign_p']:>10.2e} "
            f"{c['stouffer_z']:>11.2f} {c['stouffer_p']:>11.2e} "
            f"{c['consensus_direction']:>10}{star}"
        )
    lines.append("")
    lines.append(f"  ('*' marks the committed default window = {default_min} min. The collective")
    lines.append("   direction and Stouffer significance should be checked for stability across")
    lines.append("   W; per-sensor z grows ~sqrt(n) with W, so larger windows sharpen weak shifts.)")
    lines.append("")

    # --- Per-sensor detail at the default window --------------------------- #
    default = results_by_window[default_min]
    lines.append(sub)
    lines.append(f"PER-SENSOR SHIFT  (default window = {default_min} min)")
    lines.append(sub)
    lines.append(f"{'sensor':>8} {'n_b':>4} {'n_a':>4} {'before[V]':>11} "
                 f"{'after[V]':>11} {'shift[V]':>11} {'sigV[V]':>9} "
                 f"{'z':>7} {'p':>10} {'dir':>5} {'verdict':>9}")
    for group, channel in EXPECTED_SENSORS:
        r = default["per_sensor"][(group, channel)]
        label = sensor_label(group, channel)
        if not np.isfinite(r["z"]):
            lines.append(f"{label:>8} {r['n_before']:>4} {r['n_after']:>4}  "
                         f"(insufficient data in window)")
            continue
        lines.append(
            f"{label:>8} {r['n_before']:>4} {r['n_after']:>4} "
            f"{r['before_mean']:>11.6f} {r['after_mean']:>11.6f} "
            f"{r['shift']:>+11.6f} {r['sigma_v']:>9.6f} "
            f"{r['z']:>+7.2f} {r['p']:>10.2e} {r['direction']:>5} {r['label']:>9}"
        )
    lines.append("")
    lines.append("  shift = after - before (sign = direction). z = shift / sigma_shift,")
    lines.append("  sigma_shift = sigma_V * sqrt(1/n_before + 1/n_after).")
    lines.append("")

    # --- Collective significance ------------------------------------------- #
    c = default["collective"]
    lines.append(sub)
    lines.append(f"COLLECTIVE SHIFT  (default window = {default_min} min)")
    lines.append(sub)
    lines.append(f"Sensors usable               : {c['n_usable']}")
    lines.append(f"Moved up / down              : {c['n_up']} up, {c['n_down']} down "
                 f"(consensus: {c['consensus_direction']})")
    lines.append(f"Individually significant     : {c['n_sig2']} at >2 sigma, "
                 f"{c['n_sig3']} at >3 sigma")
    lines.append("")
    lines.append("Sign / binomial test (are up vs down counts a fair coin?)")
    lines.append(f"  two-sided p                : {c['sign_p']:.3e}  "
                 f"({_significance_label(_p_to_z(c['sign_p']))})")
    lines.append("Stouffer combined z (consistent small shifts add up)")
    lines.append(f"  Z = sum(z_i)/sqrt(N)       : {c['stouffer_z']:+.2f}")
    lines.append(f"  two-sided p                : {c['stouffer_p']:.3e}  "
                 f"({_significance_label(c['stouffer_z'])})")
    lines.append("")
    lines.append("  Interpretation: even when individual sensors are 'n.s.', a consistent")
    lines.append("  direction across all sensors (low sign_p) and/or a large Stouffer Z is")
    lines.append("  strong evidence of a real, systematic drift rather than noise.")
    lines.append("")

    lines.append(sep)
    lines.append("End of report.")
    lines.append(sep)
    return "\n".join(lines)


def _p_to_z(p: float) -> float:
    """Rough inverse of the two-sided normal p (for labelling the sign test)."""
    if not np.isfinite(p) or p <= 0:
        return float("inf") if (np.isfinite(p) and p <= 0) else float("nan")
    if p >= 1:
        return 0.0
    # Bisection on erfc(z/sqrt2) = p.
    lo, hi = 0.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid / math.sqrt(2.0)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Optional plots (skipped silently if matplotlib is unavailable)
# --------------------------------------------------------------------------- #

def write_plots(results_by_window: dict, default_min: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Per-sensor before vs after with 1-sigma_shift error bars (default W).
    default = results_by_window[default_min]
    labels, befores, afters, errs = [], [], [], []
    for group, channel in EXPECTED_SENSORS:
        r = default["per_sensor"][(group, channel)]
        if not np.isfinite(r["z"]):
            continue
        labels.append(sensor_label(group, channel))
        befores.append(r["before_mean"])
        afters.append(r["after_mean"])
        errs.append(r["sigma_shift"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x - 0.1, befores, yerr=errs, fmt="o", label="before", capsize=3)
    ax.errorbar(x + 0.1, afters, yerr=errs, fmt="s", label="after", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("voltage [V]")
    ax.set_title(f"Per-sensor before vs after (window = {default_min} min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "before_after_by_sensor.png", dpi=120)
    plt.close(fig)

    # 2) Stouffer Z vs window length.
    ws = WINDOW_LENGTHS_MIN
    zc = [results_by_window[w]["collective"]["stouffer_z"] for w in ws]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, zc, "o-")
    ax.axhline(Z_95, ls="--", color="gray", label="2 sigma")
    ax.axhline(-Z_95, ls="--", color="gray")
    ax.set_xlabel("averaging window [min]")
    ax.set_ylabel("Stouffer combined Z")
    ax.set_title("Collective shift significance vs window length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stouffer_vs_window.png", dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: per-sensor sigma_V from the lead-brick report.
    sigma_v_by_sensor, sigma_source = parse_sigma_v_from_report(LEAD_BRICK_REPORT)

    # Step 2: load all daily CSVs + QC.
    df, source_files = load_data(RAW_DATA_DIR)
    voltage = pd.to_numeric(df[VOLTAGE_COL], errors="coerce")
    adc = pd.to_numeric(df[ADC_COL], errors="coerce")
    qc = {
        "n_total": len(df),
        "n_valid": int(df["valid"].sum()),
        "n_band": int((~(np.isfinite(voltage)
                         & (voltage > VOLTAGE_VALID_MIN)
                         & (voltage <= VOLTAGE_VALID_MAX))).sum()),
        "n_sat": int((adc >= SATURATION_ADC).sum()),
        "n_bad_ts": int(df[TIMESTAMP_COL].isna().sum()),
    }

    # Step 3: resolve periods (auto-detecting the present-day edge).
    periods = resolve_periods(df[df["valid"]])

    # Steps 4-6: analyze every trial window length.
    results_by_window = {
        w: analyze_window(df, sigma_v_by_sensor, periods, w)
        for w in WINDOW_LENGTHS_MIN
    }

    default_min = DEFAULT_WINDOW_MIN
    if default_min not in results_by_window:
        default_min = WINDOW_LENGTHS_MIN[len(WINDOW_LENGTHS_MIN) // 2]

    # Report.
    report = build_report(
        results_by_window, default_min, periods, sigma_source, qc, source_files
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    # Tidy per-sensor summary CSV at the default window.
    default = results_by_window[default_min]
    rows = []
    for group, channel in EXPECTED_SENSORS:
        r = default["per_sensor"][(group, channel)]
        finite = np.isfinite(r["z"])
        rows.append({
            "sensor": sensor_label(group, channel),
            "sensor_group": group, "channel": channel,
            "window_min": default_min,
            "n_before": r["n_before"], "n_after": r["n_after"],
            "before_v": round(r["before_mean"], 6) if finite else "",
            "after_v": round(r["after_mean"], 6) if finite else "",
            "shift_v": round(r["shift"], 6) if finite else "",
            "sigma_v": round(r["sigma_v"], 6),
            "sigma_shift_v": round(r["sigma_shift"], 6) if finite else "",
            "z": round(r["z"], 3) if finite else "",
            "p": f"{r['p']:.3e}" if finite else "",
            "direction": r["direction"],
            "significance": r["label"],
        })
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    wrote_plots = write_plots(results_by_window, default_min)

    print(report)
    print(f"\nWrote report  -> {REPORT_PATH}")
    print(f"Wrote summary -> {SUMMARY_CSV}")
    if wrote_plots:
        print(f"Wrote plots   -> {PLOTS_DIR}")
    else:
        print("Plots skipped (matplotlib not installed).")


if __name__ == "__main__":
    main()
