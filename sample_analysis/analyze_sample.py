"""
analyze_sample.py

End-to-end demonstration of the RADFET dose-analysis pipeline on synthetic
data (fake_iss_data.csv). The same code runs unchanged on real ISS data in the
LeadEnclosureData.csv format -- only the INPUT_CSV path changes.

Pipeline
--------
  1. Load the CSV and clean obviously-bad timestamps (e.g. 1969 epoch).
  2. Flag data-quality issues per reading: non-finite / out-of-band voltage,
     and ADC saturation. Statistics use VALID readings only.
  3. Per sensor (sensor_group x channel): mean dVt over valid readings, using
     the per-sensor MEASURED voltage uncertainty sigma_V from the lead-brick
     run (calibration.measured_sigma_v) as the measurement uncertainty.
  4. Convert mean dVt -> dose [Rad] with automatic curve/range selection, and
     propagate a 1-sigma dose uncertainty (measurement (+) calibration).
  5. Combine R1 and R2 (two trials) per configuration by inverse-variance
     weighting, in voltage space.
  6. SHIELDING EFFECTIVENESS + STATISTICAL SIGNIFICANCE:
       - absolute dose +/- full uncertainty per configuration;
       - significance of each config vs the bare sensor, tested in VOLTAGE
         space (measurement-only -> calibration error is common-mode and
         cancels in the difference);
       - a pairwise z-score matrix across all configurations.
  7. Write a text report and a tidy per-configuration summary CSV.

Run:
    cd sample_analysis
    python generate_fake_data.py   # once, to create fake_iss_data.csv
    python analyze_sample.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import (
    SHIELDING_BY_CHANNEL,
    SIGMA_V_COVERAGE,
    SIGMA_V_SOURCE,
    TRIAL_GROUPS,
    UNSHIELDED_CHANNEL,
    dose_uncertainty,
    measured_sigma_v,
    select_curve_and_dose,
    shielding_label,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "fake_iss_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
REPORT_PATH = OUTPUT_DIR / "dose_analysis_report.txt"
SUMMARY_CSV = OUTPUT_DIR / "dose_summary.csv"

GROUP_COL = "sensor_group"
CHANNEL_COL = "channel"
VOLTAGE_COL = "voltage"        # this column carries dVt (threshold-V shift)
ADC_COL = "raw_adc"
TIMESTAMP_COL = "timestamp"

EXPECTED_CHANNELS = [1, 2, 3, 4, 5]

VOLTAGE_VALID_MIN = 0.0
VOLTAGE_VALID_MAX = 4.0
SATURATION_ADC = 4090

TIMESTAMP_VALID_MIN = pd.Timestamp("2000-01-01")
TIMESTAMP_VALID_MAX = pd.Timestamp("2100-01-01")

# Significance thresholds on |z| (difference / combined sigma).
Z_95 = 1.96      # ~2 sigma, 95% two-sided
Z_997 = 3.0      # ~3 sigma, 99.7% two-sided


# --------------------------------------------------------------------------- #
# Load & quality control
# --------------------------------------------------------------------------- #

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Run `python generate_fake_data.py` first to create it."
        )
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    out_of_range = df[TIMESTAMP_COL].notna() & (
        (df[TIMESTAMP_COL] < TIMESTAMP_VALID_MIN)
        | (df[TIMESTAMP_COL] > TIMESTAMP_VALID_MAX)
    )
    df.loc[out_of_range, TIMESTAMP_COL] = pd.NaT
    return df


def flag_valid(df: pd.DataFrame) -> pd.Series:
    voltage = pd.to_numeric(df[VOLTAGE_COL], errors="coerce")
    adc = pd.to_numeric(df[ADC_COL], errors="coerce")
    return (
        np.isfinite(voltage)
        & (voltage > VOLTAGE_VALID_MIN)
        & (voltage <= VOLTAGE_VALID_MAX)
        & (adc < SATURATION_ADC)
    )


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


# --------------------------------------------------------------------------- #
# Per-sensor (one trial) processing
# --------------------------------------------------------------------------- #

def process_trial(df: pd.DataFrame, group: str, channel: int) -> dict | None:
    """
    Aggregate one sensor in one trial and convert to dose. The measurement
    uncertainty is the lead-brick sigma_V for this sensor (NOT the SEM), i.e.
    a characterized per-sensor noise floor.
    """
    mask = (df[GROUP_COL] == group) & (df[CHANNEL_COL] == channel)
    sub = df[mask]
    if sub.empty:
        return None

    valid = sub[sub["valid"]]
    n_total, n_valid = len(sub), len(valid)
    if n_valid == 0:
        return {"group": group, "channel": channel,
                "n_total": n_total, "n_valid": 0}

    dvt = pd.to_numeric(valid[VOLTAGE_COL], errors="coerce")
    dvt_mean = float(dvt.mean())
    dvt_std = float(dvt.std(ddof=1)) if n_valid > 1 else float("nan")

    # Measurement uncertainty: the characterized lead-brick sigma_V.
    sigma_v = measured_sigma_v(group, channel)

    curve, dose, extrapolated = select_curve_and_dose(dvt_mean)
    dose_sigma = dose_uncertainty(curve, dvt_mean, sigma_v) if curve else float("nan")

    return {
        "group": group, "channel": channel,
        "n_total": n_total, "n_valid": n_valid,
        "dvt_mean": dvt_mean, "dvt_std": dvt_std, "sigma_v": sigma_v,
        "curve": curve.name if curve else "n/a",
        "dose_rad": dose, "dose_sigma_rad": dose_sigma,
        "extrapolated": extrapolated,
    }


# --------------------------------------------------------------------------- #
# Combine trials (inverse-variance) per configuration
# --------------------------------------------------------------------------- #

def combine_trials(trials: list[dict]) -> dict:
    """
    Combine R1/R2 for one channel by inverse-variance weighting in VOLTAGE
    space (the natural measurement space), then convert the combined dVt to a
    final dose with full (measurement (+) calibration) uncertainty.
    """
    good = [t for t in trials
            if t and t.get("n_valid", 0) > 0 and np.isfinite(t.get("dvt_mean", np.nan))]
    if not good:
        return {"n_trials": 0, "dvt_mean": float("nan"),
                "dvt_sigma": float("nan"), "dose_mean": float("nan"),
                "dose_sigma": float("nan"), "dose_trial_halfrange": float("nan")}

    # Inverse-variance weighted mean of the per-trial dVt means.
    xs = np.array([t["dvt_mean"] for t in good])
    sig = np.array([t["sigma_v"] for t in good])
    w = 1.0 / sig**2
    dvt_mean = float(np.sum(w * xs) / np.sum(w))
    dvt_sigma = float(math.sqrt(1.0 / np.sum(w)))   # combined measurement sigma

    curve, dose, extrapolated = select_curve_and_dose(dvt_mean)
    dose_sigma = dose_uncertainty(curve, dvt_mean, dvt_sigma) if curve else float("nan")

    doses = [t["dose_rad"] for t in good if np.isfinite(t.get("dose_rad", np.nan))]
    halfrange = (max(doses) - min(doses)) / 2.0 if len(doses) > 1 else float("nan")

    return {
        "n_trials": len(good),
        "dvt_mean": dvt_mean, "dvt_sigma": dvt_sigma,
        "curve": curve.name if curve else "n/a",
        "dose_mean": dose, "dose_sigma": dose_sigma,
        "dose_trial_halfrange": halfrange,
        "extrapolated": extrapolated,
    }


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #

def significance_vs_reference(per_channel: dict, ref_ch: int) -> dict:
    """
    Test each configuration against the reference (bare) sensor in VOLTAGE
    space, where the calibration uncertainty is common-mode and cancels:

        z = (dVt_ref - dVt_cfg) / sqrt(sigma_ref^2 + sigma_cfg^2)
    """
    out: dict = {}
    ref = per_channel.get(ref_ch, {})
    v_ref, s_ref = ref.get("dvt_mean"), ref.get("dvt_sigma")
    for ch, c in per_channel.items():
        if ch == ref_ch or not np.isfinite(c.get("dvt_mean", np.nan)):
            continue
        if not (np.isfinite(v_ref) and np.isfinite(s_ref)):
            continue
        dv = v_ref - c["dvt_mean"]
        sd = math.sqrt(s_ref**2 + c["dvt_sigma"]**2)
        z = dv / sd if sd > 0 else float("nan")
        out[ch] = {"delta_v": dv, "sigma_delta_v": sd, "z": z,
                   "p": _twosided_p(z), "label": _significance_label(z)}
    return out


def pairwise_z_matrix(per_channel: dict, channels: list[int]) -> dict:
    """|z| of dVt difference for every pair of configurations (voltage space)."""
    z = {}
    for a in channels:
        for b in channels:
            ca, cb = per_channel.get(a, {}), per_channel.get(b, {})
            va, vb = ca.get("dvt_mean"), cb.get("dvt_mean")
            sa, sb = ca.get("dvt_sigma"), cb.get("dvt_sigma")
            if a == b or not all(np.isfinite(x) for x in (va, vb, sa, sb)):
                z[(a, b)] = float("nan")
            else:
                sd = math.sqrt(sa**2 + sb**2)
                z[(a, b)] = abs(va - vb) / sd if sd > 0 else float("nan")
    return z


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_report(per_channel, trial_results, sig, zmat, qc) -> str:
    lines: list[str] = []
    sep = "=" * 78
    sub = "-" * 78

    lines.append(sep)
    lines.append("RADFET ISS DOSIMETRY - DOSE ANALYSIS (SAMPLE / SYNTHETIC DATA)")
    lines.append(sep)
    lines.append(f"Source file        : {INPUT_CSV.name}")
    lines.append(f"Output directory   : {OUTPUT_DIR}")
    lines.append("Calibration        : Varadis QF 16 (dVt = A * Dose^B), 5 ranges")
    lines.append(f"Trials per sensor  : {', '.join(TRIAL_GROUPS)} (inverse-variance combined)")
    lines.append(f"Measurement sigma  : per-sensor lead-brick sigma_V "
                 f"(coverage factor {SIGMA_V_COVERAGE:g})")
    lines.append(f"  sigma_V source   : {SIGMA_V_SOURCE}")
    lines.append(f"Valid voltage band : ({VOLTAGE_VALID_MIN}, {VOLTAGE_VALID_MAX}] V; "
                 f"saturation raw_adc >= {SATURATION_ADC}")
    lines.append("")

    # QC
    lines.append(sub)
    lines.append("DATA QUALITY")
    lines.append(sub)
    lines.append(f"Total readings       : {qc['n_total']}")
    lines.append(f"Valid readings       : {qc['n_valid']}")
    lines.append(f"Rejected (band/NaN)  : {qc['n_band']}")
    lines.append(f"Rejected (saturation): {qc['n_sat']}")
    lines.append(f"Bad/missing timestamp: {qc['n_bad_ts']}")
    lines.append("")

    # Per-trial detail
    lines.append(sub)
    lines.append("PER-SENSOR, PER-TRIAL (measurement sigma = lead-brick sigma_V)")
    lines.append(sub)
    lines.append(f"{'config':>14} {'ch':>2} {'trial':>5} {'n':>4} "
                 f"{'dVt[V]':>9} {'sigV[V]':>8} {'curve':>10} {'dose[Rad]':>11} {'+/-':>9}")
    for ch in EXPECTED_CHANNELS:
        for group in TRIAL_GROUPS:
            t = trial_results.get((group, ch))
            label = shielding_label(ch)
            if not t or t.get("n_valid", 0) == 0:
                lines.append(f"{label:>14} {ch:>2} {group:>5} {'0':>4}  (no valid data)")
                continue
            flag = " *EXTRAP*" if t.get("extrapolated") else ""
            lines.append(
                f"{label:>14} {ch:>2} {group:>5} {t['n_valid']:>4} "
                f"{t['dvt_mean']:>9.4f} {t['sigma_v']:>8.4f} {t['curve']:>10} "
                f"{t['dose_rad']:>11.1f} {t['dose_sigma_rad']:>9.1f}{flag}"
            )
    lines.append("")

    # Combined per-configuration final result
    lines.append(sub)
    lines.append("FINAL RESULT PER CONFIGURATION (R1+R2 combined)")
    lines.append(sub)
    lines.append(f"{'config':>14} {'ch':>2} {'dVt[V]':>9} {'sigV[V]':>8} "
                 f"{'curve':>10} {'dose[Rad]':>11} {'dose +/-':>10}")
    for ch in EXPECTED_CHANNELS:
        c = per_channel.get(ch, {})
        label = shielding_label(ch)
        if not np.isfinite(c.get("dose_mean", np.nan)):
            lines.append(f"{label:>14} {ch:>2}   (no valid data)")
            continue
        lines.append(
            f"{label:>14} {ch:>2} {c['dvt_mean']:>9.4f} {c['dvt_sigma']:>8.4f} "
            f"{c['curve']:>10} {c['dose_mean']:>11.1f} {c['dose_sigma']:>10.1f}"
        )
    lines.append("")
    lines.append("  dose +/- is the FULL 1-sigma (measurement (+) calibration sigma_A/sigma_B),")
    lines.append("  appropriate for the absolute dose of a single configuration.")
    lines.append("")

    # Significance vs bare
    lines.append(sub)
    lines.append(f"SIGNIFICANCE vs BARE SENSOR (channel {UNSHIELDED_CHANNEL}), voltage space")
    lines.append(sub)
    lines.append("  Tested on dVt (measurement-only): calibration error is common-mode")
    lines.append("  and cancels in the difference. dose figures shown for magnitude only.")
    lines.append("")
    ref = per_channel.get(UNSHIELDED_CHANNEL, {})
    lines.append(f"{'config':>14} {'ch':>2} {'dDose[Rad]':>11} {'reduction':>10} "
                 f"{'ddVt[V]':>9} {'z':>7} {'p':>10} {'verdict':>9}")
    ref_dose = ref.get("dose_mean", float("nan"))
    for ch in EXPECTED_CHANNELS:
        if ch == UNSHIELDED_CHANNEL:
            lines.append(f"{shielding_label(ch):>14} {ch:>2}  (reference)")
            continue
        c = per_channel.get(ch, {})
        s = sig.get(ch)
        if not s or not np.isfinite(c.get("dose_mean", np.nan)):
            lines.append(f"{shielding_label(ch):>14} {ch:>2}   (no valid data)")
            continue
        ddose = ref_dose - c["dose_mean"]
        red = (1 - c["dose_mean"] / ref_dose) * 100 if ref_dose > 0 else float("nan")
        lines.append(
            f"{shielding_label(ch):>14} {ch:>2} {ddose:>11.1f} {red:>9.1f}% "
            f"{s['delta_v']:>9.4f} {s['z']:>7.2f} {s['p']:>10.2e} {s['label']:>9}"
        )
    lines.append("")

    # Pairwise |z| matrix
    lines.append(sub)
    lines.append("PAIRWISE |z| MATRIX (dVt difference; >2 = 95%, >3 = 99.7%)")
    lines.append(sub)
    header = "        ch  " + "".join(f"{ch:>8}" for ch in EXPECTED_CHANNELS)
    lines.append(header)
    for a in EXPECTED_CHANNELS:
        row = f"  ch{a} {shielding_label(a)[:6]:>6} "
        for b in EXPECTED_CHANNELS:
            v = zmat.get((a, b), float("nan"))
            row += f"{'-':>8}" if (a == b or not np.isfinite(v)) else f"{v:>8.1f}"
        lines.append(row)
    lines.append("")
    lines.append(sep)
    lines.append("End of report.")
    lines.append(sep)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT_CSV)

    voltage = pd.to_numeric(df[VOLTAGE_COL], errors="coerce")
    adc = pd.to_numeric(df[ADC_COL], errors="coerce")
    df["valid"] = flag_valid(df)
    qc = {
        "n_total": len(df),
        "n_valid": int(df["valid"].sum()),
        "n_band": int((~(np.isfinite(voltage)
                         & (voltage > VOLTAGE_VALID_MIN)
                         & (voltage <= VOLTAGE_VALID_MAX))).sum()),
        "n_sat": int((adc >= SATURATION_ADC).sum()),
        "n_bad_ts": int(df[TIMESTAMP_COL].isna().sum()),
    }

    trial_results: dict = {}
    for group in TRIAL_GROUPS:
        for ch in EXPECTED_CHANNELS:
            res = process_trial(df, group, ch)
            if res is not None:
                trial_results[(group, ch)] = res

    per_channel: dict = {}
    for ch in EXPECTED_CHANNELS:
        trials = [trial_results.get((g, ch)) for g in TRIAL_GROUPS]
        per_channel[ch] = combine_trials([t for t in trials if t])

    sig = significance_vs_reference(per_channel, UNSHIELDED_CHANNEL)
    zmat = pairwise_z_matrix(per_channel, EXPECTED_CHANNELS)

    report = build_report(per_channel, trial_results, sig, zmat, qc)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # Tidy summary CSV.
    ref_dose = per_channel.get(UNSHIELDED_CHANNEL, {}).get("dose_mean", float("nan"))
    rows = []
    for ch in EXPECTED_CHANNELS:
        c = per_channel.get(ch, {})
        dose = c.get("dose_mean", float("nan"))
        s = sig.get(ch, {})
        rows.append({
            "channel": ch,
            "shielding": shielding_label(ch),
            "dvt_mean_v": round(c.get("dvt_mean", float("nan")), 5) if np.isfinite(c.get("dvt_mean", np.nan)) else "",
            "dvt_sigma_v": round(c.get("dvt_sigma", float("nan")), 5) if np.isfinite(c.get("dvt_sigma", np.nan)) else "",
            "dose_rad": round(dose, 2) if np.isfinite(dose) else "",
            "dose_sigma_rad": round(c.get("dose_sigma", float("nan")), 2) if np.isfinite(c.get("dose_sigma", np.nan)) else "",
            "reduction_pct": round((1 - dose / ref_dose) * 100, 1) if np.isfinite(dose) and np.isfinite(ref_dose) and ref_dose > 0 else "",
            "z_vs_bare": round(s.get("z", float("nan")), 2) if np.isfinite(s.get("z", np.nan)) else "",
            "p_vs_bare": f"{s.get('p', float('nan')):.2e}" if np.isfinite(s.get("p", np.nan)) else "",
            "significance": s.get("label", "reference" if ch == UNSHIELDED_CHANNEL else ""),
        })
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    print(report)
    print(f"\nWrote report  -> {REPORT_PATH}")
    print(f"Wrote summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
