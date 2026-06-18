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
  3. Per sensor (sensor_group x channel): mean dVt, spread, and standard
     error of the mean over valid readings.
  4. Convert mean dVt -> dose [Rad] using the Varadis calibration with
     automatic range selection (calibration.select_curve_and_dose), and
     propagate a 1-sigma dose uncertainty (calibration.dose_uncertainty).
  5. Combine R1 and R2 as two TRIALS of each shielding configuration.
  6. Compare each shielded configuration's dose against the bare sensor to
     quantify shielding effectiveness.
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
    TRIAL_GROUPS,
    UNSHIELDED_CHANNEL,
    dose_uncertainty,
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

# Validity bounds. Unlike the lead-enclosure baseline study, ISS dVt values can
# be volts-large, so the upper band is wide; saturation is caught separately by
# the ADC count rather than by the voltage band.
VOLTAGE_VALID_MIN = 0.0        # exclusive
VOLTAGE_VALID_MAX = 4.0        # volts; rejects the 3.8e14-type corruption
SATURATION_ADC = 4090         # raw_adc at/above this is a rail/saturation hit

TIMESTAMP_VALID_MIN = pd.Timestamp("2000-01-01")
TIMESTAMP_VALID_MAX = pd.Timestamp("2100-01-01")


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
    """Boolean mask of physically usable readings."""
    voltage = pd.to_numeric(df[VOLTAGE_COL], errors="coerce")
    adc = pd.to_numeric(df[ADC_COL], errors="coerce")
    return (
        np.isfinite(voltage)
        & (voltage > VOLTAGE_VALID_MIN)
        & (voltage <= VOLTAGE_VALID_MAX)
        & (adc < SATURATION_ADC)
    )


# --------------------------------------------------------------------------- #
# Per-sensor (one trial) processing
# --------------------------------------------------------------------------- #

def process_trial(df: pd.DataFrame, group: str, channel: int) -> dict | None:
    """
    Aggregate one sensor in one trial (group) and convert to dose.
    Returns None when the sensor has no valid readings.
    """
    mask = (df[GROUP_COL] == group) & (df[CHANNEL_COL] == channel)
    sub = df[mask]
    if sub.empty:
        return None

    valid = sub[sub["valid"]]
    n_total = len(sub)
    n_valid = len(valid)
    if n_valid == 0:
        return {
            "group": group, "channel": channel,
            "n_total": n_total, "n_valid": 0,
        }

    dvt = pd.to_numeric(valid[VOLTAGE_COL], errors="coerce")
    dvt_mean = float(dvt.mean())
    dvt_std = float(dvt.std(ddof=1)) if n_valid > 1 else float("nan")
    # Standard error of the mean -> measurement uncertainty on dVt_mean.
    sem = dvt_std / math.sqrt(n_valid) if n_valid > 1 else float("nan")

    curve, dose, extrapolated = select_curve_and_dose(dvt_mean)
    sigma_dvt = sem if np.isfinite(sem) else 0.0
    dose_sigma = dose_uncertainty(curve, dvt_mean, sigma_dvt) if curve else float("nan")

    return {
        "group": group, "channel": channel,
        "n_total": n_total, "n_valid": n_valid,
        "dvt_mean": dvt_mean, "dvt_std": dvt_std, "dvt_sem": sem,
        "curve": curve.name if curve else "n/a",
        "dose_rad": dose, "dose_sigma_rad": dose_sigma,
        "extrapolated": extrapolated,
    }


# --------------------------------------------------------------------------- #
# Combine trials & shielding comparison
# --------------------------------------------------------------------------- #

def combine_trials(trials: list[dict]) -> dict:
    """Combine R1/R2 trial doses for one channel into a single estimate."""
    doses = [t["dose_rad"] for t in trials
             if t and t.get("dose_rad") is not None and np.isfinite(t.get("dose_rad", np.nan))]
    if not doses:
        return {"dose_mean": float("nan"), "dose_trial_halfrange": float("nan"),
                "n_trials": 0}
    dose_mean = float(np.mean(doses))
    # Half the spread between trials -> simple trial-to-trial uncertainty.
    halfrange = (max(doses) - min(doses)) / 2.0 if len(doses) > 1 else float("nan")
    # Largest per-trial propagated sigma, kept for context.
    sigmas = [t["dose_sigma_rad"] for t in trials
              if t and np.isfinite(t.get("dose_sigma_rad", np.nan))]
    return {
        "dose_mean": dose_mean,
        "dose_trial_halfrange": halfrange,
        "dose_calib_sigma": max(sigmas) if sigmas else float("nan"),
        "n_trials": len(doses),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_report(per_channel: dict, trial_results: dict, qc: dict) -> str:
    lines: list[str] = []
    sep = "=" * 74
    sub = "-" * 74

    lines.append(sep)
    lines.append("RADFET ISS DOSIMETRY - DOSE ANALYSIS (SAMPLE / SYNTHETIC DATA)")
    lines.append(sep)
    lines.append(f"Source file        : {INPUT_CSV.name}")
    lines.append(f"Output directory   : {OUTPUT_DIR}")
    lines.append(f"Calibration        : Varadis QF 16 (dVt = A * Dose^B), 5 ranges")
    lines.append(f"Trials per sensor  : {', '.join(TRIAL_GROUPS)} (treated as replicates)")
    lines.append(f"Valid voltage band : ({VOLTAGE_VALID_MIN}, {VOLTAGE_VALID_MAX}] V")
    lines.append(f"Saturation flag    : raw_adc >= {SATURATION_ADC}")
    lines.append("")

    # QC summary
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
    lines.append("PER-SENSOR, PER-TRIAL DOSE")
    lines.append(sub)
    header = (f"{'config':>14} {'ch':>2} {'trial':>5} {'n':>4} "
              f"{'dVt[V]':>9} {'curve':>10} {'dose[Rad]':>11} {'+/-':>9}")
    lines.append(header)
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
                f"{t['dvt_mean']:>9.4f} {t['curve']:>10} "
                f"{t['dose_rad']:>11.1f} {t['dose_sigma_rad']:>9.1f}{flag}"
            )
    lines.append("")

    # Combined per-configuration + shielding effectiveness
    lines.append(sub)
    lines.append("SHIELDING EFFECTIVENESS (R1+R2 combined, vs bare sensor)")
    lines.append(sub)

    ref = per_channel.get(UNSHIELDED_CHANNEL, {})
    ref_dose = ref.get("dose_mean", float("nan"))
    lines.append(f"Reference (bare) dose: {ref_dose:.1f} Rad "
                 f"[channel {UNSHIELDED_CHANNEL}]")
    lines.append("")
    lines.append(f"{'config':>14} {'ch':>2} {'dose[Rad]':>11} "
                 f"{'trial +/-':>10} {'atten.x':>8} {'reduction':>10}")
    for ch in EXPECTED_CHANNELS:
        c = per_channel.get(ch)
        label = shielding_label(ch)
        if not c or not np.isfinite(c.get("dose_mean", np.nan)):
            lines.append(f"{label:>14} {ch:>2}   (no valid data)")
            continue
        dose = c["dose_mean"]
        hr = c.get("dose_trial_halfrange", float("nan"))
        hr_txt = f"{hr:.1f}" if np.isfinite(hr) else "n/a"
        if np.isfinite(ref_dose) and ref_dose > 0:
            atten = ref_dose / dose if dose > 0 else float("nan")
            reduction = (1.0 - dose / ref_dose) * 100.0
            atten_txt = f"{atten:.2f}" if np.isfinite(atten) else "n/a"
            red_txt = f"{reduction:+.1f}%"
        else:
            atten_txt, red_txt = "n/a", "n/a"
        marker = "  (reference)" if ch == UNSHIELDED_CHANNEL else ""
        lines.append(
            f"{label:>14} {ch:>2} {dose:>11.1f} {hr_txt:>10} "
            f"{atten_txt:>8} {red_txt:>10}{marker}"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("  * atten.x   = bare_dose / config_dose (higher = more shielding).")
    lines.append("  * reduction = percent dose reduction vs the bare sensor.")
    lines.append("  * trial +/- = half the spread between the R1 and R2 doses.")
    lines.append("  * dose +/-  (per-trial table) = 1-sigma from measurement SEM")
    lines.append("    plus calibration sigma_A / sigma_B (see calibration.py).")
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

    # Quality control flags.
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

    # Per-trial processing.
    trial_results: dict = {}
    for group in TRIAL_GROUPS:
        for ch in EXPECTED_CHANNELS:
            res = process_trial(df, group, ch)
            if res is not None:
                trial_results[(group, ch)] = res

    # Combine trials per channel.
    per_channel: dict = {}
    for ch in EXPECTED_CHANNELS:
        trials = [trial_results.get((g, ch)) for g in TRIAL_GROUPS]
        per_channel[ch] = combine_trials([t for t in trials if t])

    # Write report.
    report = build_report(per_channel, trial_results, qc)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # Tidy summary CSV (one row per shielding configuration).
    ref_dose = per_channel.get(UNSHIELDED_CHANNEL, {}).get("dose_mean", float("nan"))
    rows = []
    for ch in EXPECTED_CHANNELS:
        c = per_channel.get(ch, {})
        dose = c.get("dose_mean", float("nan"))
        rows.append({
            "channel": ch,
            "shielding": shielding_label(ch),
            "dose_rad": round(dose, 2) if np.isfinite(dose) else "",
            "trial_halfrange_rad": (round(c.get("dose_trial_halfrange", float("nan")), 2)
                                    if np.isfinite(c.get("dose_trial_halfrange", np.nan)) else ""),
            "calib_sigma_rad": (round(c.get("dose_calib_sigma", float("nan")), 2)
                                if np.isfinite(c.get("dose_calib_sigma", np.nan)) else ""),
            "n_trials": c.get("n_trials", 0),
            "attenuation_factor": (round(ref_dose / dose, 3)
                                   if np.isfinite(dose) and dose > 0 and np.isfinite(ref_dose)
                                   else ""),
            "reduction_pct": (round((1 - dose / ref_dose) * 100, 1)
                              if np.isfinite(dose) and np.isfinite(ref_dose) and ref_dose > 0
                              else ""),
        })
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    print(report)
    print(f"\nWrote report  -> {REPORT_PATH}")
    print(f"Wrote summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
