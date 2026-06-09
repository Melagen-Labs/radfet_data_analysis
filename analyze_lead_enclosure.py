"""
analyze_lead_enclosure.py

Analyze voltage stability and baseline behavior for 10 RADFET dosimeter
sensors recorded inside a lead enclosure (LeadEnclosureData.csv).

A unique sensor is the combination of:
    * sensor_group : "R1" or "R2"
    * channel      : 1 .. 5
giving 10 expected sensors total.

For each sensor this script:
    * writes a per-sensor folder + CSV (every row kept, anomalies flagged),
    * augments the CSV with validity flags, difference-from-mean and
      absolute-difference columns (and elapsed time when timestamps exist),
    * computes descriptive voltage statistics over VALID readings only,
and finally writes a combined text report (analysis_report.txt).

-------------------------------------------------------------------------------
Assumptions (made explicit)
-------------------------------------------------------------------------------
* The voltage column to analyze is "voltage" (volts) and the timestamp
  column is "timestamp" (ISO-8601).

* DATA QUALITY: the source file contains corrupted rows. We do NOT silently
  drop them; instead we flag them and exclude them from statistics so the
  baseline reflects real sensor behavior. A reading is treated as a valid
  voltage when it is finite and within (VOLTAGE_VALID_MIN, VOLTAGE_VALID_MAX].
  These bounds bracket a physically plausible RADFET source-follower output
  (legitimate readings here sit around 0.15-0.40 V); values like 3.8e14, or a
  hard 0 V alongside a non-zero ADC count, are corruption, not signal.

* TIMESTAMPS: values outside [TIMESTAMP_VALID_MIN, TIMESTAMP_VALID_MAX] (e.g.
  the 1969 epoch rows in this dataset) are treated as missing (NaT) so they
  neither become a false "first reading" nor distort elapsed-time. Sorting and
  elapsed time are computed from valid timestamps only.

* Statistics use sample standard deviation (ddof=1, pandas default). With
  fewer than 2 valid readings this is NaN, reported as such (not 0).

Run:
    python analyze_lead_enclosure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Paths are resolved relative to this script so it runs the same regardless
# of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "LeadEnclosureData.csv"
OUTPUT_DIR = SCRIPT_DIR / "sensor_analysis"
REPORT_PATH = OUTPUT_DIR / "analysis_report.txt"

# Source column names.
GROUP_COL = "sensor_group"
CHANNEL_COL = "channel"
VOLTAGE_COL = "voltage"
TIMESTAMP_COL = "timestamp"

# The 10 sensors we expect: every (group, channel) pairing.
EXPECTED_GROUPS = ["R1", "R2"]
EXPECTED_CHANNELS = [1, 2, 3, 4, 5]
EXPECTED_SENSORS = [
    (group, channel)
    for group in EXPECTED_GROUPS
    for channel in EXPECTED_CHANNELS
]

# Plausibility bounds for anomaly detection (see module docstring).
VOLTAGE_VALID_MIN = 0.0          # exclusive: a real reading is positive
VOLTAGE_VALID_MAX = 5.0          # volts: RADFET follower output ceiling
TIMESTAMP_VALID_MIN = pd.Timestamp("2000-01-01")
TIMESTAMP_VALID_MAX = pd.Timestamp("2100-01-01")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sensor_label(group: str, channel: int) -> str:
    """Filesystem-friendly identifier for a sensor, e.g. 'R1_ch3'."""
    return f"{group}_ch{channel}"


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV, parse timestamps, and null out implausible ones."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if TIMESTAMP_COL in df.columns:
        # errors="coerce" -> unparseable strings become NaT instead of raising.
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

        # Treat out-of-window timestamps (e.g. 1969 epoch corruption) as NaT
        # so they cannot masquerade as the first reading.
        out_of_range = df[TIMESTAMP_COL].notna() & (
            (df[TIMESTAMP_COL] < TIMESTAMP_VALID_MIN)
            | (df[TIMESTAMP_COL] > TIMESTAMP_VALID_MAX)
        )
        df.loc[out_of_range, TIMESTAMP_COL] = pd.NaT

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Fail fast if required columns are absent."""
    required = {GROUP_COL, CHANNEL_COL, VOLTAGE_COL}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Input is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


def have_usable_timestamps(df: pd.DataFrame) -> bool:
    """True only if a timestamp column exists with at least one valid value."""
    if TIMESTAMP_COL not in df.columns:
        return False
    return df[TIMESTAMP_COL].notna().any()


def flag_valid_voltage(voltages: pd.Series) -> pd.Series:
    """Boolean mask: finite and within the plausible RADFET voltage band."""
    numeric = pd.to_numeric(voltages, errors="coerce")
    return (
        np.isfinite(numeric)
        & (numeric > VOLTAGE_VALID_MIN)
        & (numeric <= VOLTAGE_VALID_MAX)
    )


# --------------------------------------------------------------------------- #
# Per-sensor processing
# --------------------------------------------------------------------------- #

def compute_statistics(valid_voltages: pd.Series) -> dict:
    """Descriptive statistics over a sensor's VALID voltage readings."""
    return {
        "n_valid": int(valid_voltages.count()),
        "mean": valid_voltages.mean(),
        "median": valid_voltages.median(),
        # Sample standard deviation (ddof=1). NaN when fewer than 2 readings.
        "std": valid_voltages.std(),
        "min": valid_voltages.min(),
        "max": valid_voltages.max(),
        "peak_to_peak": valid_voltages.max() - valid_voltages.min(),
    }


def process_sensor(
    sensor_df: pd.DataFrame,
    group: str,
    channel: int,
    use_timestamps: bool,
) -> dict:
    """
    Process one sensor: flag anomalies, sort, derive columns, save its CSV,
    and return a results dict (stats + anomaly counts).
    """
    # Copy so we never mutate the shared parent frame.
    sensor_df = sensor_df.copy()

    # Step 8: sort chronologically when timestamps are usable. NaT sorts last
    # (na_position default) so corrupt-timestamp rows trail valid ones.
    if use_timestamps:
        sensor_df = sensor_df.sort_values(
            TIMESTAMP_COL, kind="mergesort", na_position="last"
        ).reset_index(drop=True)

        # Step 9: elapsed seconds from this sensor's first VALID reading.
        valid_ts = sensor_df[TIMESTAMP_COL].dropna()
        first_ts = valid_ts.iloc[0] if not valid_ts.empty else pd.NaT
        sensor_df["elapsed_seconds"] = (
            sensor_df[TIMESTAMP_COL] - first_ts
        ).dt.total_seconds()

    # --- Validity flagging ------------------------------------------------- #
    sensor_df["voltage_valid"] = flag_valid_voltage(sensor_df[VOLTAGE_COL])
    n_total = len(sensor_df)
    n_valid = int(sensor_df["voltage_valid"].sum())
    n_invalid = n_total - n_valid
    n_missing = int(pd.to_numeric(sensor_df[VOLTAGE_COL], errors="coerce")
                    .isna().sum())
    n_bad_timestamp = (
        int(sensor_df[TIMESTAMP_COL].isna().sum())
        if TIMESTAMP_COL in sensor_df.columns else 0
    )

    # Statistics over valid readings only, so the baseline isn't skewed.
    valid_voltages = pd.to_numeric(
        sensor_df.loc[sensor_df["voltage_valid"], VOLTAGE_COL], errors="coerce"
    )
    stats = compute_statistics(valid_voltages)

    # Steps 6 & 7: difference / absolute difference from the (valid) mean.
    mean_voltage = stats["mean"]
    numeric_voltage = pd.to_numeric(sensor_df[VOLTAGE_COL], errors="coerce")
    sensor_df["voltage_diff_from_mean"] = numeric_voltage - mean_voltage
    sensor_df["voltage_abs_diff_from_mean"] = (
        sensor_df["voltage_diff_from_mean"].abs()
    )

    # Steps 3 & 4: one folder + one CSV per sensor (all rows kept, flagged).
    label = sensor_label(group, channel)
    sensor_folder = OUTPUT_DIR / label
    sensor_folder.mkdir(parents=True, exist_ok=True)
    sensor_df.to_csv(sensor_folder / f"{label}.csv", index=False)

    return {
        "stats": stats,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_missing": n_missing,
        "n_bad_timestamp": n_bad_timestamp,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_report(
    results: dict,
    use_timestamps: bool,
    present_sensors: set,
    unexpected_sensors: list,
) -> str:
    """Assemble the full text report as a single string."""
    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("RADFET LEAD ENCLOSURE - VOLTAGE STABILITY ANALYSIS")
    lines.append(sep)
    lines.append(f"Source file        : {INPUT_CSV.name}")
    lines.append(f"Output directory   : {OUTPUT_DIR}")
    lines.append(f"Expected sensors   : {len(EXPECTED_SENSORS)}")
    lines.append(f"Sensors found      : {len(present_sensors)}")
    lines.append(
        "Timestamps         : "
        + ("available (sorted; elapsed time from first valid reading)"
           if use_timestamps else "NOT available (time steps skipped)")
    )
    lines.append(
        f"Valid voltage band : ({VOLTAGE_VALID_MIN}, {VOLTAGE_VALID_MAX}] V"
    )
    lines.append("")

    # --- Validation -------------------------------------------------------- #
    lines.append("-" * 72)
    lines.append("VALIDATION")
    lines.append("-" * 72)

    missing_sensors = [s for s in EXPECTED_SENSORS if s not in present_sensors]
    if not missing_sensors:
        lines.append("All 10 expected sensors are present. [OK]")
    else:
        lines.append("MISSING expected sensors:")
        for group, channel in missing_sensors:
            lines.append(f"  - {sensor_label(group, channel)}")

    if unexpected_sensors:
        lines.append("UNEXPECTED sensor/channel combinations found:")
        for group, channel in unexpected_sensors:
            lines.append(f"  - sensor_group={group!r}, channel={channel!r}")
    else:
        lines.append("No unexpected sensor/channel combinations. [OK]")

    total_invalid = sum(r["n_invalid"] for r in results.values())
    total_bad_ts = sum(r["n_bad_timestamp"] for r in results.values())
    lines.append(
        f"Anomalous voltage readings (excluded from stats): {total_invalid}"
    )
    lines.append(f"Out-of-range / missing timestamps: {total_bad_ts}")
    lines.append("")

    # --- Per-sensor stats -------------------------------------------------- #
    lines.append("-" * 72)
    lines.append("PER-SENSOR VOLTAGE STATISTICS (volts, valid readings only)")
    lines.append("-" * 72)

    # Canonical expected order for a stable, readable report.
    for group, channel in EXPECTED_SENSORS:
        label = sensor_label(group, channel)
        if (group, channel) not in results:
            lines.append(f"\n[{label}]  *** NO DATA ***")
            continue

        r = results[(group, channel)]
        s = r["stats"]
        lines.append(f"\n[{label}]")
        lines.append(
            f"  samples (valid/total) : {r['n_valid']} / {r['n_total']}"
        )
        if r["n_invalid"]:
            lines.append(f"  anomalous readings    : {r['n_invalid']}")
        if r["n_bad_timestamp"]:
            lines.append(f"  bad/missing timestamps: {r['n_bad_timestamp']}")

        if r["n_valid"] == 0:
            lines.append("  (no valid readings - statistics unavailable)")
            continue

        lines.append(f"  mean                  : {s['mean']:.9f}")
        lines.append(f"  median                : {s['median']:.9f}")
        std_text = "n/a (n<2)" if pd.isna(s["std"]) else f"{s['std']:.9f}"
        lines.append(f"  std dev (sample)      : {std_text}")
        lines.append(f"  min                   : {s['min']:.9f}")
        lines.append(f"  max                   : {s['max']:.9f}")
        lines.append(f"  peak-to-peak range    : {s['peak_to_peak']:.9f}")

    lines.append("")
    lines.append(sep)
    lines.append("End of report.")
    lines.append(sep)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    # Step 1: load (with timestamp parsing + range cleaning).
    df = load_data(INPUT_CSV)
    validate_columns(df)

    # Create the output root early so it exists even for an empty dataset.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_timestamps = have_usable_timestamps(df)

    # Step 2: identify the unique sensors actually present.
    present_sensors = set(
        (group, int(channel))
        for group, channel in df[[GROUP_COL, CHANNEL_COL]]
        .dropna()
        .itertuples(index=False, name=None)
    )

    # Flag any (group, channel) outside the 10 we expect.
    unexpected_sensors = sorted(
        s for s in present_sensors if s not in set(EXPECTED_SENSORS)
    )

    # Steps 3-9: process each expected sensor that has data.
    results: dict = {}
    for group, channel in EXPECTED_SENSORS:
        mask = (df[GROUP_COL] == group) & (df[CHANNEL_COL] == channel)
        sensor_df = df[mask]
        if sensor_df.empty:
            continue  # Reported as missing in the validation section.
        results[(group, channel)] = process_sensor(
            sensor_df, group, channel, use_timestamps
        )

    # Step 10: write the combined report.
    report = build_report(
        results, use_timestamps, present_sensors, unexpected_sensors
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    # Console summary for immediate feedback.
    print(report)
    print(f"\nWrote per-sensor CSVs and report under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
