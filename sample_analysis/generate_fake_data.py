"""
generate_fake_data.py

Generate a synthetic dataset in the SAME format as LeadEnclosureData.csv so the
analysis pipeline can be exercised end-to-end without real flight data.

Columns produced (identical to the real file):
    timestamp, sensor_group, channel, raw_adc, voltage, dose_rad, raw_timestamp

How the fake data is built
--------------------------
* Each channel (1-5) is assigned a *true* accumulated dose that reflects its
  shielding: the bare sensor sees the most, the shielded sensors less.
* R1 and R2 are two TRIALS of the same five sensors, with a small per-trial
  offset so they are similar but not identical.
* For each reading we forward-model the threshold-voltage shift from the true
  dose using the Varadis fit (dVt = A * Dose^B) and add realistic measurement
  noise, then back out a plausible 12-bit ADC count.
* A handful of deliberate anomalies are injected (a synchronous saturation
  spike, some 1969-epoch timestamps, and one corrupt voltage) so the QC stage
  in the analysis has something to catch -- mirroring quirks in the real file.

This script writes:  sample_analysis/fake_iss_data.csv
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import CALIBRATION_CURVES, SHIELDING_BY_CHANNEL, TRIAL_GROUPS

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "fake_iss_data.csv"

RNG = np.random.default_rng(20260617)   # fixed seed -> reproducible dataset

N_CYCLES = 45                           # read-out cycles (one per minute)
CYCLE_PERIOD = timedelta(minutes=1)
START_TIME = datetime(2026, 6, 8, 8, 20, 1)  # fixed, mirrors the real file

# 12-bit ADC model inferred from the real data:
#   voltage = (raw_adc - BASELINE_ADC) * LSB,  LSB = Vref / 2^bits
VREF = 5.0
ADC_BITS = 12
ADC_MAX = 2 ** ADC_BITS - 1            # 4095
LSB = VREF / 2 ** ADC_BITS            # ~0.001220703 V/count
BASELINE_ADC = 1392                    # count at dVt = 0 (reproduces real data)

MEAS_NOISE_FRAC = 0.008                # ~0.8% per-reading noise on dVt

# True accumulated dose per channel (Rad). Bare sensor highest; shielding
# reduces dose. These are the "ground truth" the analysis should recover.
TRUE_DOSE_RAD = {
    1: 8200.0,   # None (bare)
    2: 5400.0,   # 2 mm Al
    3: 4300.0,   # MLC1
    4: 2900.0,   # MLC1-b + Al
    5: 3600.0,   # MLC2
}

# Per-trial multiplier so R1 and R2 differ slightly (unit-to-unit variation).
TRIAL_DOSE_FACTOR = {"R1": 1.00, "R2": 0.97}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _forward_curve_for_dose(dose: float):
    """Pick the narrowest Varadis fit whose range contains `dose`."""
    for curve in CALIBRATION_CURVES:        # narrowest -> widest
        if dose <= curve.dose_max_rad:
            return curve
    return CALIBRATION_CURVES[-1]


def _dvt_to_adc(dvt: float) -> int:
    """Back out a plausible raw ADC count from a dVt value."""
    return int(round(BASELINE_ADC + dvt / LSB))


def _adc_to_voltage(raw_adc: int) -> float:
    """Forward ADC -> voltage, matching the real file's relationship."""
    return (raw_adc - BASELINE_ADC) * LSB


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def build_rows() -> list[dict]:
    rows: list[dict] = []

    for cycle in range(N_CYCLES):
        cycle_time = START_TIME + cycle * CYCLE_PERIOD

        for group in TRIAL_GROUPS:
            # R2 readings trail R1 by ~0.75 s as in the real cadence.
            group_offset = timedelta(seconds=0.75) if group == "R2" else timedelta(0)

            for channel in (1, 2, 3, 4, 5):
                true_dose = TRUE_DOSE_RAD[channel] * TRIAL_DOSE_FACTOR[group]
                curve = _forward_curve_for_dose(true_dose)
                clean_dvt = curve.dvt_from_dose(true_dose)

                # Add per-reading measurement noise.
                dvt = clean_dvt * (1.0 + RNG.normal(0.0, MEAS_NOISE_FRAC))
                raw_adc = _dvt_to_adc(dvt)
                voltage = _adc_to_voltage(raw_adc)

                # dose_rad column: keep it populated like the real file. We
                # store the per-reading dose from the SAME curve (the analysis
                # recomputes dose itself, so this is just a convenience column).
                dose_rad = curve.dose_from_dvt(voltage) if voltage > 0 else 0.0

                ts = cycle_time + group_offset + timedelta(
                    milliseconds=12 * channel
                )
                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "sensor_group": group,
                        "channel": channel,
                        "raw_adc": raw_adc,
                        "voltage": round(voltage, 9),
                        "dose_rad": round(dose_rad, 6),
                        "raw_timestamp": ts.isoformat(),
                    }
                )

    _inject_anomalies(rows)
    return rows


def _inject_anomalies(rows: list[dict]) -> None:
    """
    Mutate `rows` in place to add realistic data-quality problems:

      1. A synchronous ADC saturation spike for one whole cycle (like the
         08:22 raw_adc~4032 event in the real file).
      2. A block of corrupted 1969-epoch raw_timestamps.
      3. One nonsensical voltage value.
    """
    # 1. Saturation spike: pick cycle index 10 (both groups, all channels).
    spike_time = START_TIME + 10 * CYCLE_PERIOD
    for row in rows:
        if abs(_parse(row["timestamp"]) - spike_time) < timedelta(seconds=30):
            row["raw_adc"] = ADC_MAX
            row["voltage"] = round(_adc_to_voltage(ADC_MAX), 9)
            row["dose_rad"] = -1.0  # obviously bogus marker

    # 2. Corrupt the raw_timestamp on the last ~12 rows to the 1969 epoch.
    for i, row in enumerate(rows[-12:]):
        bad = datetime(1969, 12, 31, 19, 0, 0) + timedelta(seconds=i)
        row["raw_timestamp"] = bad.isoformat()

    # 3. One absurd voltage reading.
    if len(rows) > 25:
        rows[25]["voltage"] = 3.8e14


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "sensor_group", "channel",
            "raw_adc", "voltage", "dose_rad", "raw_timestamp",
        ],
    )
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    print("\nTrue dose per channel (Rad), R1 trial:")
    for ch in (1, 2, 3, 4, 5):
        print(f"  ch{ch} [{SHIELDING_BY_CHANNEL[ch]:>12}] : {TRUE_DOSE_RAD[ch]:>8.0f}")


if __name__ == "__main__":
    main()
