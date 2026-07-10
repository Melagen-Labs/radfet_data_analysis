"""
config.py

Dashboard-wide constants: the ADC -> delta_v conversion settings, QC validity
rules, sensor layout, data-source registry, and UI option lists.

Voltage conversion (the non-negotiable rule of this repo):
    raw_voltage = raw_adc * (ADC_VREF_V / ADC_FULL_SCALE)
    delta_v     = raw_voltage - baseline[sensor_group] - BOARD_OFFSET_V

The CSV `voltage` / `delta_voltage_v` / `dose_rad` columns are NEVER trusted:
the baseline used to produce them changed inconsistently over time, while
raw_adc is the untouched measurement. Constants sourced from
ambient_analysis/analyze_ambient.py; the board offset voltage is the readout
board's series offset (hardware setting, may change between missions), which
is why all three settings are user-adjustable in the sidebar.
"""

from __future__ import annotations

import pandas as pd

from . import paths  # noqa: F401  (bootstrap runs via core/__init__)

import calibration  # sample_analysis, via paths.bootstrap()

# --------------------------------------------------------------------------- #
# Voltage conversion settings (defaults; the sidebar can override per session)
# --------------------------------------------------------------------------- #

ADC_VREF_V = 5.0
ADC_FULL_SCALE = 4095  # 12-bit, matches ambient_analysis convention

DV_BASELINE_BY_GROUP = {"R1": 1.71, "R2": 1.73}  # volts
BOARD_OFFSET_V = 0.10                            # volts, current board setting

# --------------------------------------------------------------------------- #
# QC / validity rules (identical to sample_analysis/analyze_sample.py)
# --------------------------------------------------------------------------- #

VOLTAGE_VALID_MIN = 0.0     # delta_v must be > this ...
VOLTAGE_VALID_MAX = 4.0     # ... and <= this [V]
SATURATION_ADC = 4090       # raw_adc >= this counts as saturated
TIMESTAMP_VALID_MIN = pd.Timestamp("2000-01-01")
TIMESTAMP_VALID_MAX = pd.Timestamp("2100-01-01")

# --------------------------------------------------------------------------- #
# Sensor layout
# --------------------------------------------------------------------------- #

TRIAL_GROUPS = list(calibration.TRIAL_GROUPS)          # ["R1", "R2"]
EXPECTED_CHANNELS = [1, 2, 3, 4, 5]
UNSHIELDED_CHANNEL = calibration.UNSHIELDED_CHANNEL    # 1

# Channel -> shielding label. calibration.SHIELDING_BY_CHANNEL is the default,
# but the mapping for ch3-5 disagrees between calibration.py and the legacy
# fake-data comments/README, so it is overridable here until confirmed.
SHIELDING_BY_CHANNEL_OVERRIDE: dict[int, str] | None = None


def shield_label(channel: int) -> str:
    mapping = SHIELDING_BY_CHANNEL_OVERRIDE or calibration.SHIELDING_BY_CHANNEL
    return mapping.get(channel, f"channel {channel}")


def sensor_id(group: str, channel: int) -> str:
    return f"{group}_ch{channel}"


ALL_SENSORS = [(g, ch) for g in TRIAL_GROUPS for ch in EXPECTED_CHANNELS]

# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #

# label -> (kind, path). kind "file" = single CSV, "dir" = folder of daily CSVs.
DATA_SOURCES: dict[str, tuple[str, object]] = {
    "ISS mission (simulated)": ("file", paths.SIM_MISSION_CSV),
    "Historic ambient (Jun-Jul 2026)": ("dir", paths.HISTORIC_DATA_DIR),
}
DEFAULT_SOURCE = "ISS mission (simulated)"
UPLOAD_SOURCE = "Upload CSV..."

# --------------------------------------------------------------------------- #
# UI options
# --------------------------------------------------------------------------- #

# label -> pandas resample frequency
WINDOW_OPTIONS: dict[str, str] = {
    "15 min": "15min",
    "1 hour": "1h",
    "6 hours": "6h",
    "1 day": "1D",
}
DEFAULT_WINDOW = "1 hour"

# Significance thresholds shared across pages (match analyze_sample.py).
Z_95 = 1.96
Z_997 = 3.0
