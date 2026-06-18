"""
calibration.py

Varadis QF 16 RADFET calibration model and dose-conversion helpers.

This module is the SINGLE SOURCE OF TRUTH for the calibration. Both the
sample analysis script and the methodology document are generated from the
coefficients defined here, so the numbers can never drift between code and docs.

-------------------------------------------------------------------------------
Calibration source
-------------------------------------------------------------------------------
Varadis "QF 16 RADFET Test Data Record", Issue No.01, Issue Date 01/08/2023.

    Device type     : 400nm IMPL RADFET (plastic package)
    Mask-set        : COMRAD
    Lot number      : P5925-W3
    Irradiation site: Institute of Nuclear Sciences Vinca, Belgrade, Serbia
    Source          : Co-60
    Dose rate       : 5.0 kRad/hour
    Measurements    : On-line (automated read-out board)
    Biasing         : All pins grounded during irradiation

    Fitting function: dVt [V] = A * Dose[Rad] ** B
    Inversion       : Dose[Rad] = (dVt / A) ** (1 / B)

dVt is the threshold-voltage shift = (Vt of irradiated RADFET) - (Vt of a
non-irradiated RADFET). In our recorded CSVs this shift is already what the
`voltage` column carries, so no extra baseline subtraction is done here.

NOTE: Varadis provides FIVE fits, each valid over a different dose range, and
explicitly advises: "For better calibration accuracy, please use the curve
applicable to the actual dose range for the application." Range selection is
therefore part of the conversion (see `select_curve_and_dose`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Calibration curves
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalibrationCurve:
    """One Varadis fit, valid for dose in (0, dose_max_rad]."""

    name: str            # human label, e.g. "0-10 kRad"
    dose_max_rad: float  # upper bound of the fit's validity range, in Rad
    A: float
    sigma_A: float
    B: float
    sigma_B: float
    r_square: float

    def dose_from_dvt(self, dvt: float) -> float:
        """Invert the fit: Dose[Rad] = (dVt / A) ** (1 / B)."""
        return (dvt / self.A) ** (1.0 / self.B)

    def dvt_from_dose(self, dose: float) -> float:
        """Forward model: dVt[V] = A * Dose[Rad] ** B."""
        return self.A * dose ** self.B


# Ordered NARROWEST -> WIDEST. The narrow fits hug the low-dose data and are
# the most accurate there; we always prefer the narrowest curve that is still
# valid for the dose we measure.
CALIBRATION_CURVES: list[CalibrationCurve] = [
    #               name          dose_max      A        sigma_A      B        sigma_B    R^2
    CalibrationCurve("0-1 kRad",     1_000.0, 0.0008, 5.374e-05, 0.9119, 1.034e-02, 1.000),
    CalibrationCurve("0-5 kRad",     5_000.0, 0.0033, 2.994e-04, 0.7062, 1.104e-02, 0.997),
    CalibrationCurve("0-10 kRad",   10_000.0, 0.0068, 4.496e-04, 0.6164, 7.490e-03, 0.996),
    CalibrationCurve("0-50 kRad",   50_000.0, 0.0241, 5.361e-04, 0.4752, 2.150e-03, 0.997),
    CalibrationCurve("0-100 kRad", 100_000.0, 0.0295, 2.992e-04, 0.4551, 9.215e-04, 0.999),
]


# --------------------------------------------------------------------------- #
# Sensor / shielding configuration
# --------------------------------------------------------------------------- #
#
# We fly 5 sensors (channels 1-5), each with a different shielding stack.
# Each sensor is read out at two positions / units, recorded as sensor_group
# "R1" and "R2", which we treat as two independent TRIALS (replicates) of the
# same shielding configuration.
#
# Channel -> shielding mapping (confirmed 2026-06-17):
SHIELDING_BY_CHANNEL: dict[int, str] = {
    1: "None (bare)",
    2: "2 mm Al",
    3: "MLC1",
    4: "MLC1-b + Al",
    5: "MLC2",
}

# The reference channel against which shielding attenuation is measured.
UNSHIELDED_CHANNEL = 1

# The two read-out groups treated as repeat trials of the same sensors.
TRIAL_GROUPS = ["R1", "R2"]


# --------------------------------------------------------------------------- #
# Per-sensor measured voltage uncertainty (the "lead-brick" noise floor)
# --------------------------------------------------------------------------- #
#
# In the lead-enclosure run the dose is essentially fixed, so each sensor's
# voltage spread is its MEASUREMENT uncertainty sigma_V (a characterized
# per-sensor noise floor). These 1-sigma values are the "std dev (sample)"
# figures reported per sensor in sensor_analysis/analysis_report.txt, and are
# used as the `sigma_dvt` term in dose_uncertainty() and as the measurement
# uncertainty in the significance tests.
#
# The numbers are READ DIRECTLY from analysis_report.txt at import time, so they
# stay in sync whenever the lead-brick analysis is re-run. The dict below is a
# transcribed fallback used only if the report cannot be found/parsed.
_FALLBACK_SIGMA_V: dict[tuple[str, int], float] = {
    ("R1", 1): 0.004474975,
    ("R1", 2): 0.004557302,
    ("R1", 3): 0.004914784,
    ("R1", 4): 0.005352881,
    ("R1", 5): 0.004241809,
    ("R2", 1): 0.005164367,
    ("R2", 2): 0.005735375,
    ("R2", 3): 0.004099858,
    ("R2", 4): 0.005166357,
    ("R2", 5): 0.004029666,
}

# Path to the lead-brick analysis report (the source of per-sensor sigma_V).
LEAD_BRICK_REPORT = (
    Path(__file__).resolve().parent.parent / "sensor_analysis" / "analysis_report.txt"
)

# Coverage factor applied to every sigma_V (1.0 = raw 1-sigma from the report,
# 2.0 = 2-sigma, etc.). Falls back to SIGMA_V_DEFAULT for unlisted sensors.
SIGMA_V_COVERAGE = 1.0
SIGMA_V_DEFAULT = 0.005


def parse_sigma_v_from_report(path: Path) -> dict[tuple[str, int], float]:
    """
    Extract per-sensor sigma_V from a lead-brick analysis_report.txt.

    The report lists each sensor as a "[R1_ch3]" block followed by an indented
    "std dev (sample) : <value>" line. Returns {(group, channel): sigma_V}.
    Blocks with "n/a" std (fewer than 2 readings) are skipped.
    """
    text = path.read_text(encoding="utf-8")
    result: dict[tuple[str, int], float] = {}
    current: tuple[str, int] | None = None
    header = re.compile(r"^\[(R\d+)_ch(\d+)\]")
    std_line = re.compile(r"std dev \(sample\)\s*:\s*([0-9eE.+-]+)")
    for line in text.splitlines():
        m = header.match(line.strip())
        if m:
            current = (m.group(1), int(m.group(2)))
            continue
        if current is not None:
            s = std_line.search(line)
            if s:
                try:
                    result[current] = float(s.group(1))
                except ValueError:
                    pass  # "n/a (n<2)" etc. -> leave unset, fall back later
                current = None
    return result


# Load sigma_V from the report at import; fall back to the transcribed dict.
SIGMA_V_SOURCE = "fallback (transcribed)"
try:
    MEASURED_SIGMA_V_BY_SENSOR = parse_sigma_v_from_report(LEAD_BRICK_REPORT)
    if MEASURED_SIGMA_V_BY_SENSOR:
        SIGMA_V_SOURCE = f"parsed from {LEAD_BRICK_REPORT.name}"
    else:
        MEASURED_SIGMA_V_BY_SENSOR = dict(_FALLBACK_SIGMA_V)
except (OSError, FileNotFoundError):
    MEASURED_SIGMA_V_BY_SENSOR = dict(_FALLBACK_SIGMA_V)


def measured_sigma_v(group: str, channel: int) -> float:
    """Per-sensor measurement uncertainty sigma_V [V], with coverage factor."""
    base = MEASURED_SIGMA_V_BY_SENSOR.get((group, channel), SIGMA_V_DEFAULT)
    return base * SIGMA_V_COVERAGE


def shielding_label(channel: int) -> str:
    """Shielding description for a channel (falls back to 'channel N')."""
    return SHIELDING_BY_CHANNEL.get(channel, f"channel {channel}")


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def select_curve_and_dose(dvt: float):
    """
    Convert a threshold-voltage shift dVt [V] to dose [Rad].

    Varadis' fits are nested (0-1 kRad within 0-5 kRad within ... 0-100 kRad)
    and the narrow fits are the most accurate at low dose. We walk the curves
    from narrowest to widest and accept the FIRST one whose own dose estimate
    lands inside its validity range. This is a self-consistent way to honour
    Varadis' "use the curve applicable to the actual dose range" instruction
    without knowing the dose in advance.

    Returns a 3-tuple:
        curve        : the CalibrationCurve used (or None for invalid input)
        dose_rad     : dose in Rad (NaN for invalid input)
        extrapolated : True if dVt is so large that even the 0-100 kRad curve
                       puts the dose beyond 100 kRad (result is an extrapolation)
    """
    if not np.isfinite(dvt) or dvt <= 0:
        return None, float("nan"), False

    last_curve, last_dose = None, float("nan")
    for curve in CALIBRATION_CURVES:  # narrowest -> widest
        dose = curve.dose_from_dvt(dvt)
        last_curve, last_dose = curve, dose
        if dose <= curve.dose_max_rad:
            return curve, dose, False

    # dVt exceeds the widest fit: report the 0-100 kRad estimate, flagged.
    return last_curve, last_dose, True


def dose_uncertainty(curve: CalibrationCurve, dvt: float, sigma_dvt: float) -> float:
    """
    1-sigma dose uncertainty by first-order (linear) error propagation of
    Dose = (dVt / A) ** (1 / B).

    With ln Dose = (1/B)(ln dVt - ln A), the relative dose uncertainty is:

        (sigma_Dose / Dose)^2 =
              ( sigma_dVt / (B * dVt) )^2     <- measurement noise on dVt
            + ( sigma_A  / (B * A) )^2        <- calibration uncertainty in A
            + ( ln(Dose) * sigma_B / B )^2    <- calibration uncertainty in B

    `sigma_dvt` is the 1-sigma uncertainty on the dVt value passed in (e.g. the
    standard error of the mean across a sensor's readings). Returns an absolute
    dose uncertainty in Rad.
    """
    dose = curve.dose_from_dvt(dvt)
    if not np.isfinite(dose) or dose <= 0:
        return float("nan")

    inv_b = 1.0 / curve.B
    term_meas = (inv_b * (sigma_dvt / dvt)) ** 2
    term_a = (inv_b * (curve.sigma_A / curve.A)) ** 2
    term_b = (math.log(dose) * curve.sigma_B / curve.B) ** 2

    rel_dose = math.sqrt(term_meas + term_a + term_b)
    return rel_dose * dose
