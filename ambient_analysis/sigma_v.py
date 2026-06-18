"""
sigma_v.py

Parse the per-sensor measurement uncertainty sigma_V [V] from a lead-brick
analysis_report.txt. These are the "std dev (sample)" figures reported for each
sensor in the lead-enclosure stability run -- the characterized per-sensor noise
floor -- and they are the measurement uncertainty used in the ambient shift test.

The numbers are READ DIRECTLY from the report so they stay in sync whenever the
lead-brick analysis is re-run. A transcribed fallback is used only if the report
cannot be found or parsed (same values, kept here for resilience).

This mirrors the parser in ../sample_analysis/calibration.py so both analyses
consume the lead-brick sigma_V identically.
"""

from __future__ import annotations

import re
from pathlib import Path

# Transcribed fallback (matches lead_brick_analysis/analysis_report.txt). Used
# only when the report is unavailable/unparseable.
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

_HEADER = re.compile(r"^\[(R\d+)_ch(\d+)\]")
_STD_LINE = re.compile(r"std dev \(sample\)\s*:\s*([0-9eE.+-]+)")


def _parse_report(path: Path) -> dict[tuple[str, int], float]:
    """
    Extract {(group, channel): sigma_V} from a lead-brick analysis_report.txt.

    Each sensor is a "[R1_ch3]" header followed by an indented
    "std dev (sample) : <value>" line. Blocks with a non-numeric std
    (e.g. "n/a (n<2)") are skipped so the caller can fall back for them.
    """
    text = path.read_text(encoding="utf-8")
    result: dict[tuple[str, int], float] = {}
    current: tuple[str, int] | None = None
    for line in text.splitlines():
        m = _HEADER.match(line.strip())
        if m:
            current = (m.group(1), int(m.group(2)))
            continue
        if current is not None:
            s = _STD_LINE.search(line)
            if s:
                try:
                    result[current] = float(s.group(1))
                except ValueError:
                    pass  # "n/a (n<2)" etc. -> leave unset, fall back later.
                current = None
    return result


def parse_sigma_v_from_report(path: Path) -> tuple[dict[tuple[str, int], float], str]:
    """
    Return (sigma_v_by_sensor, source_description). Falls back to the transcribed
    dict if the report is missing or yields nothing.
    """
    try:
        parsed = _parse_report(path)
    except (OSError, FileNotFoundError):
        parsed = {}

    if parsed:
        # Backfill any sensor the report didn't cover from the fallback.
        merged = dict(_FALLBACK_SIGMA_V)
        merged.update(parsed)
        return merged, f"parsed from {path.name}"
    return dict(_FALLBACK_SIGMA_V), "fallback (transcribed)"
