"""
paths.py

Project path constants and the sys.path bootstrap that makes the existing
analysis modules importable.

`sample_analysis/` is deliberately NOT a package (its scripts import each other
flat, e.g. `from calibration import ...`), so instead of packaging it we insert
it into sys.path and import its modules top-level (`import calibration`).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_ANALYSIS_DIR = PROJECT_ROOT / "sample_analysis"
AMBIENT_ANALYSIS_DIR = PROJECT_ROOT / "ambient_analysis"
LEAD_BRICK_REPORT = PROJECT_ROOT / "lead_brick_analysis" / "analysis_report.txt"

SIMULATION_DIR = PROJECT_ROOT / "simulation"
SIM_OUTPUT_DIR = SIMULATION_DIR / "output"
SIM_MISSION_CSV = SIM_OUTPUT_DIR / "radfet_iss_mission.csv"
SIM_GROUND_TRUTH_CSV = SIM_OUTPUT_DIR / "ground_truth.csv"
SIM_MANIFEST_JSON = SIM_OUTPUT_DIR / "simulation_manifest.json"

HISTORIC_DATA_DIR = PROJECT_ROOT / "raw_data" / "historic_data"


def bootstrap() -> None:
    """Make `import calibration` / `import analyze_sample` work everywhere."""
    for p in (SAMPLE_ANALYSIS_DIR,):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
