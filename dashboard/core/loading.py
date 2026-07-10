"""
loading.py

Data loading + the canonical raw_adc -> delta_v conversion + QC flagging.

The incoming CSV `voltage` / `delta_voltage_v` / `dose_rad` columns are never
used: delta_v is always recomputed from raw_adc with the session's conversion
settings (see config.py docstring). QC rules mirror analyze_sample.py.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import config, paths


def ensure_simulation_data() -> None:
    """Generate the simulated mission on first access (deterministic seed),
    so a fresh clone / Community Cloud deploy works without committing 25 MB."""
    if paths.SIM_MISSION_CSV.exists():
        return
    sim_dir = str(paths.SIMULATION_DIR)
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    with st.spinner("First run: generating simulated ISS mission data (~30 s)..."):
        gen = importlib.import_module("generate_iss_mission")
        gen.main()


def _file_signature(source_key: str) -> tuple:
    """(path, mtime, size) per file so the cache invalidates on data changes."""
    kind, path = config.DATA_SOURCES[source_key]
    files = [Path(path)] if kind == "file" else sorted(Path(path).glob("*.csv"))
    return tuple((str(f), f.stat().st_mtime, f.stat().st_size) for f in files)


@st.cache_data(show_spinner="Loading telemetry...")
def _load_raw(source_key: str, file_sig: tuple) -> pd.DataFrame:
    kind, path = config.DATA_SOURCES[source_key]
    files = [Path(path)] if kind == "file" else sorted(Path(path).glob("*.csv"))
    frames = []
    for f in files:
        part = pd.read_csv(f)
        part["source_file"] = f.name
        frames.append(part)
    return _normalize(pd.concat(frames, ignore_index=True))


@st.cache_data(show_spinner="Reading uploaded file...")
def _load_uploaded(content: bytes, name: str) -> pd.DataFrame:
    import io

    df = pd.read_csv(io.BytesIO(content))
    df["source_file"] = name
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Parse/clean the schema-independent part: timestamps, dtypes, ordering."""
    required = {"timestamp", "sensor_group", "channel", "raw_adc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"], errors="coerce", format="mixed"),
        "sensor_group": df["sensor_group"].astype(str),
        "channel": pd.to_numeric(df["channel"], errors="coerce").astype("Int64"),
        "raw_adc": pd.to_numeric(df["raw_adc"], errors="coerce"),
    })
    if "source_file" in df.columns:
        out["source_file"] = df["source_file"]

    bad_ts = out["timestamp"].notna() & (
        (out["timestamp"] < config.TIMESTAMP_VALID_MIN)
        | (out["timestamp"] > config.TIMESTAMP_VALID_MAX)
    )
    out.loc[bad_ts, "timestamp"] = pd.NaT
    out = out.dropna(subset=["channel"])
    out["channel"] = out["channel"].astype(np.int16)
    # Stable sort keeps duplicated rows adjacent; NaT (bad timestamps) sort last.
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def apply_conversion(raw: pd.DataFrame, baseline_r1: float, baseline_r2: float,
                     board_offset: float) -> pd.DataFrame:
    """Recompute delta_v from raw_adc and flag validity.

        raw_voltage = raw_adc * (5.0 / 4095)
        delta_v     = raw_voltage - baseline[group] - board_offset

    Flags are kept separate so the QC page can slice by cause:
        flag_band   : delta_v non-finite or outside (0, 4] V
        flag_sat    : raw_adc >= 4090 (ADC saturation)
        flag_bad_ts : timestamp missing/implausible (e.g. 1969 epoch)
        flag_dup    : exact duplicate of an earlier reading
    """
    df = raw.copy()
    baseline = df["sensor_group"].map(
        {"R1": baseline_r1, "R2": baseline_r2}
    ).fillna(baseline_r1)
    df["raw_voltage"] = df["raw_adc"] * (config.ADC_VREF_V / config.ADC_FULL_SCALE)
    df["delta_v"] = df["raw_voltage"] - baseline - board_offset

    dv = df["delta_v"].to_numpy()
    df["flag_band"] = ~(
        np.isfinite(dv)
        & (dv > config.VOLTAGE_VALID_MIN)
        & (dv <= config.VOLTAGE_VALID_MAX)
    )
    df["flag_sat"] = df["raw_adc"] >= config.SATURATION_ADC
    df["flag_bad_ts"] = df["timestamp"].isna()
    df["flag_dup"] = df.duplicated(
        subset=["timestamp", "sensor_group", "channel", "raw_adc"], keep="first"
    )
    df["valid"] = ~(df["flag_band"] | df["flag_sat"] | df["flag_bad_ts"] | df["flag_dup"])
    return df


def load_telemetry(source_key: str, baseline_r1: float, baseline_r2: float,
                   board_offset: float,
                   uploaded=None) -> pd.DataFrame:
    """Main entry: load a registered source (or an uploaded file) and convert."""
    if uploaded is not None:
        raw = _load_uploaded(uploaded.getvalue(), uploaded.name)
    else:
        if source_key == config.DEFAULT_SOURCE:
            ensure_simulation_data()
        raw = _load_raw(source_key, _file_signature(source_key))
    return apply_conversion(raw, baseline_r1, baseline_r2, board_offset)


def qc_summary(df: pd.DataFrame) -> dict:
    return {
        "n_total": len(df),
        "n_valid": int(df["valid"].sum()),
        "n_band": int(df["flag_band"].sum()),
        "n_sat": int(df["flag_sat"].sum()),
        "n_bad_ts": int(df["flag_bad_ts"].sum()),
        "n_dup": int(df["flag_dup"].sum()),
    }
