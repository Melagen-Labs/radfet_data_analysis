"""
ui.py

The shared sidebar: data source, the voltage-conversion settings (always
visible and editable - a hard requirement: scientists must see exactly which
baseline/offset produced every delta_v), sensor and date-range filters.
Widget state lives in st.session_state so selections persist across pages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import config, loading

_DEFAULTS = {
    "set_baseline_r1": config.DV_BASELINE_BY_GROUP["R1"],
    "set_baseline_r2": config.DV_BASELINE_BY_GROUP["R2"],
    "set_board_offset": config.BOARD_OFFSET_V,
    "set_sigma_coverage": 1.0,
}


def _reset_settings() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


def _init_state() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("set_source", config.DEFAULT_SOURCE)


def sidebar() -> dict:
    """Render the shared sidebar and return {df, df_valid, settings...}.
    `df` is filtered to the selected sensors/date range; rows with a bad
    timestamp are kept (they cannot be placed in time but the QC page needs
    them). `df_valid` additionally keeps only QC-valid readings."""
    _init_state()

    with st.sidebar:
        st.markdown("### Data")
        options = list(config.DATA_SOURCES) + [config.UPLOAD_SOURCE]
        source = st.selectbox("Data source", options, key="set_source")
        uploaded = None
        if source == config.UPLOAD_SOURCE:
            uploaded = st.file_uploader(
                "CSV with timestamp, sensor_group, channel, raw_adc",
                type="csv",
            )
            if uploaded is None:
                st.info("Upload a telemetry CSV to continue.")
                st.stop()

        st.markdown("### Voltage conversion")
        st.caption(
            "delta_v = raw_adc × (5.0 / 4095) − baseline − board offset. "
            "The CSV's own voltage columns are ignored (untrusted baselines); "
            "everything is recomputed from raw_adc."
        )
        b_r1 = st.number_input("R1 baseline [V]", key="set_baseline_r1",
                               min_value=0.0, max_value=5.0, step=0.01, format="%.3f")
        b_r2 = st.number_input("R2 baseline [V]", key="set_baseline_r2",
                               min_value=0.0, max_value=5.0, step=0.01, format="%.3f")
        v_off = st.number_input("Board offset voltage [V]", key="set_board_offset",
                                min_value=-1.0, max_value=1.0, step=0.01, format="%.3f",
                                help="Series offset of the readout board (current "
                                     "hardware setting: 0.100 V).")
        coverage = st.number_input("σ_V coverage factor", key="set_sigma_coverage",
                                   min_value=0.5, max_value=5.0, step=0.5, format="%.1f",
                                   help="Multiplier on the per-sensor lead-brick "
                                        "noise floor σ_V (1.0 = raw 1σ).")
        st.button("Reset to defaults", on_click=_reset_settings, width="stretch")

    df_all = loading.load_telemetry(source, b_r1, b_r2, v_off, uploaded=uploaded)

    with st.sidebar:
        st.markdown("### Filters")
        sensors = st.multiselect(
            "Sensors",
            options=config.ALL_SENSORS,
            default=config.ALL_SENSORS,
            format_func=lambda s: f"{s[0]} ch{s[1]} ({config.shield_label(s[1])})",
            key="set_sensors",
        )
        ts = df_all["timestamp"].dropna()
        if ts.empty:
            st.error("No usable timestamps in this data source.")
            st.stop()
        tmin, tmax = ts.min().to_pydatetime(), ts.max().to_pydatetime()
        t0, t1 = st.slider("Date range", min_value=tmin, max_value=tmax,
                           value=(tmin, tmax), format="MMM D", key="set_daterange")
        st.caption(
            f"Active settings — R1: {b_r1:.3f} V · R2: {b_r2:.3f} V · "
            f"offset: {v_off:.3f} V · σ×{coverage:g}"
        )

    keys = set(map(tuple, sensors)) if sensors else set(map(tuple, config.ALL_SENSORS))
    sensor_mask = [
        (g, int(c)) in keys
        for g, c in zip(df_all["sensor_group"], df_all["channel"])
    ]
    in_range = df_all["timestamp"].between(t0, t1) | df_all["timestamp"].isna()
    df = df_all[pd.Series(sensor_mask, index=df_all.index) & in_range]

    return {
        "df": df,
        "df_valid": df[df["valid"]],
        "df_all": df_all,
        "source": source,
        "baseline_r1": b_r1,
        "baseline_r2": b_r2,
        "board_offset": v_off,
        "sigma_coverage": coverage,
        "t0": pd.Timestamp(t0),
        "t1": pd.Timestamp(t1),
        "channels": sorted({int(c) for _, c in (sensors or config.ALL_SENSORS)}),
        "groups": sorted({g for g, _ in (sensors or config.ALL_SENSORS)}),
    }


def settings_caption(ctx: dict) -> str:
    """One-line provenance caption placed under every delta_v/dose chart."""
    return (
        f"Conversion: delta_v = raw_adc × (5.0/4095) − baseline "
        f"(R1 {ctx['baseline_r1']:.3f} V, R2 {ctx['baseline_r2']:.3f} V) − "
        f"board offset {ctx['board_offset']:.3f} V · "
        f"σ_V coverage ×{ctx['sigma_coverage']:g} · "
        f"CSV voltage columns ignored, recomputed from raw_adc"
    )
