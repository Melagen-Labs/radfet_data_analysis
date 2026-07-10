"""Calibration & Methodology - static reference + live conversion demo."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Calibration & Methodology", page_icon="📐", layout="wide")

from core import config, export, paths, pipeline, plots, ui  # noqa: E402
import calibration  # noqa: E402
from calibration import (  # noqa: E402
    CALIBRATION_CURVES,
    SIGMA_V_SOURCE,
    dose_uncertainty,
    measured_sigma_v,
    select_curve_and_dose,
)

st.title("Calibration & Methodology")

ctx = ui.sidebar()

# ------------------------------------------------- Varadis curves ---- #
st.subheader("Varadis QF 16 calibration curves — dVt = A · Dose^B")
c1, c2 = st.columns([1.5, 1])
with c1:
    st.plotly_chart(plots.calibration_curves_figure(), width="stretch")
    st.caption("Solid: each fit over its validity range; dotted: the same fit "
               "extrapolated. Conversion always uses the narrowest curve whose "
               "own dose estimate falls inside its range.")
with c2:
    coeff = pd.DataFrame([{
        "curve": c.name, "dose max [Rad]": c.dose_max_rad,
        "A": c.A, "σ_A": c.sigma_A, "B": c.B, "σ_B": c.sigma_B, "R²": c.r_square,
    } for c in CALIBRATION_CURVES])
    st.dataframe(coeff.style.format({
        "dose max [Rad]": "{:,.0f}", "A": "{:.4f}", "σ_A": "{:.2e}",
        "B": "{:.4f}", "σ_B": "{:.2e}", "R²": "{:.3f}",
    }), width="stretch", hide_index=True)
    st.caption("Source: Varadis “QF 16 RADFET Test Data Record”, Issue 01 "
               "(01/08/2023). 400 nm IMPL RADFET, COMRAD mask-set, Co-60 at "
               "5.0 kRad/h, all pins grounded. Coefficients are read from "
               "sample_analysis/calibration.py — the single source of truth.")
    export.download_csv(coeff, "calibration_coefficients.csv",
                        "Coefficients CSV", key="dl_coeff")

# ------------------------------------------------- live converter ---- #
st.subheader("Interactive ΔV_t → dose converter")
cc1, cc2, cc3 = st.columns([1, 1, 2])
dvt_in = cc1.number_input("ΔV_t [V]", value=0.500, min_value=0.0, max_value=4.0,
                          step=0.05, format="%.4f")
sig_in = cc2.number_input("σ_ΔVt [V]", value=0.005, min_value=0.0, max_value=0.5,
                          step=0.001, format="%.4f")
curve, dose, extrapolated = select_curve_and_dose(dvt_in)
with cc3:
    if curve is None:
        st.info("ΔV_t must be positive and finite.")
    else:
        sigma_full = dose_uncertainty(curve, dvt_in, sig_in)
        inv_b = 1.0 / curve.B
        t_meas = (inv_b * (sig_in / dvt_in)) ** 2
        t_a = (inv_b * (curve.sigma_A / curve.A)) ** 2
        t_b = (math.log(dose) * curve.sigma_B / curve.B) ** 2
        st.metric(f"Dose (curve: {curve.name}{' — EXTRAPOLATED' if extrapolated else ''})",
                  f"{dose:,.1f} Rad", f"± {sigma_full:,.1f} (1σ full)", delta_color="off")
        tot = t_meas + t_a + t_b
        st.caption(f"Error budget: measurement {100 * t_meas / tot:.0f}% · "
                   f"calibration A {100 * t_a / tot:.0f}% · "
                   f"calibration B {100 * t_b / tot:.0f}%")

# ------------------------------------------------- constants ---- #
st.subheader("Conversion settings & constants")
st.markdown(
    "**The conversion applied to every reading:**\n"
    "```\n"
    "raw_voltage = raw_adc × (5.0 / 4095)\n"
    "delta_v     = raw_voltage − baseline(sensor group) − board offset\n"
    "```"
)
st.warning(
    "The CSV `voltage` / `delta_voltage_v` / `dose_rad` columns are **never "
    "used**: the baseline subtracted to produce them changed inconsistently "
    "over time. raw_adc is the untouched measurement; everything is recomputed "
    "from it with the settings below."
)
set_tbl = pd.DataFrame([
    {"setting": "R1 baseline", "active (this session)": f"{ctx['baseline_r1']:.3f} V",
     "default": f"{config.DV_BASELINE_BY_GROUP['R1']:.3f} V"},
    {"setting": "R2 baseline", "active (this session)": f"{ctx['baseline_r2']:.3f} V",
     "default": f"{config.DV_BASELINE_BY_GROUP['R2']:.3f} V"},
    {"setting": "Board offset voltage", "active (this session)": f"{ctx['board_offset']:.3f} V",
     "default": f"{config.BOARD_OFFSET_V:.3f} V"},
    {"setting": "ADC reference / full scale",
     "active (this session)": f"{config.ADC_VREF_V:.1f} V / {config.ADC_FULL_SCALE}",
     "default": "5.0 V / 4095 (12-bit)"},
    {"setting": "Valid ΔV_t band",
     "active (this session)": f"({config.VOLTAGE_VALID_MIN}, {config.VOLTAGE_VALID_MAX}] V",
     "default": "(0, 4] V"},
    {"setting": "Saturation threshold",
     "active (this session)": f"raw_adc ≥ {config.SATURATION_ADC}",
     "default": "raw_adc ≥ 4090"},
    {"setting": "σ_V coverage factor", "active (this session)": f"×{ctx['sigma_coverage']:g}",
     "default": "×1.0"},
])
if (ctx["baseline_r1"] != config.DV_BASELINE_BY_GROUP["R1"]
        or ctx["baseline_r2"] != config.DV_BASELINE_BY_GROUP["R2"]
        or ctx["board_offset"] != config.BOARD_OFFSET_V):
    st.error("⚠ The active conversion settings differ from the defaults — every "
             "chart and table in this session uses the ACTIVE values shown below.")
st.dataframe(set_tbl, width="stretch", hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Per-sensor measurement noise floor σ_V** (lead-brick run)")
    sig_tbl = pd.DataFrame([
        {"sensor": config.sensor_id(g, ch), "σ_V [mV]": measured_sigma_v(g, ch) * 1000}
        for g, ch in config.ALL_SENSORS
    ])
    st.dataframe(sig_tbl.style.format({"σ_V [mV]": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.caption(f"σ_V source: **{SIGMA_V_SOURCE}**. (calibration.py looks for "
               f"sensor_analysis/analysis_report.txt, which does not exist — the "
               f"transcribed fallback matches lead_brick_analysis/analysis_report.txt.)")
with c2:
    st.markdown("**Channel → shielding map**")
    shield_tbl = pd.DataFrame([
        {"channel": ch, "shielding": config.shield_label(ch),
         "role": "reference (bare)" if ch == config.UNSHIELDED_CHANNEL else ""}
        for ch in config.EXPECTED_CHANNELS
    ])
    st.dataframe(shield_tbl, width="stretch", hide_index=True)
    st.warning(
        "⚠ The ch3–ch5 mapping is inconsistent across project files: "
        "calibration.py says ch3 = MLC1-b + Al, ch4 = MLC2, ch5 = MLC1, while the "
        "legacy fake-data comments and README say ch3 = MLC1, ch4 = MLC1-b + Al, "
        "ch5 = MLC2. The dashboard follows calibration.py; confirm against the "
        "flight configuration and override in dashboard/core/config.py "
        "(SHIELDING_BY_CHANNEL_OVERRIDE) if needed."
    )

# ------------------------------------------------- pipeline description ---- #
st.subheader("Analysis pipeline")
st.markdown(
    "1. **Load** telemetry; null implausible timestamps (outside 2000–2100).\n"
    "2. **Recompute** delta_v from raw_adc with the active settings (above).\n"
    "3. **QC flags**: band (ΔV_t ∉ (0, 4] V), saturation (raw_adc ≥ 4090), bad "
    "timestamp, exact duplicate. Statistics use valid readings only.\n"
    "4. **Bin in voltage space** (median per window; σ_bin = σ_V·√(π/2)/√n — the "
    "√(π/2) is the median's efficiency penalty vs the mean under Gaussian noise). "
    "Smoothing before conversion matters because per-reading dose at low ΔV_t is "
    "dominated by the ~5 mV noise floor.\n"
    "5. **Convert** each bin with narrow→wide curve selection "
    "(calibration.select_curve_and_dose) and propagate σ "
    "(calibration.dose_uncertainty: measurement ⊕ calibration A ⊕ calibration B).\n"
    "6. **Combine R1+R2** per bin by inverse-variance weighting in voltage space.\n"
    "7. **Significance** (shielding, drift) is always tested in voltage space, "
    "where calibration error is common-mode and cancels; dose-ratio CIs use "
    "Monte Carlo with the calibration sampled once per draw, shared across "
    "channels."
)

# ------------------------------------------------- simulator self-check ---- #
if ctx["source"] == config.DEFAULT_SOURCE and paths.SIM_GROUND_TRUTH_CSV.exists():
    with st.expander("Simulator self-check — recovered dose vs ground truth"):
        truth = pd.read_csv(paths.SIM_GROUND_TRUTH_CSV, parse_dates=["date"])
        valid = ctx["df_valid"]
        binned = pipeline.binned_series(valid, "1D", ctx["sigma_coverage"])
        comb = pipeline.combined_series(binned)
        figt = go.Figure()
        for ch in sorted(comb["channel"].unique()):
            sub = comb[comb["channel"] == ch]
            plots.add_series(figt, sub, "dose_rad", channel=int(ch), group="R1+R2",
                             band=None, extrapolated_markers=False)
            tr = (truth[truth["channel"] == ch]
                  .groupby("date")["true_cum_dose_rad"].mean().reset_index())
            figt.add_trace(go.Scatter(
                x=tr["date"], y=tr["true_cum_dose_rad"], mode="lines",
                line=dict(color=plots.INK_SECONDARY, width=1, dash="dot"),
                name=f"truth ch{ch}", showlegend=False,
                hovertemplate="truth %{y:.0f} Rad<extra>ch" + str(ch) + "</extra>",
            ))
        plots.apply_theme(figt, height=420, ytitle="Cumulative dose [Rad]")
        st.plotly_chart(figt, width="stretch")
        st.caption("Colored lines: dashboard-recovered combined dose (1 d bins). "
                   "Dotted gray: the simulator's true cumulative dose (R1/R2 "
                   "averaged). Agreement validates the full raw_adc → dose chain.")
