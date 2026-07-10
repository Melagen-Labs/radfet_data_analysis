"""Dose vs Time - the headline explorer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dose vs Time", page_icon="📈", layout="wide")

from core import config, export, pipeline, plots, ui  # noqa: E402

st.title("Dose vs Time")

ctx = ui.sidebar()
valid = ctx["df_valid"]
if valid.empty:
    st.warning("No valid readings in the current selection.")
    st.stop()

span_days = (ctx["t1"] - ctx["t0"]).total_seconds() / 86400.0

c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.4, 1])
window_label = c1.selectbox("Resample window", list(config.WINDOW_OPTIONS),
                            index=list(config.WINDOW_OPTIONS).index(config.DEFAULT_WINDOW))
view = c2.radio("View", ["Cumulative dose", "Dose rate"], horizontal=True)
mode = c3.radio("Series", ["R1 / R2 separate", "R1+R2 combined"], horizontal=True)
yscale = c4.radio("Y scale", ["linear", "log"], horizontal=True)

c5, c6, c7, c8 = st.columns([1.6, 1.2, 1.6, 1.6])
band_choice = c5.selectbox(
    "Uncertainty band",
    ["none", "measurement only", "full (incl. calibration)"],
    index=2 if view == "Cumulative dose" else 0,
    help="Measurement-only: sensor noise floor. Full adds the Varadis "
         "calibration σ_A/σ_B terms (common across channels).")
monotonic = c6.checkbox("Enforce non-decreasing dose (cummax)", value=False,
                        help="Physical dose is monotonic; noise and fade are not. Off = show the data as measured.")
transitions = c7.checkbox("Mark calibration-curve transitions", value=True)
overlay_raw = c8.checkbox("Overlay per-reading scatter", value=False,
                          disabled=span_days > 7,
                          help="Per-reading delta_v converted point-by-point. "
                               "Only available when the visible span is < 7 days.")

freq = config.WINDOW_OPTIONS[window_label]
band = {"none": None, "measurement only": "meas", "full (incl. calibration)": "full"}[band_choice]

binned = pipeline.binned_series(valid, freq, ctx["sigma_coverage"])
if mode == "R1+R2 combined":
    series = pipeline.combined_series(binned)
else:
    series = binned
series = pipeline.add_dose_rate(series)

ycol = "dose_rad" if view == "Cumulative dose" else "rate_rad_h"
ytitle = "Cumulative dose [Rad]" if view == "Cumulative dose" else "Dose rate [Rad/h]"

fig = go.Figure()
for (grp, ch), sub in series.groupby(["sensor_group", "channel"], observed=True):
    sub = sub.copy()
    if monotonic and view == "Cumulative dose":
        sub["dose_rad"] = sub["dose_rad"].cummax()
    plots.add_series(
        fig, pipeline.downsample_for_plot(sub), ycol,
        channel=int(ch), group=str(grp),
        band=band if view == "Cumulative dose" else None,
        hover_fmt=":.1f" if view == "Cumulative dose" else ":.2f",
        yunit="Rad" if view == "Cumulative dose" else "Rad/h",
    )

if view == "Dose rate":
    med = float(series["rate_rad_h"].median())
    if np.isfinite(med):
        fig.add_hline(y=med, line=dict(color=plots.BASELINE, width=1, dash="dash"),
                      annotation_text=f"median {med:.2f} Rad/h",
                      annotation_font=dict(size=11, color=plots.MUTED))

if transitions and view == "Cumulative dose":
    seen_labels = set()
    for (grp, ch), sub in series.groupby(["sensor_group", "channel"], observed=True):
        sub = sub.dropna(subset=["dose_rad"]).sort_values("timestamp")
        change = sub["curve_name"].ne(sub["curve_name"].shift()) & sub["curve_name"].shift().notna()
        for _, row in sub[change].iterrows():
            fig.add_vline(x=row["timestamp"],
                          line=dict(color=plots.MUTED, width=1, dash="dot"))
            label = f"→ {row['curve_name']}"
            if label not in seen_labels:
                seen_labels.add(label)
                fig.add_annotation(x=row["timestamp"], yref="paper", y=1.02,
                                   showarrow=False, text=label,
                                   font=dict(size=10, color=plots.MUTED))

if overlay_raw and span_days <= 7:
    conv = pipeline.vector_dose(valid["delta_v"].to_numpy(),
                                np.zeros(len(valid)))
    raw_pts = valid[["timestamp", "sensor_group", "channel"]].copy()
    raw_pts["dose_rad"] = conv["dose_rad"].to_numpy()
    if view == "Cumulative dose":
        for (grp, ch), sub in raw_pts.groupby(["sensor_group", "channel"], observed=True):
            sub = pipeline.downsample_for_plot(sub, 3000)
            fig.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["dose_rad"], mode="markers",
                marker=dict(size=3, color=plots.hex_to_rgba(
                    plots.CHANNEL_COLORS[int(ch)], 0.25)),
                name=f"{grp} ch{ch} raw", showlegend=False, hoverinfo="skip",
            ))

plots.apply_theme(fig, height=560, ytitle=ytitle)
if yscale == "log":
    fig.update_yaxes(type="log")
st.plotly_chart(fig, width="stretch")

captions = [ui.settings_caption(ctx)]
if view == "Dose rate":
    captions.insert(0, "Rate uncertainty uses the measurement-only dose σ: "
                       "calibration error is common-mode between neighbouring bins "
                       "and cancels in the difference.")
if mode == "R1 / R2 separate":
    captions.insert(0, "R1 solid, R2 dashed; color identifies the channel.")
captions.insert(0, "Open circles mark bins beyond the 0-100 kRad calibration "
                   "range (extrapolated). Gaps in the line are telemetry gaps.")
st.caption(" · ".join(captions))

with st.expander("Voltage space — binned ΔV_t vs time (the raw measurement, pre-calibration)"):
    figv = go.Figure()
    for (grp, ch), sub in binned.groupby(["sensor_group", "channel"], observed=True):
        sub = pipeline.downsample_for_plot(sub)
        color = plots.CHANNEL_COLORS[int(ch)]
        figv.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["dvt_med"], mode="lines",
            line=dict(color=color, width=2, dash=plots.GROUP_DASH.get(str(grp), "solid")),
            name=f"{grp} {plots.channel_name(int(ch))}", connectgaps=False,
            hovertemplate="%{y:.4f} V<extra>" + f"{grp} ch{ch}</extra>",
        ))
    plots.apply_theme(figv, height=420, ytitle="ΔV_t [V] (bin median)")
    st.plotly_chart(figv, width="stretch")
    st.caption(ui.settings_caption(ctx))

st.divider()
d1, d2 = st.columns(2)
with d1:
    export.download_csv(
        pipeline.add_dose_rate(series).drop(columns=["w", "wx"], errors="ignore"),
        f"dose_series_{freq}.csv",
        f"Binned series ({window_label} bins, as plotted)", key="dl_binned")
with d2:
    conv = pipeline.vector_dose(valid["delta_v"].to_numpy(), np.zeros(len(valid)))
    per_reading = valid[["timestamp", "sensor_group", "channel", "raw_adc",
                         "raw_voltage", "delta_v"]].reset_index(drop=True)
    per_reading = pd.concat([per_reading, conv[["dose_rad", "curve_name"]]], axis=1)
    export.download_csv(per_reading, "per_reading_visible_range.csv",
                        "Per-reading converted data (visible range)", key="dl_raw")
