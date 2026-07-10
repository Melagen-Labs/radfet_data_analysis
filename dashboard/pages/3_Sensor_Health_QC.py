"""Sensor Health / QC - data quality, gaps, trial agreement, noise floors."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sensor Health / QC", page_icon="🩺", layout="wide")

from core import config, export, pipeline, plots, ui  # noqa: E402
from calibration import SIGMA_V_SOURCE, measured_sigma_v  # noqa: E402

st.title("Sensor Health / QC")

ctx = ui.sidebar()
df, valid = ctx["df"], ctx["df_valid"]
if df.empty:
    st.warning("No readings in the current selection.")
    st.stop()

FLAG_LABELS = {"flag_sat": "saturation", "flag_band": "band",
               "flag_bad_ts": "bad_timestamp", "flag_dup": "duplicate"}

# ------------------------------------------------- per-sensor QC table ---- #
st.subheader("Per-sensor data quality")
rows = []
for (g, ch), sub in df.groupby(["sensor_group", "channel"], observed=True):
    v = sub[sub["valid"]].sort_values("timestamp")
    gaps = v["timestamp"].diff()
    cadence = gaps.median()
    rows.append({
        "sensor": config.sensor_id(g, int(ch)),
        "n_total": len(sub),
        "n_valid": len(v),
        "% valid": 100.0 * len(v) / max(len(sub), 1),
        "saturated": int(sub["flag_sat"].sum()),
        "band-rejected": int(sub["flag_band"].sum()),
        "bad timestamp": int(sub["flag_bad_ts"].sum()),
        "duplicates": int(sub["flag_dup"].sum()),
        "first seen": v["timestamp"].min(),
        "last seen": v["timestamp"].max(),
        "median cadence [s]": cadence.total_seconds() if pd.notna(cadence) else np.nan,
        "longest gap [h]": gaps.max().total_seconds() / 3600 if len(v) > 1 else np.nan,
    })
qc_table = pd.DataFrame(rows)
st.dataframe(qc_table.style.format({
    "% valid": "{:.2f}", "median cadence [s]": "{:.0f}", "longest gap [h]": "{:.1f}",
    "first seen": lambda t: f"{t:%Y-%m-%d %H:%M}", "last seen": lambda t: f"{t:%Y-%m-%d %H:%M}",
}, na_rep="—"), width="stretch", hide_index=True)

# ------------------------------------------------- anomaly timeline ---- #
st.subheader("Anomaly timeline")
invalid = df[~df["valid"]]
plottable = invalid[invalid["timestamp"].notna()]
n_unplottable = int(invalid["timestamp"].isna().sum())
fig = go.Figure()
for flag, label in FLAG_LABELS.items():
    if flag == "flag_bad_ts":
        continue  # cannot be placed on a time axis
    sub = plottable[plottable[flag]]
    if sub.empty:
        continue
    ysensor = [config.sensor_id(g, int(c))
               for g, c in zip(sub["sensor_group"], sub["channel"])]
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=ysensor, mode="markers",
        marker=dict(symbol="square", size=9, color=plots.STATUS[label]),
        name=f"■ {label}",
        hovertemplate="%{x}<br>%{y}<extra>" + label + "</extra>",
    ))
order = [config.sensor_id(g, c) for g in config.TRIAL_GROUPS[::-1]
         for c in config.EXPECTED_CHANNELS[::-1]]
plots.apply_theme(fig, height=380)
fig.update_layout(hovermode="closest")
fig.update_yaxes(categoryorder="array", categoryarray=order)
st.plotly_chart(fig, width="stretch")
st.caption(f"Each square is one rejected reading, colored by cause (status colors "
           f"+ label — never color alone). {n_unplottable:,} rows with corrupted "
           f"timestamps cannot be placed on the time axis and are counted in the "
           f"table above only.")

# ------------------------------------------------- gaps ---- #
st.subheader("Telemetry gaps (> 3× median cadence)")
gap_rows = []
for (g, ch), sub in valid.groupby(["sensor_group", "channel"], observed=True):
    t = sub["timestamp"].sort_values()
    d = t.diff()
    med = d.median()
    if pd.isna(med):
        continue
    big = d > 3 * med
    for end_t, dur in zip(t[big], d[big]):
        gap_rows.append({
            "sensor": config.sensor_id(g, int(ch)),
            "gap start": end_t - dur, "gap end": end_t,
            "duration [h]": dur.total_seconds() / 3600.0,
        })
gaps_table = (pd.DataFrame(gap_rows).sort_values("duration [h]", ascending=False)
              if gap_rows else pd.DataFrame(columns=["sensor", "gap start", "gap end", "duration [h]"]))
st.dataframe(gaps_table.head(30).style.format({"duration [h]": "{:.2f}"}),
             width="stretch", hide_index=True)
if len(gaps_table) > 30:
    st.caption(f"Showing the 30 longest of {len(gaps_table)} gaps — download the "
               "full table below.")

# ------------------------------------------------- R1 vs R2 agreement ---- #
st.subheader("R1 vs R2 trial agreement")
binned = pipeline.binned_series(valid, "1h", ctx["sigma_coverage"])
piv_v = binned.pivot_table(index=["channel", "timestamp"], columns="sensor_group",
                           values="dvt_med", observed=True)
piv_s = binned.pivot_table(index=["channel", "timestamp"], columns="sensor_group",
                           values="sigma_bin", observed=True)
both = piv_v.dropna(subset=["R1", "R2"]) if {"R1", "R2"} <= set(piv_v.columns) else pd.DataFrame()

c1, c2 = st.columns(2)
if not both.empty:
    with c1:
        figs = go.Figure()
        lo = float(min(both["R1"].min(), both["R2"].min()))
        hi = float(max(both["R1"].max(), both["R2"].max()))
        figs.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                  line=dict(color=plots.BASELINE, width=1, dash="dash"),
                                  name="y = x", hoverinfo="skip"))
        for ch in sorted({i[0] for i in both.index}):
            sub = both.xs(ch, level="channel")
            sub = pipeline.downsample_for_plot(sub.reset_index(), 1500)
            figs.add_trace(go.Scatter(
                x=sub["R1"], y=sub["R2"], mode="markers",
                marker=dict(size=5, color=plots.hex_to_rgba(plots.CHANNEL_COLORS[int(ch)], 0.5)),
                name=plots.channel_name(int(ch)),
                hovertemplate="R1 %{x:.4f} V, R2 %{y:.4f} V<extra>ch" + str(ch) + "</extra>",
            ))
        plots.apply_theme(figs, height=430, xtitle="R1 ΔV_t [V] (1 h bins)",
                          ytitle="R2 ΔV_t [V]")
        figs.update_layout(hovermode="closest")
        st.plotly_chart(figs, width="stretch")
    with c2:
        s = piv_s.loc[both.index]
        z = (both["R1"] - both["R2"]) / np.sqrt(s["R1"] ** 2 + s["R2"] ** 2)
        figz = go.Figure()
        for ch in sorted({i[0] for i in z.index}):
            sub = z.xs(ch, level="channel").reset_index()
            sub.columns = ["timestamp", "z"]
            sub = pipeline.downsample_for_plot(sub, 1500)
            figz.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["z"], mode="lines",
                line=dict(color=plots.CHANNEL_COLORS[int(ch)], width=1.5),
                name=plots.channel_name(int(ch)), connectgaps=False,
                hovertemplate="z = %{y:.2f}<extra>ch" + str(ch) + "</extra>",
            ))
        for yv in (-2, 2):
            figz.add_hline(y=yv, line=dict(color=plots.BASELINE, width=1, dash="dash"))
        plots.apply_theme(figz, height=430, ytitle="(R1 − R2) / σ  per 1 h bin")
        st.plotly_chart(figz, width="stretch")
    st.caption("Left: bin-by-bin agreement (points off y = x mean the trials "
               "disagree; a fixed offset is a real R1↔R2 dose difference). Right: "
               "pairwise z — sustained excursions beyond ±2 flag hardware drift "
               "between the two boards. R2 is dosed ≈3% below R1 by design, so a "
               "small systematic offset is expected.")
else:
    st.info("Both R1 and R2 must be selected to compare trials.")

# ------------------------------------------------- noise vs floor ---- #
st.subheader("Measured noise vs lead-brick floor")


@st.cache_data(show_spinner="Computing rolling noise...")
def rolling_noise(valid_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (g, ch), sub in valid_df.groupby(["sensor_group", "channel"], observed=True):
        s = sub.set_index("timestamp")["delta_v"].sort_index()
        detrended = s - s.rolling("6h").median()
        noise = detrended.rolling("6h").std().resample("6h").median()
        parts.append(pd.DataFrame({
            "sensor_group": g, "channel": int(ch),
            "timestamp": noise.index, "noise_v": noise.to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


noise = rolling_noise(valid)
fign = go.Figure()
floor_rows = []
for (g, ch), sub in noise.groupby(["sensor_group", "channel"], observed=True):
    floor = measured_sigma_v(g, int(ch)) * ctx["sigma_coverage"]
    current = float(sub["noise_v"].dropna().iloc[-1]) if sub["noise_v"].notna().any() else np.nan
    floor_rows.append({
        "sensor": config.sensor_id(g, int(ch)),
        "σ_V floor [mV]": floor * 1000,
        "current noise [mV]": current * 1000,
        "ratio": current / floor if floor > 0 else np.nan,
        "status": "⚠ elevated" if np.isfinite(current) and current > 1.5 * floor else "ok",
    })
    fign.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["noise_v"] * 1000, mode="lines",
        line=dict(color=plots.CHANNEL_COLORS[int(ch)], width=1.5,
                  dash=plots.GROUP_DASH.get(str(g), "solid")),
        name=f"{g} ch{ch}", connectgaps=False,
        hovertemplate="%{y:.2f} mV<extra>" + f"{g} ch{ch}</extra>",
    ))
floor_mean = np.mean([measured_sigma_v(g, c) for g, c in config.ALL_SENSORS]) * 1000
fign.add_hline(y=floor_mean * ctx["sigma_coverage"],
               line=dict(color=plots.BASELINE, width=1, dash="dash"),
               annotation_text="mean lead-brick floor",
               annotation_font=dict(size=11, color=plots.MUTED))
plots.apply_theme(fign, height=400, ytitle="rolling 6 h noise [mV]")
st.plotly_chart(fign, width="stretch")

noise_table = pd.DataFrame(floor_rows)
st.dataframe(noise_table.style.format({
    "σ_V floor [mV]": "{:.2f}", "current noise [mV]": "{:.2f}", "ratio": "{:.2f}",
}, na_rep="—"), width="stretch", hide_index=True)
st.caption(f"Noise = rolling 6 h std of detrended ΔV_t (dose trend removed by a "
           f"rolling median). Floor = per-sensor σ_V from the lead-brick "
           f"characterization — source: {SIGMA_V_SOURCE}. Ratio > 1.5 flags a "
           f"sensor noisier than characterized.")

st.divider()
d1, d2, d3 = st.columns(3)
with d1:
    export.download_csv(qc_table, "qc_per_sensor.csv", "QC table", key="dl_qc")
with d2:
    inv = invalid[["timestamp", "sensor_group", "channel", "raw_adc", "raw_voltage",
                   "delta_v", "flag_band", "flag_sat", "flag_bad_ts", "flag_dup"]]
    export.download_csv(inv, "invalid_rows.csv", "Invalid rows", key="dl_inv")
with d3:
    export.download_csv(gaps_table, "telemetry_gaps.csv", "Gaps table", key="dl_gaps")
