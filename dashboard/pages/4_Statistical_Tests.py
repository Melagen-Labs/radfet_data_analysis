"""Statistical Tests - window shift tests, trend fits, periodicity."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Statistical Tests", page_icon="🧪", layout="wide")

from core import config, export, pipeline, plots, stats, ui  # noqa: E402

st.title("Statistical Tests")

ctx = ui.sidebar()
valid = ctx["df_valid"]
if valid.empty:
    st.warning("No valid readings in the current selection.")
    st.stop()

t_min, t_max = ctx["t0"], ctx["t1"]

# ---------------------------------------------- before / after shift ---- #
st.subheader("Before / after voltage-shift test")
st.caption("Per sensor: shift = mean ΔV_t(after) − mean ΔV_t(before); "
           "z = shift / (σ_V·√(1/n_b + 1/n_a)). Collective: exact sign test + "
           "Stouffer combined Z across sensors (the ambient-analysis methodology "
           "on windows of your choosing).")

c1, c2 = st.columns(2)
with c1:
    b_range = st.date_input("Before period",
                            value=(t_min.date(), (t_min + pd.Timedelta(days=3)).date()),
                            min_value=t_min.date(), max_value=t_max.date())
with c2:
    a_range = st.date_input("After period",
                            value=((t_max - pd.Timedelta(days=3)).date(), t_max.date()),
                            min_value=t_min.date(), max_value=t_max.date())

if len(b_range) == 2 and len(a_range) == 2:
    before = (pd.Timestamp(b_range[0]), pd.Timestamp(b_range[1]) + pd.Timedelta(days=1))
    after = (pd.Timestamp(a_range[0]), pd.Timestamp(a_range[1]) + pd.Timedelta(days=1))
    table, coll = stats.shift_analysis(valid, before, after, ctx["sigma_coverage"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensors shifted up / down", f"{coll['n_up']} / {coll['n_down']}")
    m2.metric("Sign test p", f"{coll['sign_p']:.3g}" if np.isfinite(coll["sign_p"]) else "n/a")
    m3.metric("Stouffer Z", f"{coll['stouffer_z']:.2f}" if np.isfinite(coll["stouffer_z"]) else "n/a")
    m4.metric("Sensors > 3σ", f"{coll['n_sig3']}/{coll['n_usable']}")

    st.dataframe(table.style.format({
        "before_v": "{:.4f}", "after_v": "{:.4f}", "shift_v": "{:+.4f}",
        "sigma_v": "{:.4f}", "sigma_shift_v": "{:.5f}",
        "z": "{:.2f}", "p": "{:.2e}",
    }, na_rep="—"), width="stretch", hide_index=True)
    st.caption(ui.settings_caption(ctx))

    with st.expander("Window-length sensitivity (does the verdict depend on how much data you average?)"):
        lengths = ["30min", "1h", "3h", "6h", "12h", "1D", "3D"]
        sens_rows = []
        for w in lengths:
            dt = pd.Timedelta(w)
            b = (before[1] - dt, before[1])
            a = (after[0], after[0] + dt)
            _, c = stats.shift_analysis(valid, b, a, ctx["sigma_coverage"])
            sens_rows.append({"window": w, "n_usable": c["n_usable"],
                              "n_up": c["n_up"], "n_down": c["n_down"],
                              "sign_p": c["sign_p"], "stouffer_z": c["stouffer_z"],
                              "stouffer_p": c["stouffer_p"]})
        sens = pd.DataFrame(sens_rows)
        st.dataframe(sens.style.format({"sign_p": "{:.3g}", "stouffer_z": "{:.2f}",
                                        "stouffer_p": "{:.2e}"}, na_rep="—"),
                     width="stretch", hide_index=True)
        st.caption("Windows anchored at the end of the 'before' period and the "
                   "start of the 'after' period.")
else:
    st.info("Pick a start and end date for both periods.")
    table = pd.DataFrame()

# ---------------------------------------------- WLS trend fit ---- #
st.subheader("Dose-rate trend (weighted least squares)")
c1, c2, c3 = st.columns([1.4, 1.2, 1.2])
sensor_opts = ["R1+R2 combined"] + [config.sensor_id(g, c) for g, c in config.ALL_SENSORS]
which = c1.selectbox("Series", sensor_opts)
ch_fit = c2.selectbox("Channel", config.EXPECTED_CHANNELS,
                      format_func=plots.channel_name)
w_label = c3.selectbox("Bin width", list(config.WINDOW_OPTIONS), index=2)

binned = pipeline.binned_series(valid, config.WINDOW_OPTIONS[w_label],
                                ctx["sigma_coverage"])
if which == "R1+R2 combined":
    series = pipeline.combined_series(binned)
    series = series[series["channel"] == ch_fit]
else:
    g, c = which.split("_ch")
    series = binned[(binned["sensor_group"] == g) & (binned["channel"] == int(c))]
    if int(c) != ch_fit:
        st.caption("Note: the channel selector is ignored when a specific sensor is chosen.")

series = series.dropna(subset=["dose_rad"]).sort_values("timestamp")
if len(series) >= 3:
    tsec = pipeline.epoch_seconds(series["timestamp"])
    fit = stats.wls_trend(tsec, series["dose_rad"].to_numpy(),
                          series["dose_sigma_meas"].to_numpy())
    f1, f2, f3 = st.columns(3)
    f1.metric("Slope", f"{fit['slope_rad_day']:.2f} ± {fit['slope_sigma_rad_day']:.2f} Rad/day")
    f2.metric("χ²/dof", f"{fit['chi2_dof']:.1f}",
              help="≫1 means the accumulation rate is NOT constant over this span "
                   "(SAA banding, SPE) — see the Dose vs Time rate view.")
    f3.metric("Bins used", f"{fit['n']}")

    figf = go.Figure()
    color = plots.CHANNEL_COLORS[ch_fit]
    figf.add_trace(go.Scatter(x=series["timestamp"], y=series["dose_rad"],
                              mode="lines", line=dict(color=color, width=2),
                              name="binned dose", connectgaps=False,
                              hovertemplate="%{y:.0f} Rad<extra></extra>"))
    yfit = fit["intercept_rad"] + fit["slope_rad_day"] / 86400.0 * tsec
    figf.add_trace(go.Scatter(x=series["timestamp"], y=yfit, mode="lines",
                              line=dict(color=plots.INK_SECONDARY, width=1.5, dash="dash"),
                              name="WLS fit", hoverinfo="skip"))
    plots.apply_theme(figf, height=380, ytitle="Cumulative dose [Rad]")
    st.plotly_chart(figf, width="stretch")
    trend_table = pd.DataFrame([{"series": which, "channel": ch_fit,
                                 "bin": w_label, **fit}])
else:
    st.info("Not enough bins for a trend fit in the current selection.")
    trend_table = pd.DataFrame()

# ---------------------------------------------- periodicity ---- #
with st.expander("Periodicity check (Welch power spectrum of the dose rate)"):
    ps1, ps2 = st.columns(2)
    g_p = ps1.selectbox("Sensor group", config.TRIAL_GROUPS, key="per_g")
    ch_p = ps2.selectbox("Channel ", config.EXPECTED_CHANNELS, key="per_ch",
                         format_func=plots.channel_name)
    fine = pipeline.binned_series(valid, "15min", ctx["sigma_coverage"])
    one = pipeline.add_dose_rate(
        fine[(fine["sensor_group"] == g_p) & (fine["channel"] == ch_p)])
    per = stats.periodogram(one)
    if per.empty:
        st.info("Not enough data for a spectrum.")
    else:
        figp = go.Figure(go.Scatter(
            x=per["period_min"], y=per["power"], mode="lines",
            line=dict(color=plots.CHANNEL_COLORS[ch_p], width=1.5),
            hovertemplate="period %{x:.1f} min: %{y:.1f}× median<extra></extra>",
            name="power",
        ))
        for pmin, label in ((92.9, "orbital 92.9 min"), (1440.0, "24 h")):
            figp.add_vline(x=pmin, line=dict(color=plots.BASELINE, width=1, dash="dash"))
            figp.add_annotation(x=np.log10(pmin), yref="paper", y=1.02, showarrow=False,
                                text=label, font=dict(size=11, color=plots.MUTED))
        plots.apply_theme(figp, height=380, xtitle="Period [min]",
                          ytitle="Power / median", legend=False)
        figp.update_xaxes(type="log")
        st.plotly_chart(figp, width="stretch")
        st.caption("Welch-averaged (Hann, 50% overlap) spectrum of the 15-min "
                   "dose rate. Power near 92.9 min (and its daily-modulation "
                   "sidebands) is the orbital SAA-pass signature; a flat "
                   "spectrum means rate variations are below the noise floor.")

st.divider()
d1, d2 = st.columns(2)
with d1:
    if not table.empty:
        export.download_csv(table, "shift_test.csv", "Shift-test table", key="dl_shift")
with d2:
    if not trend_table.empty:
        export.download_csv(trend_table, "trend_fit.csv", "Trend-fit result", key="dl_trend")
