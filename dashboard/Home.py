"""Mission Overview - entry point. Run with:  streamlit run dashboard/Home.py"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RADFET ISS Dosimetry", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

from core import config, loading, pipeline, plots, ui  # noqa: E402

st.title("RADFET ISS Dosimetry — Mission Overview")

ctx = ui.sidebar()
df, valid = ctx["df"], ctx["df_valid"]

if valid.empty:
    st.warning("No valid readings in the current selection.")
    st.stop()

binned = pipeline.binned_series(valid, "6h", ctx["sigma_coverage"])
combined = pipeline.combined_series(binned)

# ---------------------------------------------------------------- KPIs ---- #
ts = valid["timestamp"]
t_first, t_last = ts.min(), ts.max()
elapsed_days = (t_last - t_first).total_seconds() / 86400.0

bare = combined[combined["channel"] == config.UNSHIELDED_CHANNEL].dropna(subset=["dose_rad"])
if not bare.empty:
    end = bare.iloc[-1]
    dose_txt = f"{end['dose_rad']:,.0f} Rad"
    dose_sub = f"± {end['dose_sigma_full']:,.0f} (1σ full)"
    day_ago = t_last - pd.Timedelta(hours=24)
    recent = bare[bare["timestamp"] >= day_ago]
    older = bare[bare["timestamp"] < day_ago]
    if not recent.empty and not older.empty:
        rate = (recent["dose_rad"].iloc[-1] - older["dose_rad"].iloc[-1]) / \
               max((recent["timestamp"].iloc[-1] - older["timestamp"].iloc[-1])
                   .total_seconds() / 86400.0, 1e-9)
        rate_txt = f"{rate:,.1f} Rad/day"
    else:
        rate_txt = "n/a"
else:
    dose_txt, dose_sub, rate_txt = "n/a", "no ch1 data selected", "n/a"

pct_valid = 100.0 * len(valid) / max(len(df), 1)
last_seen = valid.groupby(["sensor_group", "channel"], observed=True)["timestamp"].max()
stale_cutoff = t_last - pd.Timedelta(hours=24)
n_reporting = int((last_seen >= stale_cutoff).sum())
stale = [config.sensor_id(g, c) for (g, c), t in last_seen.items() if t < stale_cutoff]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Data span", f"{elapsed_days:.1f} days",
          help=f"{t_first:%Y-%m-%d %H:%M} → {t_last:%Y-%m-%d %H:%M} (valid readings)")
k2.metric("Total dose — bare, R1+R2", dose_txt, dose_sub, delta_color="off",
          help="Combined cumulative dose on the unshielded channel at the last "
               "6 h bin. ± is the full 1σ (measurement ⊕ calibration).")
k3.metric("Dose rate (last 24 h)", rate_txt,
          help="Bare-channel combined dose accumulated over the trailing 24 h.")
k4.metric("Valid readings", f"{pct_valid:.2f} %",
          help="Share of readings passing all QC rules in the current selection.")
k5.metric("Sensors reporting", f"{n_reporting}/{len(last_seen)}",
          help="Sensors with a valid reading in the final 24 h of the span.")
if stale:
    st.warning(f"Silent in the last 24 h of the span: {', '.join(stale)} — "
               "see Sensor Health / QC.")

# ------------------------------------------------------ headline chart ---- #
st.subheader("Cumulative dose vs time (R1+R2 combined, 6 h bins)")
fig = go.Figure()
for ch in sorted(combined["channel"].unique()):
    sub = combined[combined["channel"] == ch]
    band = "full" if ch == config.UNSHIELDED_CHANNEL else None
    plots.add_series(fig, sub, "dose_rad", channel=int(ch), group="R1+R2", band=band)
plots.apply_theme(fig, height=480, xtitle=None, ytitle="Cumulative dose [Rad]")

# auto-annotate a dose-rate outlier (e.g. a solar particle event)
if not bare.empty and len(bare) > 10:
    rated = pipeline.add_dose_rate(bare.assign(sensor_group="R1+R2"))
    r = rated["rate_rad_h"]
    med = r.median()
    spike = rated[r > 5 * med]
    if not spike.empty and med > 0:
        t_spike = spike.iloc[0]["timestamp"]
        fig.add_vline(x=t_spike, line=dict(color=plots.STATUS["saturation"],
                                           width=1, dash="dot"))
        fig.add_annotation(x=t_spike, yref="paper", y=1.0, showarrow=False,
                           text="dose-rate spike (possible SPE)",
                           font=dict(size=11, color=plots.STATUS["saturation"]))

st.plotly_chart(fig, width="stretch")
st.caption("±1σ band (measurement ⊕ calibration) shown on the bare channel only; "
           "per-channel bands are on the Dose vs Time page. "
           + ui.settings_caption(ctx))

# ----------------------------------------------------------- QC strip ---- #
qc = loading.qc_summary(df)
st.subheader("Data quality at a glance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Readings", f"{qc['n_total']:,}")
c2.metric("Band-rejected", f"{qc['n_band']:,}",
          help="delta_v non-finite or outside (0, 4] V")
c3.metric("Saturated", f"{qc['n_sat']:,}", help=f"raw_adc ≥ {config.SATURATION_ADC}")
c4.metric("Bad timestamps", f"{qc['n_bad_ts']:,}", help="missing or implausible (e.g. 1969 epoch)")
c5.metric("Duplicates", f"{qc['n_dup']:,}")
st.page_link("pages/3_Sensor_Health_QC.py", label="→ Full breakdown on the Sensor Health / QC page")
