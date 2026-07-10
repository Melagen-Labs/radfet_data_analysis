"""Shielding Effectiveness - cross-channel comparison with the imported
analyze_sample statistics stack (z-tests, pairwise matrix, Monte Carlo CIs)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Shielding Effectiveness", page_icon="🛡️", layout="wide")

from core import config, export, pipeline, plots, ui  # noqa: E402
from analyze_sample import (  # noqa: E402
    MC_SEED,
    monte_carlo_shielding_ci,
    pairwise_z_matrix,
    significance_vs_reference,
)

st.title("Shielding Effectiveness")

ctx = ui.sidebar()
valid = ctx["df_valid"]
if valid.empty:
    st.warning("No valid readings in the current selection.")
    st.stop()

# ------------------------------------------------- analysis window ---- #
preset = st.radio("Analysis window", ["Full selection", "Last 7 days", "Custom"],
                  horizontal=True)
t0, t1 = ctx["t0"], ctx["t1"]
if preset == "Last 7 days":
    t0 = t1 - pd.Timedelta(days=7)
elif preset == "Custom":
    cc1, cc2 = st.columns(2)
    t0 = pd.Timestamp(cc1.date_input("From", value=t0.date(),
                                     min_value=ctx["t0"].date(), max_value=t1.date()))
    t1 = pd.Timestamp(cc2.date_input("To", value=t1.date(),
                                     min_value=t0.date(), max_value=ctx["t1"].date())) \
        + pd.Timedelta(days=1)

trials, per_channel = pipeline.end_of_window_summary(valid, t0, t1, ctx["sigma_coverage"])
ref_ch = config.UNSHIELDED_CHANNEL
sig = significance_vs_reference(per_channel, ref_ch)
zmat = pairwise_z_matrix(per_channel, config.EXPECTED_CHANNELS)


@st.cache_data(show_spinner="Running Monte Carlo (calibration sampled common-mode)...")
def _mc(per_channel: dict, ref: int, n_draws: int) -> dict:
    return monte_carlo_shielding_ci(per_channel, ref, n_draws=n_draws)


n_draws = int(st.select_slider("Monte Carlo draws", options=[1_000, 10_000],
                               value=10_000))
mc = _mc(per_channel, ref_ch, n_draws)

st.caption(f"Window: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} · "
           f"MC seed {MC_SEED}, {n_draws:,} draws · " + ui.settings_caption(ctx))

# ------------------------------------------------- dose bar chart ---- #
st.subheader("Absolute dose by shielding configuration")
# R1/R2/combined are the same measure repeated, so they take ordered steps of
# one hue (sequential), not three categorical hues.
BAR_COLORS = {"R1": plots.BLUE_RAMP[3], "R2": plots.BLUE_RAMP[6],
              "R1+R2 combined": plots.BLUE_RAMP[10]}
fig = go.Figure()
xlabels = [plots.channel_name(ch) for ch in config.EXPECTED_CHANNELS]
for grp in config.TRIAL_GROUPS:
    ys = [trials.get((grp, ch), {}).get("dose_rad", np.nan)
          for ch in config.EXPECTED_CHANNELS]
    es = [trials.get((grp, ch), {}).get("dose_sigma_rad", np.nan)
          for ch in config.EXPECTED_CHANNELS]
    fig.add_trace(go.Bar(name=grp, x=xlabels, y=ys, marker_color=BAR_COLORS[grp],
                         error_y=dict(array=es, color=plots.INK_SECONDARY, width=3),
                         hovertemplate="%{y:.0f} Rad<extra>" + grp + "</extra>"))
ys = [per_channel.get(ch, {}).get("dose_mean", np.nan) for ch in config.EXPECTED_CHANNELS]
es = [per_channel.get(ch, {}).get("dose_sigma", np.nan) for ch in config.EXPECTED_CHANNELS]
fig.add_trace(go.Bar(name="R1+R2 combined", x=xlabels, y=ys,
                     marker_color=BAR_COLORS["R1+R2 combined"],
                     error_y=dict(array=es, color=plots.INK_SECONDARY, width=3),
                     hovertemplate="%{y:.0f} Rad<extra>combined</extra>"))
fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
plots.apply_theme(fig, height=430, ytitle="Dose over window [Rad]")
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch")
st.caption("Error bars: full 1σ (measurement ⊕ calibration) — appropriate for "
           "each configuration's absolute dose.")

# ------------------------------------------------- attenuation table ---- #
st.subheader("Attenuation & significance vs bare")
ref_dose = per_channel.get(ref_ch, {}).get("dose_mean", np.nan)
rows = []
for ch in config.EXPECTED_CHANNELS:
    c = per_channel.get(ch, {})
    s = sig.get(ch, {})
    m = mc.get(ch, {})
    att = m.get("attenuation", (np.nan,) * 3)
    red = m.get("reduction_pct", (np.nan,) * 3)
    rows.append({
        "channel": ch,
        "shielding": config.shield_label(ch),
        "dVt [V]": c.get("dvt_mean", np.nan),
        "dose [Rad]": c.get("dose_mean", np.nan),
        "dose ±1σ": c.get("dose_sigma", np.nan),
        "curve": c.get("curve", "n/a"),
        "attenuation ×": att[0],
        "atten CI95 lo": att[1], "atten CI95 hi": att[2],
        "reduction %": red[0],
        "red CI95 lo": red[1], "red CI95 hi": red[2],
        "z vs bare": s.get("z", np.nan),
        "p vs bare": s.get("p", np.nan),
        "verdict": "reference" if ch == ref_ch else s.get("label", "n/a"),
    })
summary = pd.DataFrame(rows)
st.dataframe(
    summary.style.format({
        "dVt [V]": "{:.4f}", "dose [Rad]": "{:,.0f}", "dose ±1σ": "{:,.0f}",
        "attenuation ×": "{:.2f}", "atten CI95 lo": "{:.2f}", "atten CI95 hi": "{:.2f}",
        "reduction %": "{:.1f}", "red CI95 lo": "{:.1f}", "red CI95 hi": "{:.1f}",
        "z vs bare": "{:.2f}", "p vs bare": "{:.2e}",
    }, na_rep="—"),
    width="stretch", hide_index=True,
)

# ------------------------------------------------- pairwise z heatmap ---- #
st.subheader("Pairwise |z| between configurations (voltage space)")
st.plotly_chart(plots.zmatrix_heatmap(zmat, config.EXPECTED_CHANNELS),
                width="stretch")
st.caption("|z| ≥ 2 ⇒ distinguishable at 95%; ≥ 3 ⇒ 99.7%. Tested on ΔV_t, where "
           "the calibration error is common-mode and cancels.")

# ------------------------------------------------- attenuation vs time ---- #
st.subheader("Attenuation factor vs time (weekly bins)")
weekly = pipeline.combined_series(
    pipeline.binned_series(valid, "7D", ctx["sigma_coverage"]))
piv = weekly.pivot_table(index="timestamp", columns="channel",
                         values="dose_rad", observed=True)
figt = go.Figure()
if ref_ch in piv.columns:
    for ch in config.EXPECTED_CHANNELS:
        if ch == ref_ch or ch not in piv.columns:
            continue
        ratio = piv[ref_ch] / piv[ch]
        figt.add_trace(go.Scatter(
            x=piv.index, y=ratio, mode="lines+markers",
            marker=dict(size=8), line=dict(color=plots.CHANNEL_COLORS[ch], width=2),
            name=plots.channel_name(ch), connectgaps=False,
            hovertemplate="×%{y:.2f}<extra>" + plots.channel_name(ch) + "</extra>",
        ))
    plots.apply_theme(figt, height=380, ytitle="dose_bare / dose_config")
    st.plotly_chart(figt, width="stretch")
    st.caption("A time-dependent attenuation factor indicates a changing radiation "
               "spectrum (e.g. soft SPE particles are absorbed more strongly than GCR).")
else:
    st.info("Bare channel not in the current sensor selection.")

with st.expander("Methodology"):
    st.markdown(
        "- Significance is tested in **voltage space** (measurement-only): "
        "calibration error is common-mode and cancels in the difference.\n"
        "- **Monte Carlo CIs**: per draw the calibration (A, B) of every curve is "
        "sampled **once and shared by all channels** (common-mode — it largely "
        "cancels in dose ratios); each channel's combined ΔV_t is sampled from its "
        "measurement σ; the narrow→wide curve selection is re-run per draw. "
        "Central value = MC median; CI = 95% percentile interval.\n"
        "- R1/R2 are combined by inverse-variance weighting in voltage space "
        "(analyze_sample.combine_trials)."
    )

st.divider()
d1, d2 = st.columns(2)
with d1:
    export.download_csv(summary, "shielding_summary.csv",
                        "Shielding summary (this window)", key="dl_sum")
with d2:
    mc_rows = [{"channel": ch, "n_used": m["n_used"],
                "attenuation_median": m["attenuation"][0],
                "attenuation_ci95_lo": m["attenuation"][1],
                "attenuation_ci95_hi": m["attenuation"][2],
                "reduction_pct_median": m["reduction_pct"][0],
                "reduction_pct_ci95_lo": m["reduction_pct"][1],
                "reduction_pct_ci95_hi": m["reduction_pct"][2]}
               for ch, m in mc.items()]
    export.download_csv(pd.DataFrame(mc_rows), "mc_attenuation_ci.csv",
                        "Monte Carlo CI table", key="dl_mc")
