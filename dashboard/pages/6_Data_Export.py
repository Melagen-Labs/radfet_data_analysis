"""Data Export - bulk CSV downloads of raw, derived, and summary data."""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Export", page_icon="📦", layout="wide")

from core import config, export, pipeline, ui  # noqa: E402

st.title("Data Export")

ctx = ui.sidebar()
df = ctx["df"]
if df.empty:
    st.warning("No readings in the current selection.")
    st.stop()

st.caption("Sensor and date filters from the sidebar apply to everything on "
           "this page. " + ui.settings_caption(ctx))

# ------------------------------------------------- per-reading export ---- #
st.subheader("Per-reading data")
c1, c2 = st.columns([1, 2])
validity = c1.radio("Rows", ["All", "Valid only", "Invalid only"])
col_options = {
    "raw_adc": "raw_adc",
    "raw_voltage [V]": "raw_voltage",
    "delta_v [V]": "delta_v",
    "QC flags": None,   # expands to the four flag columns
    "per-row dose [Rad]": None,  # computed on demand
}
chosen = c2.multiselect("Columns (timestamp, sensor_group, channel always included)",
                        list(col_options),
                        default=["raw_adc", "raw_voltage [V]", "delta_v [V]", "QC flags"])

sel = {"All": df, "Valid only": df[df["valid"]], "Invalid only": df[~df["valid"]]}[validity]
out = sel[["timestamp", "sensor_group", "channel"]].copy()
for label in chosen:
    if label == "QC flags":
        for f in ("flag_band", "flag_sat", "flag_bad_ts", "flag_dup", "valid"):
            out[f] = sel[f]
    elif label == "per-row dose [Rad]":
        conv = pipeline.vector_dose(sel["delta_v"].to_numpy(), np.zeros(len(sel)))
        out["dose_rad"] = conv["dose_rad"].to_numpy()
        out["curve_name"] = conv["curve_name"].to_numpy()
    else:
        out[col_options[label]] = sel[col_options[label]]

st.dataframe(out.head(500), width="stretch", hide_index=True)
size_mb = len(out) * max(len(out.columns), 1) * 12 / 1e6
st.caption(f"{len(out):,} rows selected (preview shows the first 500). "
           f"Estimated CSV size ~{size_mb:.0f} MB.")
export.download_csv(out, "radfet_per_reading.csv",
                    f"Per-reading CSV ({len(out):,} rows)", key="dl_reading")

st.divider()

# ------------------------------------------------- binned export ---- #
st.subheader("Binned dose series")
w = st.selectbox("Bin width", list(config.WINDOW_OPTIONS),
                 index=list(config.WINDOW_OPTIONS).index(config.DEFAULT_WINDOW))
valid = ctx["df_valid"]
binned = pipeline.binned_series(valid, config.WINDOW_OPTIONS[w], ctx["sigma_coverage"])
comb = pipeline.combined_series(binned).drop(columns=["w", "wx"], errors="ignore")
b1, b2 = st.columns(2)
with b1:
    export.download_csv(pipeline.add_dose_rate(binned),
                        f"dose_series_per_sensor_{config.WINDOW_OPTIONS[w]}.csv",
                        f"Per-sensor binned series ({w})", key="dl_bin")
with b2:
    export.download_csv(pipeline.add_dose_rate(comb),
                        f"dose_series_combined_{config.WINDOW_OPTIONS[w]}.csv",
                        f"R1+R2 combined series ({w})", key="dl_comb")

st.divider()

# ------------------------------------------------- summary tables ---- #
st.subheader("Summary tables (full selected window)")
from analyze_sample import significance_vs_reference, pairwise_z_matrix  # noqa: E402

trials, per_channel = pipeline.end_of_window_summary(
    valid, ctx["t0"], ctx["t1"], ctx["sigma_coverage"])
sig = significance_vs_reference(per_channel, config.UNSHIELDED_CHANNEL)
ref_dose = per_channel.get(config.UNSHIELDED_CHANNEL, {}).get("dose_mean", np.nan)
rows = []
for ch in config.EXPECTED_CHANNELS:
    c = per_channel.get(ch, {})
    s = sig.get(ch, {})
    dose = c.get("dose_mean", np.nan)
    rows.append({
        "channel": ch, "shielding": config.shield_label(ch),
        "dvt_mean_v": c.get("dvt_mean", np.nan),
        "dvt_sigma_v": c.get("dvt_sigma", np.nan),
        "curve": c.get("curve", "n/a"),
        "dose_rad": dose, "dose_sigma_rad": c.get("dose_sigma", np.nan),
        "attenuation_factor": ref_dose / dose if np.isfinite(dose) and dose > 0 else np.nan,
        "z_vs_bare": s.get("z", np.nan), "p_vs_bare": s.get("p", np.nan),
        "significance": "reference" if ch == config.UNSHIELDED_CHANNEL else s.get("label", ""),
    })
summary = pd.DataFrame(rows)

trial_rows = pd.DataFrame([
    {"sensor": config.sensor_id(t["group"], t["channel"]), **{
        k: v for k, v in t.items() if k not in ("group", "channel")}}
    for t in trials.values()
])

s1, s2 = st.columns(2)
with s1:
    st.markdown("**Per-configuration (R1+R2 combined)**")
    st.dataframe(summary.style.format({
        "dvt_mean_v": "{:.4f}", "dvt_sigma_v": "{:.5f}", "dose_rad": "{:,.0f}",
        "dose_sigma_rad": "{:,.0f}", "attenuation_factor": "{:.2f}",
        "z_vs_bare": "{:.2f}", "p_vs_bare": "{:.2e}",
    }, na_rep="—"), width="stretch", hide_index=True)
    export.download_csv(summary, "dose_summary_window.csv",
                        "Per-configuration summary", key="dl_cfg")
with s2:
    st.markdown("**Per-sensor, per-trial**")
    st.dataframe(trial_rows.style.format({
        "dvt_mean": "{:.4f}", "dvt_std": "{:.4f}", "sigma_v": "{:.4f}",
        "dose_rad": "{:,.0f}", "dose_sigma_rad": "{:,.0f}",
    }, na_rep="—"), width="stretch", hide_index=True)
    export.download_csv(trial_rows, "dose_per_trial_window.csv",
                        "Per-trial summary", key="dl_trial")
