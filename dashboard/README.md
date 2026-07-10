# RADFET ISS Dosimetry Dashboard

Multi-page Streamlit dashboard for the RADFET ISS experiment: dose vs time,
shielding effectiveness, sensor health/QC, statistical tests, calibration
reference, and CSV export.

## Run locally

```bash
pip install -r requirements.txt        # from the repo root
streamlit run dashboard/Home.py
```

On first run the app generates the simulated 90-day ISS mission dataset
(`simulation/output/`, deterministic seed) automatically. To regenerate it
manually: `python simulation/generate_iss_mission.py`.

When real ISS telemetry arrives, either drop it in as a new entry in
`dashboard/core/config.py` → `DATA_SOURCES`, or use the sidebar's
**Upload CSV** option (required columns: `timestamp, sensor_group, channel,
raw_adc` — everything else is recomputed).

## Voltage conversion — the one rule

```
raw_voltage = raw_adc × (5.0 / 4095)
delta_v     = raw_voltage − baseline(group) − board offset
```

Defaults: R1 baseline **1.71 V**, R2 baseline **1.73 V**, board offset
**0.10 V**. All three are editable in the sidebar and displayed under every
chart. The CSV's own `voltage`/`delta_voltage_v`/`dose_rad` columns are never
trusted (historically inconsistent baselines) — everything derives from
`raw_adc`.

Calibration (Varadis QF 16 power-law fits), per-sensor noise floors, and the
R1/R2 combination + Monte Carlo statistics are imported from
`sample_analysis/calibration.py` and `sample_analysis/analyze_sample.py` —
the dashboard never re-derives them.

## Deploy to Streamlit Community Cloud (shareable link)

1. Push this repo to GitHub (the simulated data is gitignored — the app
   regenerates it on first run in the cloud).
2. Go to https://share.streamlit.io → **New app** → pick the repo/branch and
   set the entrypoint to `dashboard/Home.py`.
3. Deploy. You get a `https://<name>.streamlit.app` URL to share.
   For privacy: app settings → *Sharing* → restrict viewers to specific
   email addresses.

`requirements.txt` at the repo root is the dependency manifest the cloud
installs from.
