# ambient_analysis — long-duration RADFET voltage-drift test

Analyzes the multi-day **ambient** voltage log to answer: *has any sensor's
voltage meaningfully shifted over the period, and did the sensors as a group
drift consistently in one direction?* Reads the daily CSVs under
`raw_data/historic_data`, compares a "before" window against an "after" window,
and tests both per-sensor and collective significance.

## Files

| File | Purpose |
|------|---------|
| `analyze_ambient.py` | Full pipeline → `output/ambient_analysis_report.txt`, `output/ambient_shift_summary.csv`, and (if matplotlib is present) `output/plots/`. |
| `sigma_v.py` | Parses the per-sensor measurement uncertainty σ_V from `../lead_brick_analysis/analysis_report.txt` (the "std dev (sample)" figures), with a transcribed fallback. Mirrors the parser in `sample_analysis/calibration.py`. |

## Run

```bash
cd ambient_analysis
python analyze_ambient.py
```

Depends only on `pandas`/`numpy` (see `requirements.txt`). Plots are optional —
install `matplotlib` to also get `output/plots/`.

## Method

- **delta_v is recomputed from `raw_adc`**, not read from the CSV. The CSV's
  `voltage` column is mislabeled (it is a delta_v) and the baseline subtracted
  to produce it changed over time, so it is not comparable across days. We use
  `raw_voltage = raw_adc·(5.0/4095)` then `delta_v = raw_voltage − baseline`,
  with per-group baselines `R1 = 1.71 V`, `R2 = 1.73 V`. A constant baseline
  cancels in `after − before`, so the shift test is robust to the baseline value.
- **σ_V** per sensor comes from the lead-brick report (characterized noise
  floor). The uncertainty on a window mean of *n* readings is σ_V/√n. It is a
  linear transform of `raw_adc` at the same scale, so the per-reading scatter
  carries over unchanged.
- **Per-sensor shift**: `shift = after − before`, tested as
  `z = shift / (σ_V·√(1/n_before + 1/n_after))`, two-sided `p = erfc(|z|/√2)`.
- **Collective shift**: a sign / exact-binomial test on how many sensors went
  up vs down, plus Stouffer's combined `Z = Σz_i/√N`. Consistent small shifts
  in the same direction are themselves significant even when no single sensor is.
- **Window-length sensitivity**: every length in `WINDOW_LENGTHS_MIN`
  (30 min … 3 hr) is trialed and tabulated *before* a default is committed, so
  the dependence on window choice is explicit.

## Configuration (top of `analyze_ambient.py`)

| Setting | Meaning |
|---------|---------|
| `BEFORE_START` / `BEFORE_END` / `AFTER_START` / `AFTER_END` | The exact before/after comparison times. `None` ⇒ auto-detect: "before" anchors at the earliest data, "after" at the most recent. The span **end ("present day") is detected from the data**, never hard-coded. |
| `BEFORE_ANCHOR` / `AFTER_ANCHOR` | Where the averaging window sits within each period (`"start"` / `"end"`). |
| `WINDOW_LENGTHS_MIN` / `DEFAULT_WINDOW_MIN` | Window lengths to trial, and the committed default. |
| `SPAN_START` | Earliest date considered (default `2026-06-10`). |
| `SIGMA_V_COVERAGE` | Coverage factor on every σ_V (1.0 = raw 1-σ). |
