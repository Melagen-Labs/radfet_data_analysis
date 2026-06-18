# sample_analysis — RADFET dose-analysis reference

A runnable, end-to-end demonstration of how we convert RADFET telemetry
(LeadEnclosureData.csv format) into absorbed dose, combine R1/R2 trials, and
compare shielding configurations. Runs on **synthetic** data so it works with
no flight data. On real data, only the input path in `analyze_sample.py`
changes.

## Files

| File | Purpose |
|------|---------|
| `calibration.py` | **Single source of truth.** Varadis QF 16 calibration coefficients, dVt→dose range selection, uncertainty propagation, and the channel→shielding map. |
| `generate_fake_data.py` | Builds `fake_iss_data.csv` with known true doses + injected anomalies (saturation spike, 1969 timestamps, corrupt voltage). |
| `analyze_sample.py` | Full pipeline → `output/dose_analysis_report.txt` and `output/dose_summary.csv`. |
| `generate_methodology_doc.py` | Generates `../RADFET_ISS_Dose_Analysis_Methodology.docx` from `calibration.py` (so doc and code never drift). |

## Run

```bash
cd sample_analysis
python generate_fake_data.py     # writes fake_iss_data.csv
python analyze_sample.py         # writes output/ report + summary
python generate_methodology_doc.py   # regenerates the Word doc
```

## Calibration model

```
dVt [V] = A · Dose[Rad] ^ B          (forward)
Dose[Rad] = (dVt / A) ^ (1 / B)      (inversion)
```

Five nested fits (0–1 … 0–100 kRad); the narrowest valid fit is selected per
reading. Channel→shielding: 1=None, 2=2 mm Al, 3=MLC1, 4=MLC1-b+Al, 5=MLC2;
channel 1 is the attenuation reference. Edit any of this in `calibration.py`.
