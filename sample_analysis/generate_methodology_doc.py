"""
generate_methodology_doc.py

Generate the Word (.docx) methodology document for the R&D / science team,
describing how dose analysis is performed on data returned from the ISS (in the
LeadEnclosureData.csv format).

The calibration table and shielding map are pulled directly from calibration.py
so the document can never disagree with the code.

Run:
    cd sample_analysis
    python generate_methodology_doc.py
Writes: ../RADFET_ISS_Dose_Analysis_Methodology.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

from calibration import (
    CALIBRATION_CURVES,
    SHIELDING_BY_CHANNEL,
    TRIAL_GROUPS,
    UNSHIELDED_CHANNEL,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DOCX = SCRIPT_DIR.parent / "RADFET_ISS_Dose_Analysis_Methodology.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def add_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    return h


def add_body(doc, text):
    p = doc.add_paragraph(text)
    return p


def add_bullet(doc, text):
    return doc.add_paragraph(text, style="List Bullet")


def add_numbered(doc, text):
    return doc.add_paragraph(text, style="List Number")


def add_mono(doc, text):
    """A monospace block for formulae / pseudocode."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    return p


def build() -> None:
    doc = Document()

    # Base style
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    # ---- Title block -----------------------------------------------------
    title = doc.add_heading("RADFET ISS Dose-Analysis Methodology", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    sub = doc.add_paragraph()
    r = sub.add_run("Converting flight dosimeter telemetry to absorbed dose, "
                    "with shielding comparison and uncertainty")
    r.italic = True
    r.font.color.rgb = ACCENT

    meta = doc.add_paragraph()
    meta.add_run("Audience: ").bold = True
    meta.add_run("R&D / Science team\n")
    meta.add_run("Prepared: ").bold = True
    meta.add_run("2026-06-17\n")
    meta.add_run("Status: ").bold = True
    meta.add_run("Draft for review")

    doc.add_paragraph(
        "This document describes how we will analyze the dosimetry data "
        "returned from the ISS. The flight data arrives in the same CSV format "
        "as LeadEnclosureData.csv. A runnable reference implementation of every "
        "step below lives in the sample_analysis/ folder and is demonstrated on "
        "synthetic data."
    )

    # ---- 1. Purpose & scope ---------------------------------------------
    add_heading(doc, "1. Purpose and scope", 1)
    add_body(doc,
             "We fly five RADFET dosimeters, each behind a different shielding "
             "stack, to measure how effectively each stack attenuates the "
             "on-orbit radiation dose. This document defines the agreed "
             "procedure for turning the raw voltage telemetry into absorbed "
             "dose (in Rad), for combining repeat measurements, for quantifying "
             "uncertainty, and for comparing the shielding configurations.")

    # ---- 2. Input data format -------------------------------------------
    add_heading(doc, "2. Input data format", 1)
    add_body(doc,
             "Each row is one channel read at one time. Columns (identical to "
             "LeadEnclosureData.csv):")
    cols = [
        ("timestamp", "ISO-8601 read time (host clock)."),
        ("sensor_group", "R1 or R2 - the two read-out groups (see Section 4)."),
        ("channel", "1-5 - identifies the sensor / shielding configuration."),
        ("raw_adc", "12-bit ADC count (0-4095) behind the voltage."),
        ("voltage", "Threshold-voltage shift dVt in Volts (already computed)."),
        ("dose_rad", "Convenience dose column; the analysis recomputes dose itself."),
        ("raw_timestamp", "Secondary timestamp; may contain epoch corruption."),
    ]
    t = doc.add_table(rows=1, cols=2)
    t.style = "Light Grid Accent 1"
    t.rows[0].cells[0].paragraphs[0].add_run("Column").bold = True
    t.rows[0].cells[1].paragraphs[0].add_run("Meaning").bold = True
    for name, meaning in cols:
        cells = t.add_row().cells
        cells[0].paragraphs[0].add_run(name).font.name = "Consolas"
        cells[1].text = meaning
    add_body(doc,
             "Important: the voltage column already carries the threshold-voltage "
             "shift dVt (irradiated minus non-irradiated reference), so no extra "
             "baseline subtraction is performed in the dose conversion.")

    # ---- 3. Calibration model -------------------------------------------
    add_heading(doc, "3. Calibration model", 1)
    add_body(doc,
             "We use the Varadis QF 16 RADFET calibration (400nm IMPL RADFET, "
             "plastic package; Mask-set COMRAD; Lot P5925-W3; Co-60 source at "
             "5.0 kRad/hour; Issue No.01, 01/08/2023). The fit relates the "
             "threshold-voltage shift to absorbed dose by a power law:")
    add_mono(doc, "    dVt [V] = A * Dose[Rad] ^ B")
    add_body(doc, "Inverting to recover dose from a measured dVt:")
    add_mono(doc, "    Dose[Rad] = ( dVt / A ) ^ (1 / B)")
    add_body(doc,
             "Varadis provides FIVE fits of the same data, each valid over a "
             "different dose range, and advises using the curve applicable to "
             "the actual dose. The narrow fits are the most accurate at low "
             "dose. Coefficients (single source of truth: calibration.py):")

    t = doc.add_table(rows=1, cols=6)
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(["Dose range", "A", "sigma(A)", "B", "sigma(B)", "R^2"]):
        t.rows[0].cells[i].paragraphs[0].add_run(head).bold = True
    for c in CALIBRATION_CURVES:
        cells = t.add_row().cells
        cells[0].text = c.name
        cells[1].text = f"{c.A:.4g}"
        cells[2].text = f"{c.sigma_A:.3e}"
        cells[3].text = f"{c.B:.4g}"
        cells[4].text = f"{c.sigma_B:.3e}"
        cells[5].text = f"{c.r_square:.3f}"

    # ---- 4. Sensors & shielding -----------------------------------------
    add_heading(doc, "4. Sensors, shielding, and trials", 1)
    add_body(doc,
             "Five sensors are flown, one per channel, each behind a different "
             "shielding stack:")
    t = doc.add_table(rows=1, cols=2)
    t.style = "Light Grid Accent 1"
    t.rows[0].cells[0].paragraphs[0].add_run("Channel").bold = True
    t.rows[0].cells[1].paragraphs[0].add_run("Shielding configuration").bold = True
    for ch in sorted(SHIELDING_BY_CHANNEL):
        cells = t.add_row().cells
        cells[0].text = str(ch)
        label = SHIELDING_BY_CHANNEL[ch]
        if ch == UNSHIELDED_CHANNEL:
            label += "  (reference for attenuation)"
        cells[1].text = label
    add_body(doc,
             f"Each sensor is read out at two positions/units recorded as "
             f"sensor_group {' and '.join(TRIAL_GROUPS)}. We treat R1 and R2 as "
             f"two independent TRIALS (replicates) of the same shielding "
             f"configuration. This gives, per configuration, two dose estimates "
             f"whose agreement is itself a check on repeatability.")

    # ---- 5. Range selection ---------------------------------------------
    add_heading(doc, "5. Calibration-curve (range) selection", 1)
    add_body(doc,
             "Because the correct curve depends on the dose we are trying to "
             "measure, we select it self-consistently. The five ranges are "
             "nested (0-1 kRad within 0-5 kRad within ... within 0-100 kRad), "
             "so we walk from the narrowest curve to the widest and accept the "
             "first curve whose own dose estimate lands inside its validity "
             "range:")
    add_mono(doc,
             "for curve in [0-1, 0-5, 0-10, 0-50, 0-100] kRad:   # narrow -> wide\n"
             "    dose = (dVt / curve.A) ^ (1 / curve.B)\n"
             "    if dose <= curve.dose_max:\n"
             "        return curve, dose          # most accurate valid fit\n"
             "# if none qualify, report the 0-100 kRad estimate, flagged EXTRAP")
    add_body(doc,
             "A reading whose dose exceeds even the 0-100 kRad fit is flagged as "
             "an extrapolation rather than silently trusted.")

    # ---- 6. Data quality -------------------------------------------------
    add_heading(doc, "6. Data-quality control", 1)
    add_body(doc,
             "The real telemetry contains corrupted rows. We never silently drop "
             "them; we flag them and exclude them from statistics so the dose "
             "reflects genuine sensor behavior. A reading is excluded when:")
    add_bullet(doc, "the voltage is non-finite or outside the plausible band "
                    "(e.g. the ~3.8e14 corruption seen in real data);")
    add_bullet(doc, "the ADC is at/near full scale (saturation / rail hit), such "
                    "as the synchronous spike where all channels read ~max for "
                    "one cycle;")
    add_bullet(doc, "the timestamp is out of range (e.g. 1969-epoch rows). Such "
                    "timestamps are treated as missing so they cannot masquerade "
                    "as the first reading; the voltage may still be usable.")
    add_body(doc,
             "Counts of each rejection category are reported so data quality is "
             "visible, not hidden.")

    # ---- 7. Aggregation & combination -----------------------------------
    add_heading(doc, "7. Per-sensor aggregation and trial combination", 1)
    add_numbered(doc, "For each (sensor_group, channel), average dVt over the "
                      "valid readings to get the sensor's mean dVt, with its "
                      "sample standard deviation and standard error of the mean "
                      "(SEM = std / sqrt(n)).")
    add_numbered(doc, "Convert the mean dVt to dose using the selected curve "
                      "(Section 5).")
    add_numbered(doc, "Combine the R1 and R2 dose estimates for each channel "
                      "into a single per-configuration dose (their mean). Half "
                      "the spread between the two trials is reported as a "
                      "trial-to-trial uncertainty.")

    # ---- 8. Uncertainty --------------------------------------------------
    add_heading(doc, "8. Uncertainty quantification", 1)
    add_body(doc,
             "We propagate a 1-sigma dose uncertainty through the inversion "
             "Dose = (dVt/A)^(1/B), combining measurement noise on dVt with the "
             "calibration uncertainties sigma(A) and sigma(B):")
    add_mono(doc,
             "(sigma_Dose / Dose)^2 =\n"
             "      ( sigma_dVt / (B * dVt) )^2     # measurement (SEM on dVt)\n"
             "    + ( sigma_A  / (B * A)   )^2     # calibration uncertainty in A\n"
             "    + ( ln(Dose) * sigma_B / B )^2   # calibration uncertainty in B")
    add_body(doc,
             "The measurement term uses the SEM of the sensor's readings. We "
             "report both this propagated 1-sigma and the independent "
             "trial-to-trial spread (R1 vs R2); broad disagreement between the "
             "two is a flag for an unmodeled systematic.")

    # ---- 9. Shielding effectiveness -------------------------------------
    add_heading(doc, "9. Shielding-effectiveness analysis", 1)
    add_body(doc,
             "Using the bare sensor (channel "
             f"{UNSHIELDED_CHANNEL}) as the reference, each shielded "
             "configuration is summarized by:")
    add_bullet(doc, "Attenuation factor = dose(bare) / dose(config)  "
                    "(higher means more effective shielding).")
    add_bullet(doc, "Dose reduction (%) = (1 - dose(config) / dose(bare)) x 100.")
    add_body(doc,
             "These are reported per configuration alongside the combined dose "
             "and its uncertainty.")

    # ---- 10. Outputs -----------------------------------------------------
    add_heading(doc, "10. Outputs", 1)
    add_bullet(doc, "A text report: data-quality summary, per-sensor / per-trial "
                    "dose, and the shielding-effectiveness table.")
    add_bullet(doc, "A tidy CSV: one row per shielding configuration (dose, "
                    "uncertainties, attenuation factor, % reduction) for "
                    "downstream plotting and reporting.")

    # ---- 11. Reference implementation -----------------------------------
    add_heading(doc, "11. Reference implementation", 1)
    add_body(doc, "The sample_analysis/ folder contains a runnable demonstration "
                  "on synthetic data:")
    add_bullet(doc, "calibration.py - calibration coefficients, range selection, "
                    "uncertainty, and the channel->shielding map (single source "
                    "of truth; this document is generated from it).")
    add_bullet(doc, "generate_fake_data.py - builds an ISS-format dataset with "
                    "known true doses and injected anomalies.")
    add_bullet(doc, "analyze_sample.py - runs the full pipeline; on real data "
                    "only the input path changes.")

    # ---- 12. Assumptions & open questions -------------------------------
    add_heading(doc, "12. Assumptions and open questions for review", 1)
    add_bullet(doc, "The voltage column is the threshold-voltage shift dVt "
                    "(already baseline-subtracted). Please confirm.")
    add_bullet(doc, "Channel->shielding mapping (Section 4) is as listed. "
                    "Confirm before the flight run.")
    add_bullet(doc, "The Co-60 ground calibration is assumed applicable to the "
                    "mixed ISS radiation environment; any spectral correction "
                    "factor is out of scope here and would be applied downstream.")
    add_bullet(doc, "Valid-voltage band and saturation threshold are tuned to "
                    "the lead-enclosure baseline; revisit for the higher dVt "
                    "expected on orbit.")

    doc.save(OUTPUT_DOCX)
    print(f"Wrote {OUTPUT_DOCX}")


if __name__ == "__main__":
    build()
