"""
generate_iss_mission.py

Synthetic 90-day ISS mission telemetry for the RADFET dashboard.

Physics model (bare-sensor dose rate, integrated to cumulative dose):
  * GCR / trapped background: ~6 Rad/day with a 3% 30-day modulation.
  * South Atlantic Anomaly passes: the ISS orbit (92.9 min) regresses ~22.9
    deg/orbit in ascending-node longitude; orbits whose node falls inside a
    140 deg window around the SAA get a Gaussian dose pulse (FWHM 15 min)
    whose depth follows a Gaussian in node-longitude distance. ~6 passes/day
    arriving in daily clusters -> ~49 Rad/day. This is the dominant term and
    produces the characteristic banding of LEO dosimetry.
  * One solar particle event at mission day 47.3 (exponential decay,
    tau = 10 h, ~700 Rad bare) -> a step in cumulative dose.
  Bare end-of-mission total ~5.6 kRad, so the nested Varadis calibration
  curves (0-1 / 0-5 / 0-10 kRad) hand over mid-mission.

Shielding is modelled as two-component transmission: the hard (GCR) component
is barely attenuated, the soft (trapped/SPE) component strongly, per channel.
R2 sees 0.97x the R1 dose (same convention as sample_analysis fake data).

Measurement chain per reading (matches the dashboard's inverse conversion):
    v = baseline[group] + BOARD_OFFSET_V + dvt_true + N(0, sigma_V(group, ch))
    raw_adc = clip(round(v / 5.0 * 4095), 0, 4095)

The written `delta_voltage_v` column deliberately uses a WRONG baseline
(1.81 V for both groups) and `dose_rad` is derived from it: real telemetry
CSVs carry mislabeled voltage columns, and the dashboard must recompute
everything from raw_adc. `voltage_valid` is naively True everywhere.

Injected anomalies (all logged in simulation_manifest.json):
  saturation bursts, telemetry dropouts, a dead sensor window, 1969-epoch
  timestamps, absurd ADC values, duplicate rows, one out-of-order block.

Run:  python simulation/generate_iss_mission.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "dashboard"))

from core import config  # noqa: E402  (also bootstraps sample_analysis/)
from calibration import CALIBRATION_CURVES, measured_sigma_v  # noqa: E402

# --------------------------------------------------------------------------- #
# Mission parameters
# --------------------------------------------------------------------------- #

SEED = 20270301
RNG = np.random.default_rng(SEED)

MISSION_START = np.datetime64("2027-03-01T00:00:00")
N_DAYS = 90
CADENCE_S = 300                                # one reading / 5 min / sensor
N_STEPS = N_DAYS * 86400 // CADENCE_S          # 25,920 cycles

ORBIT_PERIOD_S = 5574.0                        # 92.9 min
NODE_REGRESSION_DEG = 22.9                     # westward drift per orbit

# GCR / trapped background
GCR_RAD_PER_DAY = 6.0
GCR_MOD_FRAC = 0.03
GCR_MOD_PERIOD_D = 30.0

# SAA
SAA_LON_DEG = 320.0
SAA_WINDOW_HALFWIDTH_DEG = 70.0
SAA_DEPTH_SIGMA_DEG = 55.0
SAA_PASS_DOSE_RAD = 12.5                       # bare, at full depth
SAA_PULSE_FWHM_S = 15 * 60.0
SAA_LON0_DEG = 40.0                            # node longitude of orbit 0

# Solar particle event
SPE_DAY = 47.3
SPE_TOTAL_RAD = 700.0
SPE_TAU_S = 10 * 3600.0

# Shielding transmission: hard (GCR) and soft (trapped/SPE) components.
T_HARD = {1: 1.00, 2: 0.97, 3: 0.92, 4: 0.94, 5: 0.95}
T_SOFT = {1: 1.00, 2: 0.62, 3: 0.30, 4: 0.42, 5: 0.50}

TRIAL_DOSE_FACTOR = {"R1": 1.00, "R2": 0.97}

# Written-file corruption: the CSV's own delta column uses this wrong baseline.
WRONG_BASELINE_V = 1.81

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------- #
# Dose-rate model
# --------------------------------------------------------------------------- #

def bare_rate_components(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (hard_rate, soft_rate) in Rad/s on the time grid t [s], plus a
    log of SAA passes. Hard = GCR; soft = SAA + SPE."""
    hard = (GCR_RAD_PER_DAY / 86400.0) * (
        1.0 + GCR_MOD_FRAC * np.sin(2 * np.pi * t / (GCR_MOD_PERIOD_D * 86400.0))
    )

    soft = np.zeros_like(t)
    sigma_t = SAA_PULSE_FWHM_S / 2.3548
    n_orbits = int(np.ceil(t[-1] / ORBIT_PERIOD_S)) + 1
    passes = []
    for k in range(n_orbits):
        lon = (SAA_LON0_DEG - NODE_REGRESSION_DEG * k) % 360.0
        d = (lon - SAA_LON_DEG + 180.0) % 360.0 - 180.0
        if abs(d) > SAA_WINDOW_HALFWIDTH_DEG:
            continue
        depth = float(np.exp(-((d / SAA_DEPTH_SIGMA_DEG) ** 2)))
        center = (k + 0.5) * ORBIT_PERIOD_S
        lo = np.searchsorted(t, center - 5 * sigma_t)
        hi = np.searchsorted(t, center + 5 * sigma_t)
        if lo >= len(t):
            continue
        window = t[lo:hi]
        soft[lo:hi] += (
            SAA_PASS_DOSE_RAD * depth
            * np.exp(-0.5 * ((window - center) / sigma_t) ** 2)
            / (sigma_t * np.sqrt(2 * np.pi))
        )
        passes.append({"orbit": k, "center_s": center, "depth": round(depth, 4)})

    t_spe = SPE_DAY * 86400.0
    spe = np.where(
        t >= t_spe,
        (SPE_TOTAL_RAD / SPE_TAU_S) * np.exp(-(t - t_spe) / SPE_TAU_S),
        0.0,
    )
    return hard, soft + spe, passes


def forward_dvt(dose: np.ndarray) -> np.ndarray:
    """Vectorized forward model: dVt = A * D^B using the narrowest Varadis fit
    containing each dose (same idiom as generate_fake_data._forward_curve_for_dose)."""
    bounds = np.array([c.dose_max_rad for c in CALIBRATION_CURVES])
    A = np.array([c.A for c in CALIBRATION_CURVES])
    B = np.array([c.B for c in CALIBRATION_CURVES])
    idx = np.clip(np.searchsorted(bounds, dose, side="left"), 0, len(bounds) - 1)
    return A[idx] * np.power(dose, B[idx])


def naive_dose_from_dvt(dvt: np.ndarray) -> np.ndarray:
    """Vectorized narrow->wide curve selection, used only to fill the CSV's
    convenience dose_rad column (which the dashboard ignores)."""
    dose = np.full(dvt.shape, np.nan)
    remaining = dvt > 0
    for c in CALIBRATION_CURVES:
        cand = np.where(remaining, (np.maximum(dvt, 1e-12) / c.A) ** (1.0 / c.B), np.nan)
        take = remaining & (cand <= c.dose_max_rad)
        dose[take] = cand[take]
        remaining &= ~take
    # beyond the widest fit: extrapolate with the last curve
    c = CALIBRATION_CURVES[-1]
    dose[remaining] = (dvt[remaining] / c.A) ** (1.0 / c.B)
    return dose


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #

def build_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t = np.arange(N_STEPS, dtype=float) * CADENCE_S
    hard, soft, saa_passes = bare_rate_components(t)

    frames = []
    truth_rows = []
    end_doses: dict[str, float] = {}
    day_idx = (np.arange(1, N_DAYS + 1) * (86400 // CADENCE_S)) - 1

    for group in config.TRIAL_GROUPS:
        group_offset_s = 0.75 if group == "R2" else 0.0
        for ch in config.EXPECTED_CHANNELS:
            rate = TRIAL_DOSE_FACTOR[group] * (T_HARD[ch] * hard + T_SOFT[ch] * soft)
            dose_true = np.cumsum(rate) * CADENCE_S
            dvt_true = forward_dvt(dose_true)

            sigma = measured_sigma_v(group, ch)
            v = (
                config.DV_BASELINE_BY_GROUP[group]
                + config.BOARD_OFFSET_V
                + dvt_true
                + RNG.normal(0.0, sigma, N_STEPS)
            )
            raw_adc = np.clip(
                np.rint(v / config.ADC_VREF_V * config.ADC_FULL_SCALE), 0, 4095
            ).astype(np.int64)

            jitter = RNG.normal(0.0, 0.02, N_STEPS)
            ts = (
                MISSION_START
                + ((t + group_offset_s + (ch - 1) * 0.12 + jitter) * 1e6)
                .astype("timedelta64[us]")
            )

            frames.append(pd.DataFrame({
                "timestamp": ts,
                "sensor_group": group,
                "channel": ch,
                "cycle": np.arange(N_STEPS),
                "raw_adc": raw_adc,
            }))

            end_doses[f"{group}_ch{ch}"] = float(dose_true[-1])
            truth_rows.append(pd.DataFrame({
                "date": (MISSION_START + (day_idx * CADENCE_S * 1e6).astype("timedelta64[us]"))
                        .astype("datetime64[D]"),
                "sensor_group": group,
                "channel": ch,
                "true_cum_dose_rad": dose_true[day_idx],
            }))

    df = pd.concat(frames, ignore_index=True)
    truth = pd.concat(truth_rows, ignore_index=True)
    manifest_model = {
        "seed": SEED,
        "mission_start": str(MISSION_START),
        "n_days": N_DAYS,
        "cadence_s": CADENCE_S,
        "orbit_period_s": ORBIT_PERIOD_S,
        "gcr_rad_per_day": GCR_RAD_PER_DAY,
        "saa_pass_dose_rad": SAA_PASS_DOSE_RAD,
        "n_saa_passes": len(saa_passes),
        "spe_day": SPE_DAY,
        "spe_total_rad_bare": SPE_TOTAL_RAD,
        "t_hard": T_HARD,
        "t_soft": T_SOFT,
        "trial_dose_factor": TRIAL_DOSE_FACTOR,
        "baselines_v": config.DV_BASELINE_BY_GROUP,
        "board_offset_v": config.BOARD_OFFSET_V,
        "wrong_baseline_in_csv_v": WRONG_BASELINE_V,
        "end_of_mission_true_dose_rad": {k: round(v, 1) for k, v in end_doses.items()},
    }
    return df, truth, manifest_model


def inject_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Corrupt the telemetry the way real logs get corrupted. Operates before
    the derived voltage columns are computed, so corrupted ADC values propagate
    into the written voltage/dose columns exactly like on the real logger."""
    log: list[dict] = []
    cyc_per_day = 86400 // CADENCE_S

    # 1. Saturation bursts: R1 all channels, 4 consecutive cycles, days 12 & 63.
    for day in (12, 63):
        c0 = day * cyc_per_day
        mask = (df["sensor_group"] == "R1") & df["cycle"].between(c0, c0 + 3)
        df.loc[mask, "raw_adc"] = 4095
        log.append({"type": "saturation_burst", "group": "R1", "channels": "all",
                    "day": day, "cycles": [c0, c0 + 3], "n_rows": int(mask.sum())})

    # 2. Absurd ADC values (sensor glitches).
    glitch_rows = RNG.choice(df.index.to_numpy(), size=8, replace=False)
    df.loc[glitch_rows[:5], "raw_adc"] = 0
    df.loc[glitch_rows[5:], "raw_adc"] = 999999
    log.append({"type": "absurd_adc", "n_zero": 5, "n_overflow": 3})

    # 3. Telemetry dropouts (rows never downlinked).
    drop_windows = [(20.0, 20.25), (41.0, 41.75), (70.0, 72.0)]
    drop = pd.Series(False, index=df.index)
    day_f = df["cycle"] / cyc_per_day
    for d0, d1 in drop_windows:
        drop |= (day_f >= d0) & (day_f < d1)
    dead = (df["sensor_group"] == "R2") & (df["channel"] == 4) & \
           (day_f >= 55.0) & (day_f < 60.0)
    log.append({"type": "dropout_windows", "windows_days": drop_windows,
                "n_rows": int(drop.sum())})
    log.append({"type": "dead_sensor", "sensor": "R2_ch4",
                "days": [55, 60], "n_rows": int(dead.sum())})
    df = df[~(drop | dead)].reset_index(drop=True)

    return df, log


def corrupt_output(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Post-derivation corruption: duplicates, 1969 timestamps, out-of-order."""
    log: list[dict] = []

    # Duplicate rows (logger retransmissions).
    dup_idx = RNG.choice(df.index.to_numpy(), size=50, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=False)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    log.append({"type": "duplicate_rows", "n_rows": 50})

    # 1969-epoch timestamps (RTC dropouts), payload intact.
    bad_idx = RNG.choice(df.index.to_numpy(), size=200, replace=False)
    epoch = np.datetime64("1969-12-31T23:59:00")
    offsets = (RNG.uniform(0, 59e6, size=200)).astype("timedelta64[us]")
    df.loc[bad_idx, "timestamp"] = epoch + offsets
    log.append({"type": "epoch_1969_timestamps", "n_rows": 200})

    # One out-of-order block (buffered replay written late).
    n = len(df)
    b0, span, dest = n // 2, 30, n // 2 + 5000
    block = df.iloc[b0:b0 + span]
    rest = df.drop(df.index[b0:b0 + span]).reset_index(drop=True)
    df = pd.concat(
        [rest.iloc[:dest], block, rest.iloc[dest:]], ignore_index=True
    )
    log.append({"type": "out_of_order_block", "n_rows": span})

    return df, log


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, truth, model = build_frame()
    df, anomaly_log = inject_anomalies(df)

    # Derived columns exactly as the real logger writes them - including the
    # WRONG baseline in delta_voltage_v (the dashboard must ignore it).
    df["raw_voltage_v"] = df["raw_adc"] * (config.ADC_VREF_V / config.ADC_FULL_SCALE)
    df["delta_voltage_v"] = df["raw_voltage_v"] - WRONG_BASELINE_V
    df["voltage_valid"] = True
    df["dose_rad"] = np.round(naive_dose_from_dvt(df["delta_voltage_v"].to_numpy()), 3)
    df["dose_rad"] = df["dose_rad"].fillna(0.0)

    df, corruption_log = corrupt_output(df)
    anomaly_log.extend(corruption_log)

    cols = ["timestamp", "sensor_group", "channel", "raw_adc",
            "raw_voltage_v", "delta_voltage_v", "voltage_valid", "dose_rad"]
    df["raw_voltage_v"] = df["raw_voltage_v"].round(6)
    df["delta_voltage_v"] = df["delta_voltage_v"].round(6)
    df[cols].to_csv(OUTPUT_DIR / "radfet_iss_mission.csv", index=False)

    truth.to_csv(OUTPUT_DIR / "ground_truth.csv", index=False)

    manifest = {"model": model, "anomalies": anomaly_log}
    (OUTPUT_DIR / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(df)} rows -> {OUTPUT_DIR / 'radfet_iss_mission.csv'}")
    print(f"SAA passes: {model['n_saa_passes']}")
    print("End-of-mission true dose [Rad]:")
    for k, v in model["end_of_mission_true_dose_rad"].items():
        print(f"  {k}: {v}")
    adc = df["raw_adc"]
    print(f"raw_adc range (excl. glitches): "
          f"{adc[adc <= 4095].min()} .. {adc[adc <= 4095].max()}")


if __name__ == "__main__":
    main()
