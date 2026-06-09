"""
process_lead_enclosure.py

Turn unprocessedLeadEnclosureData.csv into a processed LeadEnclosureData.csv
whose `timestamp` column is a single, unified, monotonic real-time clock.

-------------------------------------------------------------------------------
The problem
-------------------------------------------------------------------------------
The raw log is one continuous acquisition recorded in two blocks:

  * REAL block  - timestamps like 2026-06-08T08:19:.. : valid wall-clock time.
  * EPOCH block - timestamps like 1969-12-31T19:00:.. : the device clock reset
    to the Unix epoch (1969-12-31T19:00:00 is epoch 0 in UTC-5). These are NOT
    real dates, but they still carry correct RELATIVE timing - the sampling
    cadence (~60 s per cycle) is intact within the block.

-------------------------------------------------------------------------------
The fix (assumptions made explicit)
-------------------------------------------------------------------------------
* Logging is continuous: no time was lost during the clock reset, so the epoch
  block immediately continues the real block on the same cadence.
* Each acquisition cycle begins with sensor R1/channel 1, and cycles are a
  fixed interval apart. We therefore anchor R1.ch1 -> R1.ch1: the first epoch
  cycle is placed exactly one median-cycle-interval after the last real cycle,
  and every epoch row keeps its exact intra-block offset from that anchor.
* The real block's timestamps are kept byte-for-byte; only the epoch block is
  rebased. The original raw value is preserved in a `raw_timestamp` column for
  traceability - the unified value lives in `timestamp`.

This rebasing changes only the epoch block's *absolute* reference; all relative
spacing (sub-second, intra-cycle, and inter-cycle) is preserved exactly.

Run:
    python process_lead_enclosure.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "unprocessedLeadEnclosureData.csv"
OUTPUT_CSV = SCRIPT_DIR / "LeadEnclosureData.csv"

TIMESTAMP_COL = "timestamp"
GROUP_COL = "sensor_group"
CHANNEL_COL = "channel"

# Rows at or before this instant are the corrupted "epoch" block. Anything the
# device clock reset to sits at/around 1970; real data is 2026. A 2000 cutoff
# cleanly separates the two with wide margin.
EPOCH_CUTOFF = pd.Timestamp("2000-01-01")

# The cycle always starts with this sensor; we anchor the rebase on it.
CYCLE_START = ("R1", 1)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["raw_timestamp"] = df[TIMESTAMP_COL]  # preserve provenance
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

    # Split the two blocks. The file is already in logging order, but we mask
    # by value (not row position) so the logic is robust to ordering.
    is_epoch = parsed <= EPOCH_CUTOFF
    is_real = ~is_epoch

    if not is_epoch.any():
        # Nothing to rebase - just copy through.
        df[TIMESTAMP_COL] = parsed
        df.to_csv(OUTPUT_CSV, index=False)
        print("No epoch block found; copied timestamps through unchanged.")
        return

    # --- Measure the cycle interval from the REAL block's R1.ch1 readings --- #
    # Consecutive R1.ch1 timestamps are one full cycle apart; their median is a
    # robust estimate of the sampling cadence.
    cycle_start_mask = (df[GROUP_COL] == CYCLE_START[0]) & (
        df[CHANNEL_COL] == CYCLE_START[1]
    )
    real_cycle_starts = parsed[is_real & cycle_start_mask].sort_values()
    cycle_interval = real_cycle_starts.diff().median()

    # --- Anchor the epoch block onto the real timeline --------------------- #
    last_real_cycle_start = real_cycle_starts.iloc[-1]
    epoch_times = parsed[is_epoch]
    first_epoch_cycle_start = epoch_times[cycle_start_mask].min()

    # First epoch cycle sits one interval after the last real cycle; every
    # epoch row keeps its exact offset from the first epoch cycle start.
    anchor = last_real_cycle_start + cycle_interval
    rebased = anchor + (epoch_times - first_epoch_cycle_start)

    unified = parsed.copy()
    unified.loc[is_epoch] = rebased
    df[TIMESTAMP_COL] = unified

    # Emit in unified chronological order so the file reads naturally.
    df = df.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # --- Console summary --------------------------------------------------- #
    print(f"Processed {len(df)} rows -> {OUTPUT_CSV.name}")
    print(f"  Real rows  : {int(is_real.sum())}")
    print(f"  Epoch rows : {int(is_epoch.sum())} (rebased)")
    print(f"  Cycle interval (median) : {cycle_interval}")
    print(f"  Last real cycle start   : {last_real_cycle_start}")
    print(f"  Epoch block anchored at : {anchor}")
    print(f"  Unified time span       : {df[TIMESTAMP_COL].min()}"
          f"  ->  {df[TIMESTAMP_COL].max()}")
    # Sanity: the unified series must be non-decreasing.
    monotonic = df[TIMESTAMP_COL].is_monotonic_increasing
    print(f"  Unified timestamps monotonic increasing: {monotonic}")


if __name__ == "__main__":
    main()
