"""export.py - CSV download helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def download_csv(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    st.download_button(
        label=f"⬇ {label}",
        data=csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        key=key,
    )
