"""
plots.py

Plotly theming + reusable figure builders.

Colors follow the dataviz reference palette: channel identity uses the fixed
categorical slot order (never cycled, color follows the channel regardless of
which series are visible), magnitude (heatmap) uses a single-hue blue ramp,
and QC flag states use the reserved status palette (never reused for series).
R1 vs R2 is encoded by dash style, not by color - color belongs to the channel.
Aqua/yellow slots sit below 3:1 contrast on the light surface, so every chart
ships with a legend and an adjacent table view (the relief rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config

# --- palette (reference instance, light mode) ------------------------------ #

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorical slots 1-5, fixed order, one per channel
CHANNEL_COLORS = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#008300", 5: "#4a3aa7"}
GROUP_DASH = {"R1": "solid", "R2": "dash", "R1+R2": "solid"}

# status palette (reserved for states, never series)
STATUS = {"saturation": "#d03b3b", "band": "#ec835a", "bad_timestamp": "#fab219",
          "duplicate": "#898781"}

# single-hue sequential ramp (blue 100 -> 700) for magnitude encodings
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def channel_name(ch: int) -> str:
    return f"ch{ch} – {config.shield_label(ch)}"


def hex_to_rgba(hx: str, alpha: float) -> str:
    hx = hx.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_theme(fig: go.Figure, *, height: int = 450, xtitle: str | None = None,
                ytitle: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SECONDARY, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=12, color=INK_SECONDARY)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, size=12, color=INK)),
    )
    axis = dict(gridcolor=GRID, gridwidth=1, zeroline=False,
                linecolor=BASELINE, linewidth=1,
                tickfont=dict(color=MUTED, size=12),
                title_font=dict(color=INK_SECONDARY, size=13))
    fig.update_xaxes(title_text=xtitle, **axis)
    fig.update_yaxes(title_text=ytitle, **axis)
    return fig


def add_band(fig: go.Figure, x, lo, hi, color: str, name: str,
             legendgroup: str) -> None:
    """Shaded +/-sigma band (a fill-between pair, excluded from the legend)."""
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([hi, lo[::-1]]),
        fill="toself", fillcolor=hex_to_rgba(color, 0.15),
        line=dict(width=0), hoverinfo="skip", showlegend=False,
        legendgroup=legendgroup, name=name,
    ))


def add_series(fig: go.Figure, sub: pd.DataFrame, ycol: str, *, channel: int,
               group: str, band: str | None = None,
               extrapolated_markers: bool = True,
               hover_fmt: str = ":.1f", yunit: str = "Rad") -> None:
    """One sensor series: 2px line, gaps rendered as gaps, optional sigma band
    ('meas' | 'full'), extrapolated bins drawn as open markers."""
    color = CHANNEL_COLORS[channel]
    lg = f"{group}-ch{channel}"
    name = f"{group} {channel_name(channel)}"
    sub = sub.sort_values("timestamp")

    if band in ("meas", "full") and len(sub) > 1:
        sig = sub["dose_sigma_meas" if band == "meas" else "dose_sigma_full"]
        ok = sub[ycol].notna() & sig.notna()
        if ok.sum() > 1:
            s = sub[ok]
            add_band(fig, s["timestamp"].to_numpy(),
                     (s[ycol] - sig[ok]).to_numpy(),
                     (s[ycol] + sig[ok]).to_numpy(), color, name, lg)

    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub[ycol], mode="lines",
        line=dict(color=color, width=2, dash=GROUP_DASH.get(group, "solid")),
        name=name, legendgroup=lg, connectgaps=False,
        hovertemplate=f"%{{y{hover_fmt}}} {yunit}<extra>{name}</extra>",
    ))

    if extrapolated_markers and "extrapolated" in sub.columns and sub["extrapolated"].any():
        ex = sub[sub["extrapolated"]]
        fig.add_trace(go.Scatter(
            x=ex["timestamp"], y=ex[ycol], mode="markers",
            marker=dict(symbol="circle-open", size=8, color=color,
                        line=dict(width=1.5)),
            name=f"{name} (extrapolated)", legendgroup=lg, showlegend=False,
            hovertemplate=f"%{{y{hover_fmt}}} {yunit} (beyond calibration)<extra>{name}</extra>",
        ))


def zmatrix_heatmap(zmat: dict, channels: list[int]) -> go.Figure:
    """Annotated 5x5 pairwise |z| heatmap on the sequential blue ramp."""
    z = np.array([[zmat.get((a, b), np.nan) for b in channels] for a in channels])
    labels = [channel_name(c) for c in channels]
    text = [["" if (i == j or not np.isfinite(z[i, j])) else f"{z[i, j]:.1f}"
             for j in range(len(channels))] for i in range(len(channels))]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale=[[0, BLUE_RAMP[0]], [1, BLUE_RAMP[-1]]],
        text=text, texttemplate="%{text}",
        textfont=dict(color=INK_SECONDARY, size=13),
        hovertemplate="%{y} vs %{x}: |z| = %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="|z|", font=dict(color=INK_SECONDARY)),
                      tickfont=dict(color=MUTED), outlinewidth=0),
        xgap=2, ygap=2,
    ))
    apply_theme(fig, height=420, legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def calibration_curves_figure() -> go.Figure:
    """The 5 Varadis fits on log-log, solid over each fit's validity range and
    dotted beyond it. One color per curve from the ordered blue ramp (they are
    ordered magnitudes of the same family, not independent categories)."""
    from calibration import CALIBRATION_CURVES

    ramp_idx = np.linspace(3, len(BLUE_RAMP) - 1, len(CALIBRATION_CURVES)).astype(int)
    fig = go.Figure()
    prev_max = 1.0
    for i, c in enumerate(CALIBRATION_CURVES):
        color = BLUE_RAMP[ramp_idx[i]]
        d_valid = np.geomspace(prev_max, c.dose_max_rad, 80)
        fig.add_trace(go.Scatter(
            x=d_valid, y=c.A * d_valid**c.B, mode="lines",
            line=dict(color=color, width=2), name=c.name,
            hovertemplate="D=%{x:.0f} Rad, dVt=%{y:.4f} V<extra>" + c.name + "</extra>",
        ))
        d_beyond = np.geomspace(c.dose_max_rad, 150_000, 40)
        fig.add_trace(go.Scatter(
            x=d_beyond, y=c.A * d_beyond**c.B, mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False, legendgroup=c.name, hoverinfo="skip",
        ))
    apply_theme(fig, height=470, xtitle="Dose [Rad]", ytitle="dVt [V]")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig
