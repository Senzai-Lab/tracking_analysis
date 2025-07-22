"""Figure creation utilities for the Dash application."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


def make_figures(
    pos: np.ndarray,
    times: np.ndarray,
    frames: np.ndarray,
    markers: List[int],
    speed: np.ndarray,
    t_speed: np.ndarray,
    frames_speed: np.ndarray,
    ang_speed: np.ndarray,
    t_ang_speed: np.ndarray,
    frames_ang: np.ndarray,
    highlight_time: float | None = None,
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Create trajectory and speed figures."""
    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Rainbow"),
            line=dict(color="blue"),
            name="trajectory",
            customdata=np.stack([frames, times], axis=-1),
            hovertemplate=(
                "Frame %{customdata[0]}<br>"
                "X %{x:.3f}<br>Y %{y:.3f}<br>Z %{z:.3f}<extra></extra>"
            ),
        )
    )
    for idx in markers:
        fig3d.add_trace(
            go.Scatter3d(
                x=[pos[idx, 0]],
                y=[pos[idx, 1]],
                z=[pos[idx, 2]],
                mode="markers",
                marker=dict(color="orange", symbol="triangle-down", size=5),
                showlegend=False,
            )
        )

    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Trajectory",
    )

    fig2d = go.Figure()
    fig2d.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="lines+markers",
            marker=dict(size=3, color=times, colorscale="Rainbow"),
            name="trajectory",
            customdata=np.stack([frames, times], axis=-1),
            hovertemplate=(
                "Frame %{customdata[0]}<br>"
                "X %{x:.3f}<br>Y %{y:.3f}<extra></extra>"
            ),
        )
    )
    for idx in markers:
        fig2d.add_trace(
            go.Scatter(
                x=[pos[idx, 0]],
                y=[pos[idx, 1]],
                mode="markers",
                marker=dict(color="orange", symbol="triangle-down", size=8),
                showlegend=False,
            )
        )
    fig2d.update_layout(xaxis_title="X", yaxis_title="Y", title="2D Trajectory")
    fig2d.update_yaxes(scaleanchor="x", scaleratio=1)

    fig_speed = go.Figure()
    fig_speed.add_trace(
        go.Scatter(
            x=t_speed,
            y=speed,
            mode="lines",
            name="speed",
            line=dict(color="blue"),
            customdata=frames_speed,
            hovertemplate="Frame %{customdata}<br>Speed %{y:.3f}<extra></extra>",
        )
    )
    for tm in markers:
        if tm < len(times):
            fig_speed.add_vline(x=times[tm], line_color="orange", line_dash="dash")
    fig_speed.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Linear Speed",
        title="Linear Speed",
    )

    fig_ang = go.Figure()
    fig_ang.add_trace(
        go.Scatter(
            x=t_ang_speed,
            y=ang_speed,
            mode="lines",
            name="angular",
            line=dict(color="blue"),
            customdata=frames_ang,
            hovertemplate="Frame %{customdata}<br>Angular %{y:.3f}<extra></extra>",
        )
    )
    for tm in markers:
        if tm < len(times):
            fig_ang.add_vline(x=times[tm], line_color="orange", line_dash="dash")
    fig_ang.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Angular Speed",
        title="Angular Speed",
    )

    if highlight_time is not None:
        idx = int(np.searchsorted(times, highlight_time, side="left"))
        idx = max(0, min(idx, len(times) - 1))
        hx, hy, hz = pos[idx]
        fig3d.add_trace(
            go.Scatter3d(
                x=[hx],
                y=[hy],
                z=[hz],
                mode="markers",
                marker=dict(color="gray", size=6),
                showlegend=False,
            )
        )
        fig2d.add_trace(
            go.Scatter(
                x=[hx],
                y=[hy],
                mode="markers",
                marker=dict(color="gray", size=8),
                showlegend=False,
            )
        )
        fig_speed.add_vline(x=highlight_time, line_color="gray", line_dash="dot")
        fig_ang.add_vline(x=highlight_time, line_color="gray", line_dash="dot")

    return fig3d, fig2d, fig_speed, fig_ang
