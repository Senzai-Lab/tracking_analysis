"""Data loading and preprocessing helpers for the Dash application."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data
from tracking_analysis.grouping import group_entities
from tracking_analysis.filtering import (
    filter_anomalies,
    filter_position,
    apply_ranges,
)
from tracking_analysis.kinematics import (
    compute_linear_velocity,
    compute_angular_speed,
    compute_angular_velocity,
)


def apply_filters(x: np.ndarray, times: np.ndarray, filters: List[dict]) -> Dict[str, np.ndarray]:
    """Apply a sequence of filters to ``x``.

    Parameters
    ----------
    x:
        Input signal array.
    times:
        Time stamps corresponding to ``x``.
    filters:
        Iterable of filter configuration dictionaries.

    Returns
    -------
    dict
        Mapping of filter name to filtered output array.
    """
    if not filters or len(x) == 0:
        return {}

    fs = 1.0 / float(np.mean(np.diff(times))) if len(times) > 1 else 1.0
    results: Dict[str, np.ndarray] = {}
    for idx, cfg in enumerate(filters):
        ftype = cfg.get("type")
        if not ftype:
            continue
        name = cfg.get("name", ftype or f"f{idx}")
        y = x
        if ftype == "moving_average":
            window = max(1, int(cfg.get("window", 5)))
            kernel = np.ones(window) / window
            y = np.convolve(x, kernel, mode="same")
        elif ftype == "ema":
            alpha = float(cfg.get("alpha", 0.3))
            y = np.empty_like(x)
            y[0] = x[0]
            for i in range(1, len(x)):
                y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        elif ftype == "butterworth":
            from scipy.signal import butter, filtfilt, lfilter

            order = int(cfg.get("order", 3))
            cutoff_hz = float(cfg.get("cutoff", 1.0))
            nyq = 0.5 * fs
            norm_cutoff = min(cutoff_hz / nyq, 0.99)
            b, a = butter(order, norm_cutoff, btype="low")
            padlen = 3 * max(len(a), len(b))
            y = filtfilt(b, a, x) if len(x) > padlen else lfilter(b, a, x)
        elif ftype == "savgol":
            from scipy.signal import savgol_filter

            window = int(cfg.get("window", 5))
            if window % 2 == 0:
                window += 1
            poly = int(cfg.get("polyorder", 2))
            y = savgol_filter(x, window, poly)
        elif ftype == "window":
            window = int(cfg.get("window", 10))
            if window < 2:
                window = 2
            if window % 2 != 0:
                window += 1
            half = window // 2
            padded = np.pad(x, (half, half), mode="edge")
            y = np.empty_like(x, dtype=float)
            for i in range(len(x)):
                seg = padded[i : i + window]
                m1 = np.mean(seg[:half])
                m2 = np.mean(seg[half:])
                y[i] = (m1 + m2) / 2
        elif ftype == "decimal_removal":
            digits = int(cfg.get("digits", 1))
            digits = max(0, digits)
            factor = 10 ** digits
            scaled = x / 180.0
            scaled = np.trunc(scaled * factor) / factor
            y = scaled * 180.0
        elif ftype == "fir":
            from scipy.signal import firwin, lfilter

            taps = int(cfg.get("numtaps", 21))
            cutoff_hz = float(cfg.get("cutoff", 1.0))
            nyq = 0.5 * fs
            y = lfilter(firwin(taps, cutoff_hz / nyq), [1.0], x)
        else:
            continue

        results[name] = y

    return results


def slice_range(times: np.ndarray, start: float, end: float) -> slice:
    """Return ``slice`` covering ``start``..``end`` within ``times``."""
    i0 = int(np.searchsorted(times, start, side="left"))
    i1 = int(np.searchsorted(times, end, side="right"))
    return slice(i0, i1)


def build_table(data: dict, start: float, end: float) -> List[dict]:
    """Create table rows for the given time range."""
    sl = slice_range(data["times"], start, end)
    times = data["times"][sl]
    frames = data["frames"][sl]
    pos = data["pos"][sl]
    iv = np.searchsorted(data["t_speed"], times)
    ia = np.searchsorted(data["t_ang_vel"], times)
    rows = []
    for idx, t in enumerate(times):
        spd = data["speed"][iv[idx]] if iv[idx] < len(data["speed"]) else float("nan")
        ang = data["ang_speed"][ia[idx]] if ia[idx] < len(data["ang_speed"]) else float("nan")
        rows.append(
            {
                "frame": int(frames[idx]),
                "time": float(t),
                "x": float(pos[idx, 0]),
                "y": float(pos[idx, 1]),
                "z": float(pos[idx, 2]),
                "speed": float(spd),
                "angular_speed": float(ang),
            }
        )
    return rows


def prepare_data(cfg: Config) -> Tuple[Dict[str, dict], List[str]]:
    """Load and preprocess the CSV based on ``cfg``."""

    input_file = cfg.get("input_file")
    if not os.path.exists(input_file):
        fallback = os.path.join("data", "input.csv")
        input_file = fallback if os.path.exists(fallback) else input_file

    try:
        df, _, time_col = load_data(input_file)
    except Exception:  # noqa: BLE001
        df = pd.read_csv(input_file, skiprows=[0, 1, 2, 5], header=[0, 1, 2, 3])
        time_col = next(c for c in df.columns if c[3] == "Time (Seconds)")

    groups_all = group_entities(df)
    selected = cfg.get("groups") or list(groups_all.keys())
    selected = [g for g in selected if g in groups_all]

    start_time = cfg.get("interval", "start_time", default=0.0)
    end_time = cfg.get("interval", "end_time")
    times_full = df[time_col].values
    frames_full = np.arange(len(times_full)) + 1
    start = int(np.searchsorted(times_full, start_time, side="left"))
    if end_time == float("inf"):
        end = float("inf")
    else:
        end = int(np.searchsorted(times_full, end_time, side="right")) - 1
    if start >= len(times_full) or (end != float("inf") and end < start):
        start = 0
        end = len(times_full) - 1

    kin_cfg = cfg.get("kinematics") or {}
    smoothing = kin_cfg.get("smoothing", False)
    window = kin_cfg.get("smoothing_window", 5)
    polyorder = kin_cfg.get("smoothing_polyorder", 2)
    method = kin_cfg.get("smoothing_method", "savgol")

    filt_cfg = cfg.get("filtering") or {}
    filter_defs = cfg.get("filter_test", "filters", default=[]) or []

    output: Dict[str, dict] = {}
    for gid in selected:
        if end == float("inf"):
            sub = df.iloc[start:]
            times = times_full[start:]
            frames = frames_full[start:]
        else:
            sub = df.iloc[start : end + 1]
            times = times_full[start : end + 1]
            frames = frames_full[start : end + 1]

        ent_df = sub.xs(gid, level=0, axis=1)
        if "Position" not in ent_df.columns.get_level_values(1):
            continue

        pos = (
            ent_df.xs("Position", level=1, axis=1)
            .droplevel(0, axis=1)[["X", "Y", "Z"]]
            .values
        )

        rot = None
        if "Rotation" in ent_df.columns.get_level_values(1):
            rot = (
                ent_df.xs("Rotation", level=1, axis=1)
                .droplevel(0, axis=1)[["X", "Y", "Z", "W"]]
                .values
            )

        speed, t_v = compute_linear_velocity(
            pos,
            times,
            smoothing=smoothing,
            window=window,
            polyorder=polyorder,
            method=method,
        )

        if rot is not None:
            ang_speed, t_as = compute_angular_speed(
                rot, times, smoothing=smoothing, window=window, polyorder=polyorder
            )
            ang_vel, t_av = compute_angular_velocity(
                rot, times, smoothing=smoothing, window=window, polyorder=polyorder
            )
        else:
            ang_speed, t_as = np.array([]), np.array([])
            ang_vel, t_av = np.zeros((0, 3)), np.array([])

        if filt_cfg.get("enable"):
            start_frames = start + 1
            speed_ranges: List[Tuple[int, int]] = []
            ang_ranges: List[Tuple[int, int]] = []
            pos_ranges: List[Tuple[int, int]] = []

            if (
                filt_cfg.get("speed_lower") is not None
                or filt_cfg.get("speed_upper") is not None
            ):
                speed, speed_ranges = filter_anomalies(
                    speed,
                    start_frames,
                    filt_cfg.get("speed_lower"),
                    filt_cfg.get("speed_upper"),
                )

            if (
                filt_cfg.get("angular_speed_lower") is not None
                or filt_cfg.get("angular_speed_upper") is not None
            ):
                ang_speed, ang_ranges = filter_anomalies(
                    ang_speed,
                    start_frames,
                    filt_cfg.get("angular_speed_lower"),
                    filt_cfg.get("angular_speed_upper"),
                )

            if any(
                filt_cfg.get(f"{axis}_{b}") is not None
                for axis in ("x", "y", "z")
                for b in ("lower", "upper")
            ):
                pos, pos_ranges = filter_position(
                    pos,
                    start,
                    filt_cfg.get("x_lower"),
                    filt_cfg.get("x_upper"),
                    filt_cfg.get("y_lower"),
                    filt_cfg.get("y_upper"),
                    filt_cfg.get("z_lower"),
                    filt_cfg.get("z_upper"),
                )

            ang_speed = apply_ranges(ang_speed, start_frames, speed_ranges)
            speed = apply_ranges(speed, start_frames, ang_ranges)

            if pos_ranges:
                rng_conv = [(max(start_frames, s), e + 1) for s, e in pos_ranges]
                speed = apply_ranges(speed, start_frames, rng_conv)
                ang_speed = apply_ranges(ang_speed, start_frames, rng_conv)

            pos = apply_ranges(pos, start, [(s - 1, e) for s, e in speed_ranges])
            pos = apply_ranges(pos, start, [(s - 1, e) for s, e in ang_ranges])

        speed_filters = apply_filters(speed, t_v, filter_defs)
        ang_filters = apply_filters(ang_speed, t_as, filter_defs)

        markers = []
        for tm in cfg.get("time_markers") or []:
            idx = int(np.searchsorted(times, tm, side="left"))
            if 0 <= idx < len(times):
                markers.append(idx)

        frames_speed = frames[1:] if len(frames) > 1 else np.array([])
        frames_ang = frames[1:] if len(frames) > 1 else np.array([])

        output[gid] = {
            "pos": pos,
            "times": times,
            "frames": frames,
            "speed": speed,
            "t_speed": t_v,
            "frames_speed": frames_speed,
            "ang_speed": ang_speed,
            "t_ang_speed": t_as,
            "frames_ang": frames_ang,
            "ang_vel": ang_vel,
            "t_ang_vel": t_av,
            "markers": markers,
            "speed_filters": speed_filters,
            "ang_speed_filters": ang_filters,
        }

    return output, selected
