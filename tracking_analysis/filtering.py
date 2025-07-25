# File: tracking_analysis/filtering.py
import pandas as pd
import numpy as np

def filter_missing(df, start_frame, end_frame):
    """
    Identify entities whose X coordinate never changes in the given frame interval.
    Handles end_frame = float('inf') as “up through the last frame.”
    Returns a list of entity names whose Position→X is static.
    """
    # Slice the DataFrame by frame index
    if end_frame == float('inf'):
        sub = df.iloc[start_frame:]
    else:
        sub = df.iloc[start_frame:end_frame + 1]

    missing = []
    # Top-level entity names
    entities = df.columns.get_level_values(0).unique()

    for entity in entities:
        try:
            # 1) select only this entity (drops level 0)
            ent_df    = sub.xs(entity, level=0, axis=1)
            # 2) select the 'Position' measurement (now level 1 of ent_df)
            pos_block = ent_df.xs('Position', level=1, axis=1)
            # 3) drop the ID-level (first level) to leave components X,Y,Z
            pos_df    = pos_block.droplevel(0, axis=1)
            # 4) extract the X-coordinate series
            x_series  = pos_df['X']
        except KeyError:
            # no Position→X for this entity
            continue

        # Robust unique count handling, even if x_series is accidentally a
        # DataFrame due to unexpected column structure. Flatten the values to a
        # 1D array, drop NaNs, and measure the number of unique entries.
        vals = pd.unique(pd.Series(x_series.values.ravel()).dropna())
        if len(vals) <= 1:
            missing.append(entity)

    return missing


def filter_anomalies(values, start_frame, low=None, high=None):
    """Replace values outside the [low, high] range with NaN.

    Parameters
    ----------
    values : array_like
        Numeric data to filter.
    start_frame : int
        Frame index corresponding to ``values[0]``.
    low : float, optional
        Values ``<= low`` are removed when specified.
    high : float, optional
        Values ``>= high`` are removed when specified.

    Returns
    -------
    filtered : ndarray
        Array with out-of-range entries replaced by ``NaN``.
    ranges : list of tuple
        List of ``(start_frame, end_frame)`` ranges for removed sections.
    """
    arr = np.asarray(values, dtype=float)
    filtered = arr.copy()
    ranges = []
    i = 0
    n = len(arr)
    while i < n:
        cond = False
        if low is not None and arr[i] <= low:
            cond = True
        if high is not None and arr[i] >= high:
            cond = True
        if not cond:
            i += 1
            continue

        j = i
        while j < n:
            violate = False
            if low is not None and arr[j] <= low:
                violate = True
            if high is not None and arr[j] >= high:
                violate = True
            if not violate:
                break
            filtered[j] = np.nan
            j += 1
        ranges.append((int(start_frame + i), int(start_frame + j)))
        i = j

    return filtered, ranges


def apply_ranges(values, start_frame, ranges):
    """Apply ``ranges`` of frame indices as ``NaN`` to ``values``.

    Parameters
    ----------
    values : array_like
        Array of numeric values. Modified copy is returned.
    start_frame : int
        Frame index corresponding to ``values[0]``.
    ranges : iterable of tuple
        ``(start_frame, end_frame)`` pairs as produced by :func:`filter_anomalies`.

    Returns
    -------
    ndarray
        Array with selected ranges replaced by ``NaN``.
    """
    arr = np.asarray(values, dtype=float).copy()
    for a, b in ranges:
        i = max(0, int(a - start_frame))
        j = max(0, int(b - start_frame))
        if arr.ndim == 1:
            arr[i:j] = np.nan
        else:
            arr[i:j, ...] = np.nan
    return arr


def compute_stats(values, frames, times):
    """Compute summary statistics for a time series."""
    vals = np.asarray(values, dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return {}
    v = vals[mask]
    f = np.asarray(frames)[mask]
    t = np.asarray(times)[mask]
    stats = {
        'mean': float(v.mean()),
        'sd': float(v.std()),
        'q25': float(np.quantile(v, 0.25)),
        'q75': float(np.quantile(v, 0.75)),
        'q90': float(np.quantile(v, 0.90)),
        'q95': float(np.quantile(v, 0.95)),
        'max': float(v.max()),
        'max_frame': int(f[np.argmax(v)]),
        'max_time': float(t[np.argmax(v)]),
        'min': float(v.min()),
        'min_frame': int(f[np.argmin(v)]),
        'min_time': float(t[np.argmin(v)]),
    }
    return stats


def filter_position(
    pos,
    start_frame,
    x_lower=None,
    x_upper=None,
    y_lower=None,
    y_upper=None,
    z_lower=None,
    z_upper=None,
):
    """Filter position array by coordinate thresholds."""
    arr = np.asarray(pos, dtype=float)
    mask = np.zeros(len(arr), dtype=bool)

    if x_lower is not None:
        mask |= arr[:, 0] <= x_lower
    if x_upper is not None:
        mask |= arr[:, 0] >= x_upper
    if arr.shape[1] > 1:
        if y_lower is not None:
            mask |= arr[:, 1] <= y_lower
        if y_upper is not None:
            mask |= arr[:, 1] >= y_upper
    if arr.shape[1] > 2:
        if z_lower is not None:
            mask |= arr[:, 2] <= z_lower
        if z_upper is not None:
            mask |= arr[:, 2] >= z_upper

    filtered = arr.copy()
    filtered[mask] = np.nan

    ranges = []
    i = 0
    n = len(arr)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        ranges.append((int(start_frame + i), int(start_frame + j)))
        i = j

    return filtered, ranges


def merge_ranges(ranges):
    """Merge overlapping ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for a, b in ranges[1:]:
        m_a, m_b = merged[-1]
        if a <= m_b:
            merged[-1][1] = max(m_b, b)
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged]


def filter_no_moving(speed, start_frame, window=10, after=10):
    """Filter sequences of zero ``speed`` lasting ``window`` frames.

    Parameters
    ----------
    speed : array_like
        Speed values for consecutive frames.
    start_frame : int
        Frame index corresponding to ``speed[0]``.
    window : int, optional
        Minimum length of a zero-velocity block to remove.
    after : int, optional
        Number of additional frames to remove after a block.

    Returns
    -------
    filtered : ndarray
        ``speed`` with zero-velocity blocks replaced by ``NaN``.
    ranges : list of tuple
        ``(start_frame, end_frame)`` ranges of removed sections.
    """

    arr = np.asarray(speed, dtype=float)
    filtered = arr.copy()
    ranges = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] != 0:
            i += 1
            continue
        j = i
        while j < n and arr[j] == 0:
            j += 1
        if j - i >= window:
            end = min(n, j + after)
            filtered[i:end] = np.nan
            ranges.append((int(start_frame + i), int(start_frame + end)))
        i = j

    return filtered, ranges


def _lateral_inhibition(data, tau_fast=2, tau_slow=8, k_inhibit=1.0):
    """Causal difference-of-exponentials filter used by ``apply_filter_chain``."""
    alpha_fast = 1.0 / float(tau_fast)
    alpha_slow = 1.0 / float(tau_slow)

    ema_fast = np.empty_like(data, dtype=float)
    ema_slow = np.empty_like(data, dtype=float)
    ema_fast[0] = data[0]
    ema_slow[0] = data[0]
    for k in range(1, len(data)):
        ema_fast[k] = ema_fast[k - 1] + alpha_fast * (data[k] - ema_fast[k - 1])
        ema_slow[k] = ema_slow[k - 1] + alpha_slow * (data[k] - ema_slow[k - 1])

    return ema_fast - k_inhibit * ema_slow


def apply_filter_chain(x, times, filters):
    """Apply a sequence of filters to ``x``.

    This is shared by the interactive application and comparison tool.
    Supported filter types are ``moving_average``, ``ema``, ``butterworth``,
    ``savgol``, ``window`` and ``lateral_inhibition``.
    """

    if not filters or len(x) == 0:
        return {}

    fs = 1.0 / float(np.mean(np.diff(times))) if len(times) > 1 else 1.0
    results = {}
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
        elif ftype == "lateral_inhibition":
            tau_fast = int(cfg.get("tau_fast", 2))
            tau_slow = int(cfg.get("tau_slow", 8))
            k_inhibit = float(cfg.get("k_inhibit", 1.0))
            y = _lateral_inhibition(x, tau_fast=tau_fast, tau_slow=tau_slow, k_inhibit=k_inhibit)
        else:
            continue

        results[name] = y

    return results

