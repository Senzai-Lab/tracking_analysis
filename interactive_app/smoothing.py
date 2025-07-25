"""Collection of smoothing functions used by the web app."""

from __future__ import annotations

from typing import Callable, Dict
import inspect
import numpy as np

SMOOTHING_FUNCS: Dict[str, Callable[..., np.ndarray]] = {}

def register(name: str) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Decorator to register a smoothing function."""
    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        SMOOTHING_FUNCS[name] = func
        return func
    return decorator


def apply(method: str, data: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a registered smoothing ``method`` to ``data``."""
    func = SMOOTHING_FUNCS.get(method)
    if func is None:
        raise ValueError(f"Unknown smoothing method '{method}'")
    sig = inspect.signature(func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(data, **filtered)


@register("savgol")
def savgol(data: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    from scipy.signal import savgol_filter

    if window % 2 == 0:
        window += 1
    return savgol_filter(data, window, polyorder, axis=0)


@register("ema")
def ema(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.empty_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


@register("window")
def window_avg(data: np.ndarray, window: int = 4) -> np.ndarray:
    """Simple windowed average smoothing."""
    if window < 2:
        return data
    if window % 2 != 0:
        window += 1
    half = window // 2
    if data.ndim == 1:
        padded = np.pad(data, (half, half), mode="edge")
    else:
        padded = np.pad(data, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(data, dtype=float)
    for i in range(len(data)):
        segment = padded[i : i + window]
        out[i] = np.mean(segment, axis=0)
    return out



@register("lateral_inhibition")
def lateral_inhibition(
    data: np.ndarray,
    tau_fast: int = 2,
    tau_slow: int = 8,
    k_inhibit: float = 1.0,
) -> np.ndarray:
    """Causal difference-of-exponentials filter.

    Parameters
    ----------
    data:
        Input signal (1-D or 2-D array where rows are samples).
    tau_fast:
        Time constant of the fast exponential in samples.
    tau_slow:
        Time constant of the slow exponential in samples.
    k_inhibit:
        Strength of the slow component.

    Returns
    -------
    np.ndarray
        Filtered signal of the same shape as ``data``.
    """

    alpha_fast = 1.0 / float(tau_fast)
    alpha_slow = 1.0 / float(tau_slow)

    ema_fast = np.empty_like(data, dtype=float)
    ema_slow = np.empty_like(data, dtype=float)
    ema_fast[0] = data[0]
    ema_slow[0] = data[0]
    for i in range(1, len(data)):
        ema_fast[i] = ema_fast[i - 1] + alpha_fast * (data[i] - ema_fast[i - 1])
        ema_slow[i] = ema_slow[i - 1] + alpha_slow * (data[i] - ema_slow[i - 1])

    out = ema_fast - k_inhibit * ema_slow

    return out
