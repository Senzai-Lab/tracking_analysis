"""Generate SVG plots comparing different filters on a signal from the CSV."""


from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data, preprocess_csv
from tracking_analysis.grouping import group_entities
from tracking_analysis.kinematics import compute_linear_velocity, compute_angular_speed
from interactive_app.data_utils import apply_filters


def _load_signal(cfg: Config) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load a 1-D signal from the configured CSV.

    Returns (times, signal, fs).
    """
    in_path = cfg.get("input_file")
    pre_cfg = cfg.get("preprocess") or {}
    data_path = in_path
    if pre_cfg.get("enable"):
        out_file = pre_cfg.get(
            "output_file",
            os.path.join(cfg.get("output", "output_dir"), "trimmed.csv"),
        )
        summary = pre_cfg.get(
            "summary_file",
            os.path.join(cfg.get("output", "output_dir"), "summary.txt"),
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        os.makedirs(os.path.dirname(summary), exist_ok=True)
        preprocess_csv(in_path, out_file, summary)
        data_path = out_file

    try:
        df, frame_col, time_col = load_data(data_path)
    except Exception:
        df = pd.read_csv(data_path, skiprows=[0, 1, 2, 5], header=[0, 1, 2, 3])
        frame_col = next(c for c in df.columns if c[3] == "Frame")
        time_col = next(c for c in df.columns if c[3] == "Time (Seconds)")

    groups = group_entities(df)
    group = cfg.get("filter_test", "group")
    if not group or group not in groups:
        group = next(iter(groups)) if groups else None
    if group is None:
        raise RuntimeError("No valid group found in data")

    src = cfg.get("filter_test", "source", default="speed").lower()

    start = cfg.get(
        "filter_test",
        "start_time",
        default=cfg.get("interval", "start_time", default=0.0),
    )
    end = cfg.get(
        "filter_test",
        "end_time",
        default=cfg.get("interval", "end_time", default=float("inf")),
    )

    times_all = df[time_col].values
    i0 = int(np.searchsorted(times_all, start, side="left"))
    if end == float("inf"):
        i1 = len(times_all)
    else:
        i1 = int(np.searchsorted(times_all, end, side="right"))

    ent_df = df.xs(group, level=0, axis=1)
    pos = (
        ent_df.xs("Position", level=1, axis=1)
        .droplevel(0, axis=1)[["X", "Y", "Z"]]
        .values
    )[i0:i1]
    times = times_all[i0:i1]

    if src == "speed":
        signal, t = compute_linear_velocity(pos, times)
    elif src == "angular_speed":
        if "Rotation" not in ent_df.columns.get_level_values(1):
            raise ValueError("No rotation data available for angular speed")
        rot = (
                  ent_df.xs("Rotation", level=1, axis=1)
                  .droplevel(0, axis=1)[["X", "Y", "Z", "W"]]
                  .values
              )[i0:i1]
        signal, t = compute_angular_speed(rot, times)
    else:
        comp = {"position_x": 0, "position_y": 1, "position_z": 2}.get(src)
        if comp is None:
            raise ValueError(f"Unknown source '{src}'")
        signal = pos[:, comp]
        t = times

    fs = 1.0 / float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    return t, signal, fs




def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filter comparison plots")
    parser.add_argument("-c", "--config", default="config.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = Config(args.config)
    if not cfg.get("filter_test", "enable", default=False):
        print("filter_test.enable is not set")
        return

    t, base, _ = _load_signal(cfg)

    filters = cfg.get("filter_test", "filters", default=[]) or []
    results = apply_filters(base, t, filters)


    out_dir = cfg.get("output", "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(out_dir, f"filter_test_{stamp}")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(100, 12))
    plt.plot(t, base, label="original")
    for name, arr in results.items():
        plt.plot(t, arr, label=name)
    plt.xlabel("Time (s)")
    plt.title("Filter Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comparison.svg"))
    plt.close()

    for name, arr in results.items():
        plt.figure(figsize=(18, 10))
        plt.plot(t, base, label="original", alpha=0.5)
        plt.plot(t, arr, label=name)
        plt.xlabel("Time (s)")
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{name}.svg"))
        plt.close()

    print(f"Plots written to {plot_dir}")


if __name__ == "__main__":
    main()
