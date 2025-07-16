import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_trajectory_2d(pos, times, time_markers, out_path):
    fig, ax = plt.subplots()
    points = pos[:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(times.min(), times.max())
    lc = LineCollection(segments, cmap='rainbow', norm=norm)
    lc.set_array(times[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Time (s)')
    # Add markers on the trajectory
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(pos[tm,0], pos[tm,1], marker='v', s=60, edgecolor='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Trajectory')
    fig.savefig(out_path)
    plt.close(fig)

def plot_trajectory_3d(pos, times, time_markers, out_path):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    segments = np.stack([pos[:-1], pos[1:]], axis=1)
    norm = Normalize(times.min(), times.max())
    lc = Line3DCollection(segments, cmap='rainbow', norm=norm)
    lc.set_array(times[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1, 1, 1))
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Time (s)')
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(
                pos[tm,0], pos[tm,1], pos[tm,2],
                marker='v', s=60, edgecolor='black'
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    fig.savefig(out_path)
    plt.close(fig)

def plot_time_series(values, times, ylabel, time_markers, out_path):
    fig, ax = plt.subplots()
    ax.plot(times, values)
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.axvline(x=times[tm], linestyle='--', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Time")
    fig.savefig(out_path)
    plt.close(fig)