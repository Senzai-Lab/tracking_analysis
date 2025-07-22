# Tracking Analysis

This project provides a small pipeline for analyzing motion tracking CSV files. Configuration is managed entirely through `config.yaml`.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the analysis:
   ```bash
   python -m tracking_analysis.cli --config config.yaml
   ```
   The script will automatically trim the CSV when `preprocess.enable` is `true`.

All plots, trimmed files and statistics are written to the directory specified by `output.output_dir` (default `results/`). The trimming step also produces `summary.txt` describing the input CSV.

Each run creates a timestamped subfolder inside `output.output_dir`. The exact configuration used is saved as `config_used.txt` within that folder.

## Configuration Options

Below is an overview of the keys supported in `config.yaml`.

### `input_file`
Path to the CSV file to process.

### `interval`
`start_time` and `end_time` specify the slice of the recording in seconds. Use `null` for `end_time` to include the entire file.

### `time_markers`
List of timestamps (seconds) to highlight on the plots. Markers are drawn as small red triangles so the same event can be located across all figures.

### `groups`
List of rigid-body base names to analyse. An empty list means all available groups.

### `kinematics`
Options for smoothing the computed velocities (`smoothing`, `smoothing_window`, `smoothing_polyorder`, `smoothing_method`).

### `filtering`
Enable range filtering and define upper/lower thresholds for linear speed, angular speed and X/Y/Z coordinates.

### `output`
Controls which figures and exports are produced. `full_size_plots` enlarges plots to 16x10 inches. `x_limit` and `y_limit` specify the axis maxima for time-series plots (set to `null` for automatic scaling).

### `preprocess`
If `enable` is true, the CSV is trimmed before analysis and a brief `summary_file` is written.

You can also run the trimming step manually:

```bash
python -m tracking_analysis.trim <input.csv> [output.csv] --summary summary.txt
```

## Code Structure

The `interactive_app` package is split into small modules so UI logic, data
handling and plotting remain independent:

- `data_utils.py` contains data loading, preprocessing and filtering helpers.
- `plotting.py` defines the functions that build Plotly figures.
- `ui_components.py` holds the Dash form builders used in the configuration
  panel.
- `utils.py` re-exports these helpers for backward compatibility.

## Interactive Web Application

The Dash-based web interface mirrors the CLI processing pipeline. It reads
`config.yaml`, applies the same filtering, slicing and smoothing settings and
highlights any time markers. Four interactive graphs are displayed:
3D trajectory, 2D trajectory, linear speed and angular speed. A drop-down menu
lists every filter from `filter_test.filters`, matching the options available in
`filter_compare`. Use the time-range slider to focus on a portion of the
recording. The **Play** button replays the trajectory and marks the current time
across all plots. Clicking a point on any graph highlights the same moment on
the others using grey markers. A collapsible *Edit configuration* window lists
every setting from `config.yaml` with simple form controls. Switches toggle
boolean options and inputs update numeric or text values. Saving reloads the data
without restarting the server.


```bash
python -m interactive_app.app --config config.yaml
```

The app listens on the port specified by `webapp.port` (default `3010`).
If `input_file` does not exist the viewer falls back to `data/input.csv`.




## Filter Comparison Utility

The `interactive_app.filter_compare` script loads the CSV defined by
`input_file`, extracts a one‑dimensional signal and saves SVG plots for the
configured filters.
The signal source, time interval and filters are controlled by the
`filter_test` section in `config.yaml`, for example:

```yaml
filter_test:
  enable: true
  group: null       # use first group
  source: speed     # or position_x/position_y/position_z
  start_time: 0
  end_time: null
  filters:
    - type: moving_average
      window: 5
    - type: window
      window: 10
    - type: decimal_removal
      digits: 1
```

The `window` filter averages the first and last half of the
configured window. The `decimal_removal` filter truncates the
signal after scaling it to the -1..1 range before converting back to
degrees.

Run the tool with:

```bash
python -m interactive_app.filter_compare --config config.yaml
```

Generated figures are stored under the `results/` directory.

Legacy callbacks for the old YAML editor remain commented out in
`interactive_app/app.py` for reference. The web interface now exposes a simple
form-based configuration panel instead of a raw text area.
