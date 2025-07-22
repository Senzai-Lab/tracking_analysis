This repository contains utilities for analyzing tracking CSV files.

* Keep code modular and easy to extend.
* Configure all behaviour through `config.yaml`. Avoid interactive prompts.
* Place all generated files under the `results/` folder.
* The preprocessing/trimming step is controlled by the `preprocess` section of the config.
* The summary file should contain a concise description of the input CSV (frames, duration, etc.).

* Ignore `output1.csv` if it appears – this file is for internal testing only.


* IGNORE output1.csv if exists. That's a internal testing file shouldn't be uploaded.
* Do not modify the `tracking_analysis` package; make changes only within `interactive_app`.
* The web application code is split into small modules:
  * `data_utils.py` handles data loading and filtering
  * `plotting.py` contains Plotly figure builders
  * `ui_components.py` creates Dash form elements
  * `utils.py` re-exports these helpers for backward compatibility
