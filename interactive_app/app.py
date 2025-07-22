"""Dash-based interactive viewer for tracking analysis."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
from dash import Dash, dcc, html, dash_table, callback_context
import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from tracking_analysis.config import Config

from interactive_app.data_utils import build_table, prepare_data, slice_range
from interactive_app.plotting import make_figures
from interactive_app.ui_components import build_config_form


def create_app(cfg: Config) -> Dash:
    """Create Dash application."""
    data, groups = prepare_data(cfg)
    default_gid = groups[0] if groups else None

    app = Dash(__name__)
    filters_cfg = cfg.get("filter_test", "filters", default=[]) or []
    filter_names = [
        "base",
        *[f.get("name", f.get("type", f"f{idx}")) for idx, f in enumerate(filters_cfg)],
    ]
    times_ref = data[default_gid]["times"] if default_gid else np.array([0.0, 1.0])
    t_min, t_max = float(times_ref[0]), float(times_ref[-1])
    slider_step = round(max((t_max - t_min) / 50, 0.1), 1)

    config_children = build_config_form(cfg._cfg)
    app.layout = html.Div(
        [
            html.H2("Interactive Trajectory Viewer"),
            dcc.Dropdown(
                options=[{"label": g, "value": g} for g in groups],
                value=default_gid,
                id="entity-dropdown",
                clearable=False,
            ),
            html.Div(
                dcc.RangeSlider(
                    id="time-range",
                    min=t_min,
                    max=t_max,
                    step=slider_step,
                    value=[t_min, t_max],
                    allowCross=False,
                    tooltip={"placement": "bottom"},
                ),
                style={"width": "100%"},
            ),
            dcc.Dropdown(
                id="filter-dropdown",
                options=[{"label": n, "value": n} for n in filter_names],
                value="base",
                clearable=False,
                style={"width": "200px", "margin": "10px 0"},
            ),
            html.Div(
                [
                    html.Button("Play", id="play-btn", n_clicks=0),
                    dcc.Interval(id="play-int", interval=1000, disabled=True),
                    html.Div(id="status-bar", children="Ready"),


                ],
                style={"display": "flex", "gap": "10px", "alignItems": "center"},
            ),
            html.Div(
                [
                    dcc.Graph(id="traj3d", style={"flex": "1", "height": "600px"}),

                  dcc.Graph(
                        id="traj2d",
                        style={"flex": "1", "height": "600px", "width": "600px"},
                    ),


                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            ),
            dcc.Graph(id="speed", style={"width": "100%", "height": "400px"}),
            dcc.Graph(id="angular", style={"width": "100%", "height": "400px"}),
            html.Button("Show Table", id="toggle-table", n_clicks=0),
            html.Button("Edit Config", id="toggle-config", n_clicks=0),
            html.Div(
                dash_table.DataTable(
                    id="raw-table",
                    columns=[
                        {"name": n, "id": n}
                        for n in [
                            "frame",
                            "time",
                            "x",
                            "y",
                            "z",
                            "speed",
                            "angular_speed",
                        ]
                    ],
                    page_size=10,
                ),
                id="table-container",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "✕",

                                id="close-config",
                                n_clicks=0,
                                style={"float": "right"},
                            ),
                            html.H4("Configuration"),
                            html.Div(id="config-form", children=config_children),
                            html.Button("Save", id="save-config"),

                            html.Div(id="save-status"),
                        ],
                        style={
                            "background": "white",
                            "padding": "15px",
                            "width": "320px",
                            "maxHeight": "80vh",
                            "overflowY": "auto",
                            "fontFamily": "sans-serif",

                        },
                    )
                ],
                id="config-modal",
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "background": "rgba(0,0,0,0.4)",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "zIndex": "1000",
                },
            ),
            html.Pre(id="info", children="Hover or click on any plot for details"),
            dcc.Store(id="selected-time"),
        ]
    )

    @app.callback(
        Output("traj3d", "figure"),
        Output("traj2d", "figure"),
        Output("speed", "figure"),
        Output("angular", "figure"),
        Output("raw-table", "data"),
        Output("status-bar", "children"),
        Input("entity-dropdown", "value"),
        Input("time-range", "value"),
        Input("filter-dropdown", "value"),
        Input("selected-time", "data"),
    )
    def _update_plots(selected_id, t_range, filt_name, sel_time):
        if not selected_id:
            empty = go.Figure()
            return empty, empty, empty, empty, [], "No data"
        d = data[selected_id]
        start, end = t_range or [d["times"][0], d["times"][-1]]
        sl = slice_range(d["times"], start, end)
        sl_v = slice_range(d["t_speed"], start, end)
        sl_a = slice_range(d["t_ang_speed"], start, end)
        spd = d["speed"]
        ang = d["ang_speed"]
        if filt_name and filt_name != "base":
            spd = d["speed_filters"].get(filt_name, spd)
            ang = d["ang_speed_filters"].get(filt_name, ang)

        figs = make_figures(
            d["pos"][sl],
            d["times"][sl],
            d["frames"][sl],
            [m - sl.start for m in d["markers"] if sl.start <= m < sl.stop],
            spd[sl_v],
            d["t_speed"][sl_v],
            d["frames_speed"][sl_v],
            ang[sl_a],
            d["t_ang_speed"][sl_a],
            d["frames_ang"][sl_a],
            sel_time,
        )
        table = build_table(d, start, end)
        return (*figs, table, "Updated")


    @app.callback(
        Output("play-int", "disabled"),
        Output("play-btn", "children"),
        Output("selected-time", "data", allow_duplicate=True),
        Input("play-btn", "n_clicks"),
        State("play-int", "disabled"),
        State("time-range", "value"),
        prevent_initial_call=True,
    )
    def _toggle_play(n, disabled, t_range):
        disabled = not disabled
        start = t_range[0] if t_range else 0.0
        return disabled, ("Play" if disabled else "Pause"), (start if not disabled else dash.no_update)


    @app.callback(
        Output("time-range", "value", allow_duplicate=True),
        Input("speed", "relayoutData"),
        Input("angular", "relayoutData"),
        Input("traj3d", "relayoutData"),
        Input("traj2d", "relayoutData"),
        State("time-range", "min"),
        State("time-range", "max"),
        prevent_initial_call=True,
    )
    def _sync_slider(r_speed, r_ang, r3d, r2d, tmin, tmax):

        rdata = r_speed or r_ang or r3d or r2d
        if not rdata:
            raise PreventUpdate
        if "xaxis.autorange" in rdata:
            return [tmin, tmax]

        if "xaxis.range" in rdata:
            x0, x1 = rdata["xaxis.range"]
        elif "xaxis.range[0]" in rdata:
            x0 = rdata["xaxis.range[0]"]
            x1 = rdata["xaxis.range[1]"]
        else:
            raise PreventUpdate
        return [float(x0), float(x1)]

    @app.callback(
        Output("selected-time", "data", allow_duplicate=True),
        Input("play-int", "n_intervals"),
        State("selected-time", "data"),
        State("time-range", "value"),
        prevent_initial_call=True,
    )
    def _advance_time(_, current, trange):
        start, end = trange
        step = round(max((end - start) / 200, 0.05), 2)
        pos = start if current is None else float(current)
        nxt = pos + step
        if nxt >= end:
            return end
        return nxt


    @app.callback(
        Output("table-container", "style"),
        Input("toggle-table", "n_clicks"),
        State("table-container", "style"),
        prevent_initial_call=True,
    )
    def _toggle_table(n, style):
        disp = style.get("display", "block")
        return {"display": "none" if disp != "none" else "block"}

    @app.callback(
        Output("info", "children"),
        Output("selected-time", "data", allow_duplicate=True),
        Input("traj3d", "hoverData"),
        Input("traj2d", "hoverData"),
        Input("speed", "hoverData"),
        Input("angular", "hoverData"),
        Input("traj3d", "clickData"),
        Input("traj2d", "clickData"),
        Input("speed", "clickData"),
        Input("angular", "clickData"),
        State("entity-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _update_info(h3d_h, h2d_h, hs_h, ha_h, h3d_c, h2d_c, hs_c, ha_c, gid):
        if not gid:
            raise PreventUpdate
        trig = callback_context.triggered_id
        d = data[gid]
        info = "Hover or click on any plot"
        t_val = dash.no_update
        def select_point(pnt, times_source, frames_source, pos_source=None):
            if not pnt:
                return None
            pt = pnt["points"][0]
            idx = pt.get("pointIndex")
            if idx is None:
                idx = pt.get("pointNumber")
            if idx is None:
                return None
            t = times_source[idx] if idx < len(times_source) else float("nan")
            frame = frames_source[idx] if idx < len(frames_source) else float("nan")
            if pos_source is not None:
                pos = pos_source[idx]
                return (
                    f"Frame: {int(frame)}\nTime: {float(t):.3f}s\nX: {pos[0]:.3f}\nY: {pos[1]:.3f}\nZ: {pos[2]:.3f}",
                    float(t),
                )
            else:
                val = pt.get("y", float("nan"))
                label = "Speed" if times_source is d["t_speed"] else "Angular Speed"
                return (
                    f"Frame: {int(frame)}\nTime: {float(t):.3f}s\n{label}: {float(val):.3f}",
                    float(t),
                )

        if trig == "traj3d":
            res = select_point(h3d_c or h3d_h, d["times"], d["frames"], d["pos"])
        elif trig == "traj2d":
            res = select_point(h2d_c or h2d_h, d["times"], d["frames"], d["pos"])
        elif trig == "speed":
            res = select_point(hs_c or hs_h, d["t_speed"], d["frames_speed"])
        elif trig == "angular":
            res = select_point(ha_c or ha_h, d["t_ang_speed"], d["frames_ang"])
        else:
            res = None
        if res:
            info, t_val = res
        return info, t_val


    @app.callback(
        Output("config-modal", "style"),
        Input("toggle-config", "n_clicks"),
        Input("close-config", "n_clicks"),
        State("config-modal", "style"),
        prevent_initial_call=True,
    )
    def _toggle_config(open_n, close_n, style):
        disp = style.get("display", "none")
        trig = callback_context.triggered_id
        if trig == "toggle-config":
            style["display"] = "flex" if disp == "none" else "none"
        else:
            style["display"] = "none"
        return style

    @app.callback(
        Output({"type": "cfg-toggle", "key": MATCH}, "children"),
        Output({"type": "cfg-toggle", "key": MATCH}, "style"),
        Input({"type": "cfg-toggle", "key": MATCH}, "n_clicks"),
        State({"type": "cfg-toggle", "key": MATCH}, "children"),
        prevent_initial_call=True,
    )
    def _toggle_generic(_, state):
        enable = state == "Off"
        return ("On" if enable else "Off"), {"color": "black" if enable else "grey"}


    @app.callback(
        Output("selected-time", "data", allow_duplicate=True),
        Input("time-range", "value"),
        State("play-int", "disabled"),
        prevent_initial_call=True,
    )
    def _sync_selected(val, disabled):

        if not disabled:
            raise PreventUpdate
        return val[0] if val else None


    @app.callback(
        Output("entity-dropdown", "options"),
        Output("entity-dropdown", "value"),
        Output("time-range", "min"),
        Output("time-range", "max"),
        Output("time-range", "step"),
        Output("time-range", "value", allow_duplicate=True),
        Output("filter-dropdown", "options"),
        Output("selected-time", "data", allow_duplicate=True),
        Output("config-form", "children"),
        Output("save-status", "children"),
        Input("save-config", "n_clicks"),
        State({"type": "cfg-input", "key": ALL}, "value"),
        State({"type": "cfg-input", "key": ALL}, "id"),
        State({"type": "cfg-toggle", "key": ALL}, "children"),
        State({"type": "cfg-toggle", "key": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _save_config(_, values, ids, toggles, tid):
        nonlocal data, groups, filter_names

        for val, ident in zip(values, ids):
            path = ident["key"].split(".")
            node = cfg._cfg
            for p in path[:-1]:
                node = node.setdefault(p, {})
            key = path[-1]
            orig = node.get(key)
            if isinstance(orig, bool):
                node[key] = bool(val)
            elif isinstance(orig, int) and not isinstance(orig, bool):
                try:
                    node[key] = int(val)
                except (TypeError, ValueError):
                    node[key] = orig
            elif isinstance(orig, float):
                try:
                    node[key] = float(val)
                except (TypeError, ValueError):
                    node[key] = orig
            elif isinstance(orig, list):
                node[key] = [v.strip() for v in str(val).split(",") if v.strip()]
            else:
                node[key] = None if val in ("", None) else val

        for text, ident in zip(toggles, tid):
            path = ident["key"].split(".")
            node = cfg._cfg
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = text == "On"

        cfg._normalize()


        data, groups = prepare_data(cfg)
        default_gid = groups[0] if groups else None
        filters_cfg = cfg.get("filter_test", "filters", default=[]) or []
        filter_names = [
            "base",
            *[f.get("name", f.get("type", f"f{idx}")) for idx, f in enumerate(filters_cfg)],
        ]


        times_ref = data[default_gid]["times"] if default_gid else np.array([0.0, 1.0])
        t_min, t_max = float(times_ref[0]), float(times_ref[-1])
        slider_step = round(max((t_max - t_min) / 50, 0.1), 1)

        out_dir = cfg.get("output", "output_dir")
        os.makedirs(out_dir, exist_ok=True)
        name = f"web_saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(cfg.as_yaml())

        dropdown_opts = [{"label": g, "value": g} for g in groups]
        filter_opts = [{"label": n, "value": n} for n in filter_names]

        form = build_config_form(cfg._cfg)

        return (
            dropdown_opts,
            default_gid,
            t_min,
            t_max,
            slider_step,
            [t_min, t_max],
            filter_opts,
            None,
            form,
            f"Saved to {path}",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive trajectory viewer")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    cfg = Config(args.config)
    port = cfg.get("webapp", "port", default=3010)
    app = create_app(cfg)
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
