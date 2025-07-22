"""Dash UI component helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from dash import dcc, html


def build_config_form(cfg_dict: Dict[str, Any], prefix: str = "") -> List[Any]:
    """Recursively convert a config dictionary into form components."""
    fields: List[Any] = []
    for key, value in cfg_dict.items():
        full = f"{prefix}{key}"
        comp_id = {"type": "cfg-input", "key": full}
        if isinstance(value, dict):
            children = build_config_form(value, prefix=f"{full}.")
            fields.append(
                html.Details([
                    html.Summary(key),
                    html.Div(children, style={"marginLeft": "15px"}),
                ])
            )
        elif isinstance(value, bool):
            fields.append(
                html.Div(
                    [
                        html.Span(key, style={"marginRight": "10px"}),
                        html.Button(
                            "On" if value else "Off",
                            id={"type": "cfg-toggle", "key": full},
                            n_clicks=0,
                            style={"color": "black" if value else "grey"},
                        ),
                    ],
                    style={"marginBottom": "6px"},
                )
            )
        elif isinstance(value, (int, float)):
            fields.append(
                html.Div(
                    [
                        html.Label(key),
                        dcc.Input(value=value, type="number", id=comp_id, style={"marginLeft": "5px"}),
                    ],
                    style={"marginBottom": "6px"},
                )
            )
        elif isinstance(value, list):
            fields.append(
                html.Div(
                    [
                        html.Label(key),
                        dcc.Input(
                            value=",".join(str(v) for v in value),
                            id=comp_id,
                            style={"marginLeft": "5px"},
                        ),
                    ],
                    style={"marginBottom": "6px"},
                )
            )
        else:
            fields.append(
                html.Div(
                    [
                        html.Label(key),
                        dcc.Input(value=value or "", id=comp_id, style={"marginLeft": "5px"}),
                    ],
                    style={"marginBottom": "6px"},
                )
            )
    return fields
