# -*- coding: utf-8 -*-
import logging
import math
import urllib.parse

import dash
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

from app import app
from components import (
    _build_detail_panel,
    _build_tanimoto_heatmap_figure,
    _format_workflow_label,
    _net_spinner,
    _render_mol_png_data_uri,
    _search_pubchem_candidates,
)
from bin.normalizer import _find_first_column, _infer_resolved_workflow, _truncate_text
from bin.constants import SUPPORTED_WORKFLOWS, PPM_SCALE, MZ_PRECISION_FLOOR
import io

import pandas as pd

from bin.loader import (
    _load_library_matches,
    _load_task_information,
    _compute_scores_cached,
)
from bin.scorer import DEFAULT_SCORE_WEIGHTS, _normalize_score_weights

dash_app = dash.Dash(
    name="dashinterface2",
    server=app,
    url_base_pathname="/dashinterface2/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

dash_app.title = "GNPS2-AnnEx"

logger = logging.getLogger(__name__)

LAPLACE_ALPHA = 4.0
LAPLACE_BETA = 5.0
ENABLE_DEV_SCORE_WEIGHT_CONTROLS = True

_SCORE_WEIGHT_INPUT_DEFS = [
    {
        "key": "structure_agreement",
        "id": "score-weight-structure",
        "label": "Structure agreement",
    },
    {"key": "tanimoto", "id": "score-weight-tanimoto", "label": "Tanimoto"},
    {
        "key": "cosine_consistency",
        "id": "score-weight-spectral",
        "label": "Cosine consistency",
    },
    {"key": "max_cosine", "id": "score-weight-max-cosine", "label": "Max cosine"},
    {"key": "shared_peaks", "id": "score-weight-shared-peaks", "label": "Shared peaks"},
]
_SCORE_WEIGHT_QUERY_PARAMS = {
    "structure_agreement": "w_structure",
    "tanimoto": "w_tanimoto",
    "cosine_consistency": "w_spectral",
    "max_cosine": "w_cosine",
    "shared_peaks": "w_shared",
}

# Column definitions for the summary AG Grid table.
# Kept at module level so the column-visibility callback can reference them
# without re-building the full spec on every toggle.
_SUMMARY_COLUMN_DEFS = [
    {
        "headerName": "Scan",
        "field": "scan",
        "maxWidth": 95,
        "type": "numericColumn",
        "filter": "agNumberColumnFilter",
        "valueGetter": {"function": "Number(params.data.scan)"},
    },
    {"headerName": "Prec. m/z", "field": "precursor_mz", "maxWidth": 110},
    {
        "headerName": "Proposed Compound",
        "field": "compound_link",
        "cellRenderer": "markdown",
        "linkTarget": "_blank",
        "minWidth": 180,
        "headerTooltip": "🔁 same structure key proposed for multiple scans.\n⚡ amber highlight = not the top-cosine match for this scan (hover for top-cosine compound name).",
        "tooltipField": "top_cosine_compound",
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.data.top_cosine_overridden === true",
                    "style": {
                        "backgroundColor": "#fff3cd",
                        "color": "#664d03",
                        "fontWeight": 600,
                    },
                }
            ]
        },
    },
    {
        "headerName": "Structure Support",
        "headerTooltip": "Fraction of library hits supporting the proposed annotation / total matches (based on structure key).",
        "field": "support",
        "width": 100,
        "minWidth": 100,
        "cellStyle": {
            "textAlign": "center",
        },
    },
    {
        "headerName": "Adjusted Support",
        "field": "adjusted_support_score",
        "headerTooltip": f"Adjusted support score with additive (Laplace) smoothing to mitigate small-sample effects.\n alpha = number of supporting matches added; beta = alpha + number of non-supporting matches added. α={LAPLACE_ALPHA:g}, β={LAPLACE_BETA:g}",
        "width": 80,
        "minWidth": 80,
        "maxWidth": 190,
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.value >= 0.9",
                    "style": {
                        "backgroundColor": "#0f5132",
                        "color": "#d1e7dd",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value >= 0.8 && params.value < 0.9",
                    "style": {
                        "backgroundColor": "#52b788",
                        "color": "#ffffff",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value >= 0.7 && params.value < 0.8",
                    "style": {
                        "backgroundColor": "#d1e7dd",
                        "color": "#0f5132",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value >= 0.55 && params.value < 0.7",
                    "style": {
                        "backgroundColor": "#fff3cd",
                        "color": "#664d03",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value >= 0.4 && params.value < 0.55",
                    "style": {
                        "backgroundColor": "#ffe5cc",
                        "color": "#7c3c00",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value < 0.4",
                    "style": {
                        "backgroundColor": "#f8d7da",
                        "color": "#842029",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
            ],
            "defaultStyle": {"fontWeight": 600, "textAlign": "center"},
        },
    },
    {
        "headerName": "Score",
        "field": "confidence_score",
        "maxWidth": 95,
        "cellStyle": {"fontWeight": 600, "textAlign": "center"},
    },
    {
        "headerName": "Evidence Level",
        "field": "confidence_label",
        "maxWidth": 170,
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.value === 'Consistent evidence'",
                    "style": {
                        "backgroundColor": "#d1e7dd",
                        "color": "#0f5132",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value === 'Inconclusive'",
                    "style": {
                        "backgroundColor": "#fff3cd",
                        "color": "#664d03",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value === 'Inconsistent evidence'",
                    "style": {
                        "backgroundColor": "#f8d7da",
                        "color": "#842029",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
            ],
            "defaultStyle": {"fontWeight": 600, "textAlign": "center"},
        },
    },
    {
        "headerName": "Input Spectrum Entropy (S)",
        "field": "query_entropy",
        "maxWidth": 135,
        "headerTooltip": "Spectral entropy of the query spectrum (nats). Higher S = more uniform peak distribution. Thresholds: <0.5 noisy, 0.5–1.0 moderate, 1.0–1.75 clean, ≥1.75 very diffuse.",
        "tooltipField": "query_entropy_label",
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.value !== null && params.value < 0.5",
                    "style": {
                        "backgroundColor": "#f8d7da",
                        "color": "#842029",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value !== null && params.value >= 0.5 && params.value < 1.0",
                    "style": {
                        "backgroundColor": "#fff3cd",
                        "color": "#664d03",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value !== null && params.value >= 1.0 && params.value < 1.75",
                    "style": {
                        "backgroundColor": "#d1e7dd",
                        "color": "#0f5132",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
                {
                    "condition": "params.value !== null && params.value >= 1.75",
                    "style": {
                        "backgroundColor": "#cff4fc",
                        "color": "#055160",
                        "fontWeight": 600,
                        "textAlign": "center",
                    },
                },
            ],
            "defaultStyle": {"textAlign": "center"},
        },
    },
    {"field": "query_entropy_label", "hide": True, "headerName": "Entropy Label"},
    {
        "field": "top_cosine_overridden",
        "hide": True,
        "headerName": "Top Cosine Overridden",
    },
    {"field": "top_cosine_compound", "hide": True, "headerName": "Top Cosine Compound"},
    {
        "headerName": "Representative Structure Key",
        "field": "structure_key",
        "minWidth": 220,
    },
    {
        "headerName": "Tanimoto (ECFP4)",
        "field": "tanimoto_score",
        "maxWidth": 120,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Mean Cosine",
        "field": "mean_cosine",
        "maxWidth": 120,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Max Cosine",
        "field": "max_cosine",
        "maxWidth": 120,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Cosine Consistency",
        "field": "cosine_consistency",
        "maxWidth": 140,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Tanimoto Source",
        "field": "tanimoto_source",
        "minWidth": 180,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Consistency Source",
        "field": "cosine_consistency_source",
        "minWidth": 165,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Missing structures (no SMILES/InChIKey)",
        "field": "missing_structures",
        "maxWidth": 145,
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Shared Peaks (mean)",
        "field": "shared_peaks",
        "maxWidth": 130,
        "headerTooltip": "Mean number of peaks shared between the query spectrum and library hits in the top support group. Higher values increase confidence.",
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "Median |MZError| (ppm)",
        "field": "median_abs_ppm_error",
        "maxWidth": 145,
        "headerTooltip": "Median absolute precursor m/z error (ppm) across hits in the top support group. Lower values indicate better mass accuracy.",
        "cellStyle": {"textAlign": "center"},
    },
    {
        "headerName": "MZ Precision Factor",
        "field": "mz_precision",
        "maxWidth": 145,
        "headerTooltip": "Multiplicative factor applied to the base score based on MZErrorPPM. exp(-|ppm| / 10), floored at 0.5. 1.0 = no penalty.",
        "cellStyle": {"textAlign": "center"},
    },
]

# Fields hidden by default (diagnostic / advanced columns).
_DEFAULT_HIDDEN_FIELDS = {
    "tanimoto_source",
    "cosine_consistency_source",
    "query_entropy_label",
    "mean_cosine",
    "max_cosine",
    "cosine_consistency",
    "median_abs_ppm_error",
    "mz_precision",
    "shared_peaks_score",
    "top_cosine_overridden",
    "top_cosine_compound",
}


def _taskinfo_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _taskinfo_has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _stringify_taskinfo_value(value, max_length=160):
    if not _taskinfo_has_value(value):
        return ""

    if isinstance(value, dict):
        text = ", ".join(f"{key}={val}" for key, val in list(value.items())[:4])
    elif isinstance(value, (list, tuple, set)):
        if all(not isinstance(item, (dict, list, tuple, set)) for item in value):
            text = ", ".join(str(item) for item in value)
        else:
            text = f"{len(value)} item(s)"
    else:
        text = str(value)

    return _truncate_text(text, max_length=max_length)


def _iter_taskinfo_items(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield key, value
            yield from _iter_taskinfo_items(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_taskinfo_items(item)


def _find_taskinfo_value(task_info, candidate_keys):
    if not isinstance(task_info, dict):
        return ""

    normalized_keys = {_taskinfo_key(key) for key in candidate_keys}

    for key, value in _iter_taskinfo_items(task_info):
        if _taskinfo_key(key) in normalized_keys and _taskinfo_has_value(value):
            return value

    return ""


def _build_task_metadata_details(task_info, resolved_workflow):
    if not isinstance(task_info, dict) or not task_info:
        return None

    task_name = _find_taskinfo_value(
        task_info, ["taskname", "task_name", "title", "description"]
    )
    status_value = _find_taskinfo_value(
        task_info, ["task_status", "taskstatus", "status"]
    )
    workflow_value = _find_taskinfo_value(
        task_info, ["workflowname", "workflow", "workflowtype"]
    )
    created_value = _find_taskinfo_value(
        task_info,
        [
            "created",
            "create_time",
            "createtime",
            "creationtime",
            "submitted",
            "submittedon",
            "starttime",
            "createdon",
        ],
    )

    summary_rows = []
    for label, value in [
        ("Task name", task_name),
        ("Workflow", workflow_value or _format_workflow_label(resolved_workflow)),
        ("Created", created_value),
    ]:
        if _taskinfo_has_value(value):
            summary_rows.append(
                html.Div(
                    [
                        html.Span(f"{label}: ", className="fw-semibold"),
                        html.Span(_stringify_taskinfo_value(value)),
                    ],
                    className="mb-1",
                )
            )

    if not summary_rows:
        return None

    return html.Details(
        [
            html.Summary("GNPS task information", className="fw-semibold"),
            html.Div(summary_rows, className="small mt-2"),
        ],
        className="mt-3",
    )


def _get_url_param(param_dict, key, default=""):
    return param_dict.get(key, [default])[0]


def _format_weight_percent(weight):
    weight_percent = round(float(weight) * 100.0, 1)
    if weight_percent.is_integer():
        return f"{int(weight_percent)}%"
    return f"{weight_percent:.1f}%"


def _score_weight_inputs_from_weights(score_weights=None):
    normalized_score_weights = _normalize_score_weights(score_weights)
    return {
        key: round(normalized_score_weights[key] * 100.0, 1)
        for key in DEFAULT_SCORE_WEIGHTS
    }


def _parse_score_weights_from_inputs(
    structure_value,
    tanimoto_value,
    spectral_value,
    max_cosine_value,
    shared_peaks_value=None,
):
    input_values = {
        "structure_agreement": structure_value,
        "tanimoto": tanimoto_value,
        "cosine_consistency": spectral_value,
        "max_cosine": max_cosine_value,
        "shared_peaks": shared_peaks_value,
    }
    parsed_weights = {}
    for key, default_value in DEFAULT_SCORE_WEIGHTS.items():
        raw_value = input_values.get(key)
        try:
            parsed_value = float(raw_value)
        except (TypeError, ValueError):
            parsed_value = default_value * 100.0
        parsed_weights[key] = max(0.0, parsed_value) / 100.0
    return _normalize_score_weights(parsed_weights)


def _parse_score_weights_from_query(query_dict):
    parsed_weights = {}
    for key, param_name in _SCORE_WEIGHT_QUERY_PARAMS.items():
        raw_value = _get_url_param(query_dict, param_name, "").strip()
        if not raw_value:
            continue
        try:
            parsed_weights[key] = max(0.0, float(raw_value)) / 100.0
        except ValueError:
            continue
    return _normalize_score_weights(parsed_weights if parsed_weights else None)


def _score_weights_are_default(score_weights=None):
    normalized_score_weights = _normalize_score_weights(score_weights)
    return all(
        abs(normalized_score_weights[key] - DEFAULT_SCORE_WEIGHTS[key]) < 1e-9
        for key in DEFAULT_SCORE_WEIGHTS
    )


def _serialize_score_weights_for_query(score_weights=None):
    normalized_score_weights = _normalize_score_weights(score_weights)
    if _score_weights_are_default(normalized_score_weights):
        return ""

    query_parts = []
    for key, param_name in _SCORE_WEIGHT_QUERY_PARAMS.items():
        value_text = f"{normalized_score_weights[key] * 100.0:.3f}".rstrip("0").rstrip(
            "."
        )
        query_parts.append(f"&{param_name}={urllib.parse.quote(value_text)}")
    return "".join(query_parts)


def _parse_ppm_from_query(query_dict):
    ppm_scale_raw = _get_url_param(query_dict, "ppm_scale", "").strip()
    ppm_floor_raw = _get_url_param(query_dict, "ppm_floor", "").strip()
    try:
        ppm_scale = float(ppm_scale_raw) if ppm_scale_raw else PPM_SCALE
        if not (math.isfinite(ppm_scale) and ppm_scale > 0):
            ppm_scale = PPM_SCALE
    except (ValueError, TypeError):
        ppm_scale = PPM_SCALE
    try:
        ppm_floor = float(ppm_floor_raw) if ppm_floor_raw else MZ_PRECISION_FLOOR
        if not (math.isfinite(ppm_floor) and 0.0 <= ppm_floor <= 1.0):
            ppm_floor = MZ_PRECISION_FLOOR
    except (ValueError, TypeError):
        ppm_floor = MZ_PRECISION_FLOOR
    return ppm_scale, ppm_floor


def _parse_ppm_from_inputs(ppm_scale_value, ppm_floor_value):
    try:
        ppm_scale = float(ppm_scale_value) if ppm_scale_value is not None else PPM_SCALE
        if not (math.isfinite(ppm_scale) and ppm_scale > 0):
            ppm_scale = PPM_SCALE
    except (ValueError, TypeError):
        ppm_scale = PPM_SCALE
    try:
        ppm_floor = (
            float(ppm_floor_value)
            if ppm_floor_value is not None
            else MZ_PRECISION_FLOOR
        )
        if not (math.isfinite(ppm_floor) and 0.0 <= ppm_floor <= 1.0):
            ppm_floor = MZ_PRECISION_FLOOR
    except (ValueError, TypeError):
        ppm_floor = MZ_PRECISION_FLOOR
    return ppm_scale, ppm_floor


def _ppm_params_are_default(ppm_scale, ppm_floor):
    return (
        abs(float(ppm_scale) - PPM_SCALE) < 1e-6
        and abs(float(ppm_floor) - MZ_PRECISION_FLOOR) < 1e-6
    )


def _serialize_ppm_for_query(ppm_scale, ppm_floor):
    suffix = ""
    if not _ppm_params_are_default(ppm_scale, ppm_floor):
        if abs(float(ppm_scale) - PPM_SCALE) >= 1e-6:
            suffix += f"&ppm_scale={float(ppm_scale):.6g}"
        if abs(float(ppm_floor) - MZ_PRECISION_FLOOR) >= 1e-6:
            suffix += f"&ppm_floor={float(ppm_floor):.6g}"
    return suffix


def _format_score_weight_summary(score_weights=None):
    normalized_score_weights = _normalize_score_weights(score_weights)
    labels = {item["key"]: item["label"] for item in _SCORE_WEIGHT_INPUT_DEFS}
    parts = [
        f"{labels[key]} {_format_weight_percent(normalized_score_weights[key])}"
        for key in DEFAULT_SCORE_WEIGHTS
    ]
    return "Score weights: " + " · ".join(parts)


def _build_score_weight_controls():
    input_defaults = _score_weight_inputs_from_weights()
    control_columns = []
    for control in _SCORE_WEIGHT_INPUT_DEFS:
        control_columns.append(
            dbc.Col(
                [
                    dbc.Label(
                        control["label"],
                        html_for=control["id"],
                        className="small fw-semibold text-uppercase text-muted mb-1",
                    ),
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id=control["id"],
                                type="number",
                                min=0,
                                step=0.5,
                                value=input_defaults[control["key"]],
                                style={"borderColor": "rgba(0,0,0,0.12)"},
                            ),
                            dbc.InputGroupText("%"),
                        ],
                        size="sm",
                    ),
                ],
                md=3,
            )
        )

    ppm_row = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label(
                        "PPM penalty scale",
                        html_for="ppm-scale",
                        className="small fw-semibold text-uppercase text-muted mb-1",
                    ),
                    dbc.Input(
                        id="ppm-scale",
                        type="number",
                        min=1,
                        max=200,
                        step=0.5,
                        value=PPM_SCALE,
                        style={"borderColor": "rgba(0,0,0,0.12)"},
                        size="sm",
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label(
                        "PPM penalty floor",
                        html_for="ppm-floor",
                        className="small fw-semibold text-uppercase text-muted mb-1",
                    ),
                    dbc.Input(
                        id="ppm-floor",
                        type="number",
                        min=0,
                        max=1,
                        step=0.05,
                        value=MZ_PRECISION_FLOOR,
                        style={"borderColor": "rgba(0,0,0,0.12)"},
                        size="sm",
                    ),
                ],
                md=3,
            ),
        ],
        className="g-3 mt-2",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Score weight tuning", className="fw-semibold"),
                            dbc.Badge(
                                "Dev only",
                                color="warning",
                                text_color="dark",
                                className="ms-2",
                            ),
                        ],
                        className="d-flex align-items-center flex-wrap",
                    ),
                    html.Small(
                        "Relative weights used for annotation scoring. Values are normalized automatically when scores are computed.",
                        className="text-muted d-block mt-1",
                    ),
                ]
            ),
            dbc.Row(control_columns, className="g-3 mt-1"),
            html.Hr(className="my-2", style={"borderColor": "rgba(0,0,0,0.08)"}),
            html.Small(
                "PPM error penalization — applied as a multiplicative factor: exp(−|ppm| / scale), floored at the floor value.",
                className="text-muted",
            ),
            ppm_row,
        ],
        className="mt-3 p-3 border rounded-4",
        style={
            "background": "rgba(248,249,250,0.9)",
            "borderColor": "rgba(0,0,0,0.08)",
            "display": "block" if ENABLE_DEV_SCORE_WEIGHT_CONTROLS else "none",
        },
    )


def _parse_dashboard_query(search):
    try:
        query_dict = urllib.parse.parse_qs((search or "").lstrip("?"))
    except Exception:
        query_dict = {}

    workflow_choice = (
        _get_url_param(query_dict, "workflow", "auto").strip().lower() or "auto"
    )
    if workflow_choice not in SUPPORTED_WORKFLOWS:
        workflow_choice = "auto"

    qe_raw = _get_url_param(query_dict, "qe", "").strip()
    try:
        precomputed_query_entropy = float(qe_raw) if qe_raw else None
    except ValueError:
        precomputed_query_entropy = None

    score_weights = _parse_score_weights_from_query(query_dict)
    ppm_scale, ppm_floor = _parse_ppm_from_query(query_dict)

    return {
        "task_id": _get_url_param(query_dict, "task", "").strip(),
        "workflow_choice": workflow_choice,
        "score_weights": score_weights,
        "ppm_scale": ppm_scale,
        "ppm_floor": ppm_floor,
        "detail_mode": _get_url_param(query_dict, "detail", "0") == "1",
        "scan": _get_url_param(query_dict, "scan", "").strip(),
        "precomputed_query_entropy": precomputed_query_entropy,
    }


NAVBAR = dbc.Navbar(
    children=[
        dbc.NavbarBrand(
            "GNPS2-AnnEx: Annotation Evidence Explorer",
            href="/",
        ),
    ],
    color="light",
    dark=False,
    sticky="top",
)


CONTROLS_CARD = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                "Task ID",
                                html_for="task-id",
                                className="small fw-semibold text-uppercase text-muted mb-1",
                            ),
                            dbc.Input(
                                id="task-id",
                                placeholder="Enter FBMN, Classical Networking, or Library Search task ID",
                                value="",
                                style={
                                    "borderRadius": "10px",
                                    "borderColor": "rgba(0,0,0,0.12)",
                                },
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "Workflow",
                                html_for="workflow-choice",
                                className="small fw-semibold text-uppercase text-muted mb-1",
                            ),
                            dbc.Select(
                                id="workflow-choice",
                                options=[
                                    {"label": "Auto-detect", "value": "auto"},
                                    {
                                        "label": "Classical Networking",
                                        "value": "classic",
                                    },
                                    {"label": "FBMN", "value": "fbmn"},
                                    {
                                        "label": "Library Search",
                                        "value": "librarysearch",
                                    },
                                ],
                                value="auto",
                                style={
                                    "borderRadius": "10px",
                                    "borderColor": "rgba(0,0,0,0.12)",
                                },
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3",
            ),
            html.Div(
                _build_score_weight_controls(),
            ),
            html.Div(
                [
                    dbc.Button(
                        "Load annotations",
                        id="run-analysis",
                        color="primary",
                        n_clicks=0,
                        style={
                            "borderRadius": "999px",
                            "paddingLeft": "1.5rem",
                            "paddingRight": "1.5rem",
                        },
                    ),
                    dbc.Button(
                        "How does scoring work?",
                        id="open-info-modal",
                        color="link",
                        size="sm",
                        className="text-muted ms-2",
                        style={"textDecoration": "none"},
                    ),
                ],
                className="mt-3",
            ),
        ],
        className="p-4",
    ),
    className="border-0 shadow-sm",
    style={"borderRadius": "18px", "background": "rgba(255,255,255,0.9)"},
)


INFO_MODAL = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("How scoring works"), close_button=True),
        dbc.ModalBody(
            [
                html.P(
                    "The evidence score (0–100) is computed per scan from five weighted signals, then multiplied by an m/z precision factor.",
                    className="text-muted",
                ),
                html.H6(
                    "Score components (default weights)",
                    className="mt-3 mb-2 small fw-semibold text-uppercase text-muted",
                ),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem(
                            [
                                html.Strong("~45% "),
                                "Structure agreement",
                                html.Br(),
                                html.Small(
                                    "Supporting hits ÷ total hits. Hits are grouped by planar InChIKey (stereo ignored) and positional-isomer family (Murcko scaffold + molecular formula). Higher fractions mean multiple independent library entries agree on the same annotation.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Strong("~14% "),
                                "Tanimoto structural diversity (WCRS)",
                                html.Br(),
                                html.Small(
                                    "WCRS (Weighted Centroid Representativeness Score) picks the most representative annotation group using ECFP4 fingerprints (Morgan r=2, 2048-bit). The reported score is d̄ᵢ — the weighted average pairwise Tanimoto similarity from the winner to all other evidence groups. Falls back to structure agreement when fewer than 2 unique valid SMILES are available.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Strong("~23% "),
                                "Spectral reproducibility (cosine consistency)",
                                html.Br(),
                                html.Small(
                                    "1 − clamp(std/mean, 0, 1) of MQScores (cosine similarities) within the winning support group. Measures how consistently the spectrum matches independent library entries — a score of 1.0 means all cosines are identical, 0.0 means maximal spread. Single-hit groups default to 1.0.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Strong("~9% "),
                                "Best cosine score",
                                html.Br(),
                                html.Small(
                                    "The maximum MQScore (cosine similarity) observed across all library hits for this scan. Reflects the quality of the best individual spectral match.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Strong("~9% "),
                                "Shared peaks",
                                html.Br(),
                                html.Small(
                                    "Soft-normalised mean fragment-ion overlap: x / (x + 5). Counts fragment ions matched between the query spectrum and library hits in the winning group. Penalises scans with very few shared peaks, even when cosine is high.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                    ],
                    flush=True,
                    className="mb-3",
                ),
                html.H6(
                    "m/z precision multiplier",
                    className="mt-3 mb-2 small fw-semibold text-uppercase text-muted",
                ),
                html.P(
                    [
                        "The weighted sum is multiplied by ",
                        html.Code("exp(−|ppm| / 10)"),
                        " (floored at 0.5). The PPM value used is the ",
                        html.Strong("median"),
                        " absolute precursor m/z error across supporting hits — robust to single miscalibrated library entries. A scan with ≤1 ppm error loses less than 10% of its score; one at 10 ppm loses ~37%.",
                    ],
                    className="small text-muted",
                ),
                dbc.Alert(
                    "Percentages are approximate (nominal ratio 50:15:25:10:10, normalised to sum to 1).",
                    color="light",
                    className="mb-2 py-2 small border",
                ),
                (
                    dbc.Alert(
                        "Dev note: the local dashboard can override these defaults with the score-weight tuning controls.",
                        color="warning",
                        className="mb-2 py-2 small",
                    )
                    if ENABLE_DEV_SCORE_WEIGHT_CONTROLS
                    else None
                ),
                dbc.Alert(
                    "Stereochemistry is ignored — planar InChIKey connectivity (or fallbacks) is used for structure grouping.",
                    color="info",
                    className="mb-2 py-2 small",
                ),
                dbc.Alert(
                    "Auto mode tries the GNPS2 Library Search librarymatches JSON view in addition to classic and FBMN loaders.",
                    color="secondary",
                    className="mb-0 py-2 small",
                ),
            ]
        ),
        dbc.ModalFooter(
            dbc.Button(
                "Close",
                id="close-info-modal",
                color="secondary",
                outline=True,
                size="sm",
            )
        ),
    ],
    id="info-modal",
    is_open=False,
    centered=True,
    size="lg",
)


BODY = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="matches-store", storage_type="memory"),
        dcc.Store(id="run-params-store", storage_type="memory"),
        dcc.Store(id="scan-data-store", storage_type="memory"),
        dcc.Store(id="pubchem-overrides-store", storage_type="memory"),
        dcc.Download(id="download-summary"),
        INFO_MODAL,
        dbc.Row(
            [
                dbc.Col(html.Div(CONTROLS_CARD, id="controls-container"), md=12),
            ],
            style={"marginTop": 24},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        id="loading-detail-panel",
                        custom_spinner=_net_spinner("loading detail view"),
                        children=html.Div(
                            [
                                html.Div(id="detail-score-panel"),
                                html.Div(id="detail-compounds-panel"),
                                html.Div(id="detail-masst-panel"),
                            ],
                            id="detail-panel",
                        ),
                    ),
                    md=12,
                ),
            ],
            style={"marginTop": 12},
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Checklist(
                                        id="pubchem-enabled",
                                        options=[
                                            {
                                                "label": "Use selected PubChem SMILES in heatmap and score",
                                                "value": "enabled",
                                            }
                                        ],
                                        value=[],
                                        switch=True,
                                    ),
                                    html.Small(
                                        "Enable to apply manually selected PubChem SMILES to the Tanimoto heatmap "
                                        "and the detail confidence score.",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                            className="border-0 shadow-sm",
                            style={
                                "borderRadius": "14px",
                                "background": "rgba(255,255,255,0.9)",
                            },
                        ),
                        id="pubchem-controls",
                        style={"display": "none"},
                    ),
                    md=12,
                ),
            ],
            style={"marginTop": 8},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        id="loading-heatmap",
                        custom_spinner=_net_spinner("building heatmap"),
                        children=html.Div(id="heatmap-panel"),
                    ),
                    md=12,
                ),
            ],
            style={"marginTop": 4},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        id="loading-status",
                        custom_spinner=_net_spinner("fetching matches"),
                        children=html.Div(id="load-status"),
                    ),
                    md=12,
                ),
            ],
            style={"marginTop": 12},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        id="loading-output",
                        custom_spinner=_net_spinner("scoring annotations"),
                        children=html.Div(id="results-panel"),
                    ),
                    md=12,
                ),
            ],
            style={"marginTop": 4},
        ),
    ],
    fluid=True,
)


dash_app.layout = html.Div([NAVBAR, BODY])


@dash_app.callback(
    [
        Output("task-id", "value"),
        Output("workflow-choice", "value"),
        Output("score-weight-structure", "value"),
        Output("score-weight-tanimoto", "value"),
        Output("score-weight-spectral", "value"),
        Output("score-weight-max-cosine", "value"),
        Output("score-weight-shared-peaks", "value"),
        Output("ppm-scale", "value"),
        Output("ppm-floor", "value"),
    ],
    [Input("url", "search")],
)
def determine_task_from_url(search):
    query_state = _parse_dashboard_query(search)
    score_weight_inputs = _score_weight_inputs_from_weights(
        query_state["score_weights"]
    )
    return (
        query_state["task_id"],
        query_state["workflow_choice"],
        score_weight_inputs["structure_agreement"],
        score_weight_inputs["tanimoto"],
        score_weight_inputs["cosine_consistency"],
        score_weight_inputs["max_cosine"],
        score_weight_inputs["shared_peaks"],
        query_state["ppm_scale"],
        query_state["ppm_floor"],
    )


@dash_app.callback(
    Output("controls-container", "style"),
    [Input("url", "search")],
)
def toggle_controls_for_detail_view(search):
    if _parse_dashboard_query(search)["detail_mode"]:
        return {"display": "none"}
    return {"display": "block"}


@dash_app.callback(
    [
        Output("detail-score-panel", "children"),
        Output("detail-compounds-panel", "children"),
        Output("detail-masst-panel", "children"),
        Output("scan-data-store", "data"),
    ],
    [Input("url", "search")],
)
def render_detail_panel(search):
    query_state = _parse_dashboard_query(search)

    if not query_state["detail_mode"]:
        return html.Div(), html.Div(), html.Div(), None

    task_id = query_state["task_id"]
    scan = query_state["scan"]
    workflow_choice = query_state["workflow_choice"]
    precomputed_query_entropy = query_state["precomputed_query_entropy"]
    score_weights = query_state["score_weights"]
    ppm_scale = query_state["ppm_scale"]
    ppm_floor = query_state["ppm_floor"]

    if not task_id or not scan:
        return (
            dbc.Alert(
                "Detail view requires task and scan query parameters.", color="warning"
            ),
            html.Div(),
            html.Div(),
            None,
        )

    try:
        panel_ui, compounds_data = _build_detail_panel(
            task_id,
            scan,
            workflow_choice,
            score_weights=score_weights,
            precomputed_query_entropy=precomputed_query_entropy,
            ppm_scale=ppm_scale,
            mz_precision_floor=ppm_floor,
        )
        children = panel_ui.children if getattr(panel_ui, "children", None) else []
        score_panel = children[0] if len(children) > 0 else html.Div()
        compounds_panel = html.Div(children[1:3]) if len(children) > 1 else html.Div()
        masst_panel = children[3] if len(children) > 3 else html.Div()
        return (
            score_panel,
            compounds_panel,
            masst_panel,
            {
                "compounds": compounds_data,
                "task_id": task_id,
                "scan": scan,
                "workflow_choice": workflow_choice,
                "score_weights": score_weights,
                "ppm_scale": ppm_scale,
                "ppm_floor": ppm_floor,
                "precomputed_query_entropy": precomputed_query_entropy,
            },
        )
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger"), html.Div(), html.Div(), None


@dash_app.callback(
    [
        Output("detail-score-panel", "children", allow_duplicate=True),
        Output("detail-masst-panel", "children", allow_duplicate=True),
    ],
    [
        Input("scan-data-store", "data"),
        Input("pubchem-overrides-store", "data"),
        Input("pubchem-enabled", "value"),
    ],
    prevent_initial_call=True,
)
def refresh_detail_score_panel(scan_data, pubchem_overrides, pubchem_value):
    """Update score panel and MASST card on PubChem selection without remounting the structure table."""
    if not scan_data:
        return html.Div(), html.Div()

    task_id = _truncate_text(str(scan_data.get("task_id", "")), max_length=200)
    scan = _truncate_text(str(scan_data.get("scan", "")), max_length=200)
    workflow_choice = _truncate_text(
        str(scan_data.get("workflow_choice", "auto")), max_length=40
    )
    score_weights = _normalize_score_weights(scan_data.get("score_weights"))
    ppm_scale, ppm_floor = _parse_ppm_from_inputs(
        scan_data.get("ppm_scale"), scan_data.get("ppm_floor")
    )
    use_pubchem_for_score = (
        isinstance(pubchem_value, list) and "enabled" in pubchem_value
    )
    precomputed_query_entropy = scan_data.get("precomputed_query_entropy")

    if not task_id or not scan:
        return html.Div(), html.Div()

    try:
        panel_ui, _ = _build_detail_panel(
            task_id,
            scan,
            workflow_choice,
            score_weights=score_weights,
            pubchem_overrides=pubchem_overrides,
            use_pubchem_for_score=use_pubchem_for_score,
            precomputed_query_entropy=precomputed_query_entropy,
            ppm_scale=ppm_scale,
            mz_precision_floor=ppm_floor,
        )
        children = panel_ui.children if getattr(panel_ui, "children", None) else []
        score_panel = children[0] if len(children) > 0 else panel_ui
        masst_panel = children[3] if len(children) > 3 else html.Div()
        return score_panel, masst_panel
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger"), html.Div()


@dash_app.callback(
    Output("pubchem-overrides-store", "data"),
    Input("url", "search"),
)
def reset_pubchem_overrides_on_page_change(search):
    if _parse_dashboard_query(search)["detail_mode"]:
        return {}
    return None


@dash_app.callback(
    [
        Output("matches-store", "data"),
        Output("run-params-store", "data"),
        Output("load-status", "children"),
    ],
    [Input("run-analysis", "n_clicks"), Input("url", "search")],
    [
        State("task-id", "value"),
        State("workflow-choice", "value"),
        State("score-weight-structure", "value"),
        State("score-weight-tanimoto", "value"),
        State("score-weight-spectral", "value"),
        State("score-weight-max-cosine", "value"),
        State("score-weight-shared-peaks", "value"),
        State("ppm-scale", "value"),
        State("ppm-floor", "value"),
    ],
)
def load_library_matches_callback(
    _n_clicks,
    search,
    task_id,
    workflow_choice,
    structure_weight_value,
    tanimoto_weight_value,
    spectral_weight_value,
    max_cosine_weight_value,
    shared_peaks_weight_value,
    ppm_scale_value,
    ppm_floor_value,
):
    """Stage 1 — fetch library matches and report what was found."""
    query_state = _parse_dashboard_query(search)
    should_autorun = (
        bool(query_state["task_id"])
        and not query_state["detail_mode"]
        and (_n_clicks or 0) < 1
    )

    if should_autorun:
        task_id = query_state["task_id"]
        workflow_choice = query_state["workflow_choice"]
        score_weights = query_state["score_weights"]
        ppm_scale = query_state["ppm_scale"]
        ppm_floor = query_state["ppm_floor"]
    else:
        task_id = (task_id or "").strip()
        workflow_choice = (workflow_choice or "auto").strip().lower()
        score_weights = _parse_score_weights_from_inputs(
            structure_weight_value,
            tanimoto_weight_value,
            spectral_weight_value,
            max_cosine_weight_value,
            shared_peaks_weight_value,
        )
        ppm_scale, ppm_floor = _parse_ppm_from_inputs(ppm_scale_value, ppm_floor_value)

    if workflow_choice not in SUPPORTED_WORKFLOWS:
        workflow_choice = "auto"

    if query_state["detail_mode"] and (_n_clicks or 0) < 1:
        return no_update, no_update, no_update

    if not task_id or ((_n_clicks is None or _n_clicks < 1) and not should_autorun):
        return no_update, no_update, html.Div()

    try:
        matches_df = _load_library_matches(task_id, workflow_choice)
        resolved_workflow = _infer_resolved_workflow(matches_df, workflow_choice)
    except Exception as exc:
        return no_update, no_update, dbc.Alert(str(exc), color="danger")

    scan_col = _find_first_column(matches_df, ["#Scan#", "scan", "Scan"])
    n_matches = len(matches_df)
    n_scans = matches_df[scan_col].nunique() if scan_col else "?"

    status = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        dbc.Badge("Loading", color="primary", className="me-2"),
                        html.Span(
                            "Preparing scan-level confidence summary",
                            className="fw-semibold",
                        ),
                    ],
                    className="d-flex align-items-center mb-2",
                ),
                html.Div(
                    [
                        dbc.Badge(
                            f"{n_matches:,} matches",
                            color="light",
                            text_color="dark",
                            className="me-2",
                        ),
                        dbc.Badge(
                            f"{n_scans:,} scans", color="light", text_color="dark"
                        ),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    html.Span(
                        "Scoring model: structure agreement, WCRS Tanimoto, cosine consistency, shared peaks. Results will appear below.",
                        className="text-muted",
                    ),
                ),
            ]
        ),
        className="border-0 shadow-sm mb-0",
        style={
            "borderRadius": "14px",
            "background": "linear-gradient(135deg, #f7fbff 0%, #eef4ff 100%)",
            "border": "1px solid rgba(90, 130, 190, 0.2)",
        },
    )

    task_info = {}
    try:
        task_info = _load_task_information(task_id)
    except Exception as exc:
        logger.warning("Could not retrieve task information for %s: %s", task_id, exc)

    params = {
        "task_id": task_id,
        "workflow_choice": workflow_choice,
        "resolved_workflow": resolved_workflow,
        "score_weights": score_weights,
        "ppm_scale": ppm_scale,
        "ppm_floor": ppm_floor,
        "task_info": task_info,
    }

    return matches_df.to_json(orient="split"), params, status


@dash_app.callback(
    Output("results-panel", "children"),
    [Input("matches-store", "data"), Input("run-params-store", "data")],
    prevent_initial_call=True,
)
def compute_and_render_scores(matches_json, params):
    """Stage 2 — score all scans and render the results table."""
    if matches_json is None or params is None:
        return html.Div()

    task_id = params["task_id"]
    workflow_choice = params["workflow_choice"]
    resolved_workflow = params["resolved_workflow"]
    score_weights = _normalize_score_weights(params.get("score_weights"))
    ppm_scale, ppm_floor = _parse_ppm_from_inputs(
        params.get("ppm_scale"), params.get("ppm_floor")
    )
    task_info = params.get("task_info") or {}
    custom_score_weights_active = not _score_weights_are_default(score_weights)
    score_weight_query_suffix = _serialize_score_weights_for_query(score_weights)
    ppm_query_suffix = _serialize_ppm_for_query(ppm_scale, ppm_floor)

    try:
        score_weights_tuple = tuple(sorted(score_weights.items()))
        summary_json = _compute_scores_cached(
            task_id,
            workflow_choice,
            resolved_workflow,
            score_weights_tuple,
            ppm_scale=ppm_scale,
            mz_precision_floor=ppm_floor,
        )
        if summary_json is None:
            return dbc.Alert(
                "No library matches were found for this task.", color="warning"
            )
        summary_df = pd.read_json(io.StringIO(summary_json), orient="split")
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger")

    if len(summary_df) == 0:
        return dbc.Alert(
            "No library matches were found for this task.", color="warning"
        )

    badge_counts = {
        "Consistent evidence": int(
            (summary_df["confidence_label"] == "Consistent evidence").sum()
        ),
        "Inconclusive": int((summary_df["confidence_label"] == "Inconclusive").sum()),
        "Inconsistent evidence": int(
            (summary_df["confidence_label"] == "Inconsistent evidence").sum()
        ),
    }
    task_metadata_details = _build_task_metadata_details(task_info, resolved_workflow)

    table_df = summary_df.copy()
    table_df["adjusted_support_score"] = (
        (table_df["supporting_matches"].astype(float) + LAPLACE_ALPHA)
        / (table_df["total_matches"].astype(float) + LAPLACE_BETA)
    ).round(3)
    table_df["support"] = (
        table_df["supporting_matches"].astype(str)
        + " / "
        + table_df["total_matches"].astype(str)
    )

    _sk_counts = table_df["structure_key"].value_counts()
    _duplicate_structure_keys = set(_sk_counts[_sk_counts > 1].index)

    table_df["compound_link"] = table_df.apply(
        lambda row: (
            ("🔁 " if row.get("structure_key") in _duplicate_structure_keys else "")
            + f"[{row['top_compound']}](/dashinterface2/?task={urllib.parse.quote(str(task_id))}"
            f"&workflow={urllib.parse.quote(str(workflow_choice))}"
            f"{score_weight_query_suffix}"
            f"{ppm_query_suffix}"
            f"&scan={urllib.parse.quote(str(row['scan']))}"
            + (
                f"&qe={row['query_entropy']}"
                if row.get("query_entropy") is not None
                else ""
            )
            + "&detail=1)"
        ),
        axis=1,
    )
    table_df = table_df[[col["field"] for col in _SUMMARY_COLUMN_DEFS]]

    initial_visible = {
        col["field"] for col in _SUMMARY_COLUMN_DEFS
    } - _DEFAULT_HIDDEN_FIELDS
    initial_col_defs = [
        {**col, "hide": col["field"] not in initial_visible}
        for col in _SUMMARY_COLUMN_DEFS
    ]

    results_table = dag.AgGrid(
        id="confidence-datatable",
        rowData=table_df.to_dict("records"),
        columnDefs=initial_col_defs,
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
            "wrapText": True,
            "autoHeight": True,
            "wrapHeaderText": True,
            "autoHeaderHeight": True,
            "floatingFilter": True,
            "cellStyle": {"fontSize": "13px", "lineHeight": "1.35"},
        },
        dashGridOptions={
            "tooltipShowDelay": 10,
            "pagination": True,
            "paginationPageSize": 25,
            "animateRows": False,
            "rowHeight": 40,
            "theme": "themeBalham",
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
        },
        className="ag-theme-alpine",
        style={"height": "760px", "width": "100%"},
    )

    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Annotation Evidence Summary")),
                    dbc.CardBody(
                        [
                            html.P(
                                [
                                    "Task: ",
                                    html.A(
                                        task_id,
                                        href=f"https://gnps2.org/status?task={task_id}",
                                        target="_blank",
                                    ),
                                    " · Scoring: structure agreement, WCRS Tanimoto, cosine consistency, shared peaks",
                                ]
                            ),
                            html.Small(
                                _format_score_weight_summary(score_weights),
                                className="text-muted d-block mb-2",
                            ),
                            task_metadata_details,
                            (
                                dbc.Badge(
                                    "Custom score weights active",
                                    color="warning",
                                    text_color="dark",
                                    className="me-2",
                                )
                                if custom_score_weights_active
                                else None
                            ),
                            dbc.Badge(
                                f"Consistent evidence: {badge_counts['Consistent evidence']}",
                                color="success",
                                className="me-2",
                            ),
                            dbc.Badge(
                                f"Inconclusive: {badge_counts['Inconclusive']}",
                                color="warning",
                                className="me-2",
                            ),
                            dbc.Badge(
                                f"Inconsistent evidence: {badge_counts['Inconsistent evidence']}",
                                color="danger",
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Scan-Level Annotations")),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Small(
                                            "Click a proposed compound to open the detail view with structure images and mirror-plot links.",
                                            className="text-muted",
                                        ),
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "⬇ Export CSV",
                                                    id="download-csv-btn",
                                                    color="outline-secondary",
                                                    size="sm",
                                                    className="me-2",
                                                ),
                                                dbc.DropdownMenu(
                                                    label="Columns",
                                                    children=dbc.Checklist(
                                                        id="column-visibility-checklist",
                                                        options=[
                                                            {
                                                                "label": col[
                                                                    "headerName"
                                                                ]
                                                                .split("(")[0]
                                                                .strip(),
                                                                "value": col["field"],
                                                            }
                                                            for col in _SUMMARY_COLUMN_DEFS
                                                        ],
                                                        value=list(initial_visible),
                                                        className="px-3 py-2",
                                                        style={"minWidth": "220px"},
                                                    ),
                                                    size="sm",
                                                    align_end=True,
                                                ),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                        width="auto",
                                    ),
                                ],
                                align="center",
                                className="mb-2",
                            ),
                            html.Hr(className="mt-0"),
                            results_table,
                        ]
                    ),
                ]
            ),
        ]
    )


@dash_app.callback(
    Output("confidence-datatable", "columnDefs"),
    Input("column-visibility-checklist", "value"),
    prevent_initial_call=True,
)
def update_column_visibility(visible_fields):
    """Show/hide columns in the summary table based on the toggle checklist."""
    visible = set(visible_fields or [])
    return [
        {**col, "hide": col["field"] not in visible} for col in _SUMMARY_COLUMN_DEFS
    ]


@dash_app.callback(
    Output("download-summary", "data"),
    Input("download-csv-btn", "n_clicks"),
    [State("matches-store", "data"), State("run-params-store", "data")],
    prevent_initial_call=True,
)
def download_summary_csv(n_clicks, matches_json, params):
    """Export the full annotation summary (all columns, clean values) as a CSV."""
    if not n_clicks or matches_json is None or params is None:
        return no_update

    task_id = params.get("task_id", "unknown")
    resolved_workflow = params.get("resolved_workflow", "auto")
    score_weights = _normalize_score_weights(params.get("score_weights"))
    ppm_scale, ppm_floor = _parse_ppm_from_inputs(
        params.get("ppm_scale"), params.get("ppm_floor")
    )

    try:
        workflow_choice = params.get("workflow_choice", "auto")
        score_weights_tuple = tuple(sorted(score_weights.items()))
        summary_json = _compute_scores_cached(
            task_id,
            workflow_choice,
            resolved_workflow,
            score_weights_tuple,
            ppm_scale=ppm_scale,
            mz_precision_floor=ppm_floor,
        )
        if summary_json is None:
            return no_update
        summary_df = pd.read_json(io.StringIO(summary_json), orient="split")
    except Exception:
        return no_update

    export_df = summary_df.copy()
    export_df["adjusted_support_score"] = (
        (export_df["supporting_matches"].astype(float) + LAPLACE_ALPHA)
        / (export_df["total_matches"].astype(float) + LAPLACE_BETA)
    ).round(3)
    export_df["task_id"] = task_id
    for key, value in score_weights.items():
        export_df[f"weight_{key}"] = round(float(value), 6)

    filename = f"annex_summary_{task_id[:12]}.csv"
    return dcc.send_data_frame(export_df.to_csv, filename, index=False)


@dash_app.callback(
    Output("spectrum-collapse", "is_open"),
    Input("spectrum-collapse-btn", "n_clicks"),
    State("spectrum-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_spectrum_collapse(n_clicks, is_open):
    """Toggle the query spectrum panel open/closed."""
    if n_clicks:
        return not is_open
    return is_open


@dash_app.callback(
    Output({"type": "pubchem-candidates-div", "index": MATCH}, "children"),
    Input({"type": "pubchem-search-btn", "index": MATCH}, "n_clicks"),
    State({"type": "pubchem-name-input", "index": MATCH}, "value"),
    State({"type": "compound-info-store", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def search_pubchem_for_card(n_clicks, manual_name_query, compound_info):
    """Fetch PubChem candidates for one structure row and render a dropdown picker."""
    if not n_clicks or not compound_info:
        return no_update

    triggered_index = ctx.triggered_id["index"]
    smiles = (compound_info.get("smiles") or "").strip()
    typed_query = (manual_name_query or "").strip()

    if smiles:
        return dbc.Alert(
            "PubChem lookup is only available when SMILES is missing.",
            color="info",
            className="py-2 small mb-0",
        )

    if not typed_query:
        return dbc.Alert(
            "Type a compound name before searching PubChem.",
            color="info",
            className="py-2 small mb-0",
        )

    candidates = _search_pubchem_candidates("", typed_query)

    if not candidates:
        return dbc.Alert(
            f'No PubChem match found for "{_truncate_text(typed_query, max_length=40)}".',
            color="warning",
            className="py-2 small mb-0",
        )

    dropdown_options = []
    for c in candidates:
        candidate_smiles = c.get("smiles") or ""
        if not candidate_smiles:
            continue

        matched_name = _truncate_text(c.get("matched_name", ""), max_length=30)
        source_label = c.get("source", "name")
        query_note = (
            f" | query: {matched_name}"
            if source_label == "name-fuzzy" and matched_name
            else ""
        )
        dropdown_options.append(
            {
                "label": (
                    f"CID {c.get('cid', '?')} ({source_label}) - "
                    f"{_truncate_text(c.get('iupac_name', ''), max_length=45)}{query_note}"
                ),
                "value": candidate_smiles,
            }
        )

    if not dropdown_options:
        return dbc.Alert(
            "PubChem returned results but none had a SMILES string.",
            color="warning",
            className="py-2 small mb-0",
        )

    return html.Div(
        [
            html.Small(
                f"{len(dropdown_options)} candidate(s) found - select one to apply it:",
                className="text-muted d-block mb-2",
            ),
            dcc.Dropdown(
                id={"type": "pubchem-dropdown", "index": triggered_index},
                options=dropdown_options,
                value=None,
                placeholder="Select the best PubChem match",
                clearable=True,
            ),
        ]
    )


@dash_app.callback(
    Output("pubchem-overrides-store", "data", allow_duplicate=True),
    Input({"type": "pubchem-dropdown", "index": ALL}, "value"),
    [
        State({"type": "pubchem-dropdown", "index": ALL}, "id"),
        State("pubchem-overrides-store", "data"),
    ],
    prevent_initial_call=True,
)
def apply_pubchem_override(dropdown_values, dropdown_ids, current_overrides):
    """Write the user-selected PubChem SMILES into the overrides store."""
    if not isinstance(ctx.triggered_id, dict):
        return no_update

    triggered_index = ctx.triggered_id["index"]

    selected_smiles = None
    found_triggered = False
    for rid, rval in zip(dropdown_ids or [], dropdown_values or []):
        if isinstance(rid, dict) and rid.get("index") == triggered_index:
            found_triggered = True
            selected_smiles = (rval or "").strip() if isinstance(rval, str) else ""
            break

    if not found_triggered:
        return no_update

    overrides = dict(current_overrides or {})
    triggered_key = str(triggered_index)

    # Ignore initial empty dropdown value when candidates are first rendered.
    if not selected_smiles and triggered_key not in overrides:
        return no_update

    if selected_smiles and overrides.get(triggered_key) == selected_smiles:
        return no_update

    if selected_smiles:
        overrides[triggered_key] = selected_smiles
    else:
        overrides.pop(triggered_key, None)
    return overrides


@dash_app.callback(
    Output({"type": "pubchem-structure-preview", "index": ALL}, "children"),
    Input("pubchem-overrides-store", "data"),
    [
        State({"type": "pubchem-structure-preview", "index": ALL}, "id"),
        State({"type": "compound-info-store", "index": ALL}, "data"),
    ],
    prevent_initial_call=True,
)
def refresh_pubchem_structure_previews(overrides, preview_ids, compound_infos):
    """Update structure previews on missing-structure rows after PubChem selection."""
    preview_ids = preview_ids or []
    compound_infos = compound_infos or []

    if not preview_ids:
        return []

    overrides = overrides or {}
    updated_children = []

    for preview_id, compound_info in zip(preview_ids, compound_infos):
        card_idx = str((preview_id or {}).get("index", ""))
        info = compound_info or {}
        base_smiles = (info.get("smiles") or "").strip()
        inchi = (info.get("inchi") or "").strip()
        selected_smiles = (overrides.get(card_idx) or "").strip()
        smiles_for_render = selected_smiles or base_smiles

        image_url = _render_mol_png_data_uri(smiles=smiles_for_render, inchi=inchi)
        if image_url:
            updated_children.append(
                html.Div(
                    [
                        html.Img(
                            src=image_url,
                            style={
                                "width": "180px",
                                "height": "180px",
                                "objectFit": "contain",
                            },
                        ),
                        html.Small(
                            (
                                "Structure from selected PubChem match"
                                if selected_smiles
                                else "Structure from library/InChI"
                            ),
                            className="text-muted d-block",
                        ),
                    ]
                )
            )
        else:
            updated_children.append(
                html.Small(
                    "No structure image yet. Select a PubChem match to display one.",
                    className="text-muted d-block",
                )
            )

    return updated_children


@dash_app.callback(
    Output("pubchem-controls", "style"),
    [Input("scan-data-store", "data")],
)
def toggle_pubchem_controls(scan_data):
    if scan_data:
        return {"display": "block"}
    return {"display": "none"}


@dash_app.callback(
    Output("heatmap-panel", "children"),
    [
        Input("scan-data-store", "data"),
        Input("pubchem-overrides-store", "data"),
        Input("pubchem-enabled", "value"),
    ],
    prevent_initial_call=True,
)
def render_tanimoto_heatmap(scan_data, overrides, pubchem_value):
    """Render the pairwise Tanimoto heatmap, applying any user-selected PubChem overrides."""
    if not scan_data:
        return html.Div()

    use_selected_pubchem = (
        isinstance(pubchem_value, list) and "enabled" in pubchem_value
    )
    overrides = overrides or {}

    # Apply per-card overrides only when the toggle is enabled.
    compounds_data = []
    applied_overrides = 0
    for c in scan_data.get("compounds", []):
        c_copy = dict(c)
        override_smiles = ""
        if use_selected_pubchem:
            override_smiles = (overrides.get(str(c.get("card_idx", ""))) or "").strip()
        if override_smiles:
            c_copy["smiles"] = override_smiles
            applied_overrides += 1
        compounds_data.append(c_copy)

    if len(compounds_data) < 2:
        return dbc.Alert(
            "At least 2 unique compounds are needed to display a Tanimoto heatmap.",
            color="info",
            className="mb-0 py-2",
        )

    fig = _build_tanimoto_heatmap_figure(compounds_data, use_pubchem=False)

    if fig is None:
        return dbc.Alert(
            "RDKit/Plotly unavailable — cannot build Tanimoto heatmap.", color="warning"
        )

    n_missing = sum(1 for c in compounds_data if not c.get("smiles"))

    if applied_overrides:
        source_note = (
            f"{applied_overrides} structure(s) updated via selected PubChem SMILES."
        )
    elif overrides and not use_selected_pubchem:
        source_note = (
            "PubChem selections are available but currently disabled by the toggle."
        )
    elif n_missing:
        source_note = (
            f"{n_missing} compound(s) have no SMILES — cells are blank. "
            "Type a name and use 'Search PubChem' in the table above, then enable the toggle below."
        )
    else:
        source_note = "All SMILES sourced from library."

    heatmap_max_width = max(420, min(960, len(compounds_data) * 38 + 240))

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Pairwise Tanimoto Similarity Heatmap")),
            dbc.CardBody(
                [
                    html.Small(
                        [
                            f"ECFP4 pairwise Tanimoto across {len(compounds_data)} unique compound(s). "
                            "Green = structurally similar, red = dissimilar.",
                            html.Span(source_note, className="text-muted"),
                        ]
                    ),
                    html.Hr(),
                    html.Div(
                        dcc.Graph(
                            figure=fig,
                            config={"displayModeBar": False},
                            style={"width": "100%"},
                        ),
                        style={
                            "width": "100%",
                            "maxWidth": f"{heatmap_max_width}px",
                            "margin": "0 auto",
                        },
                    ),
                ]
            ),
        ],
        className="mb-3",
        style={
            "width": "100%",
            "maxWidth": f"{heatmap_max_width}px",
            "margin": "0 auto",
        },
    )


@dash_app.callback(
    Output("info-modal", "is_open"),
    [Input("open-info-modal", "n_clicks"), Input("close-info-modal", "n_clicks")],
    [State("info-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_info_modal(_open, _close, is_open):
    return not is_open


if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", port=5000, debug=True)
