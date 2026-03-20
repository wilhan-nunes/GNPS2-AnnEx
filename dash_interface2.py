# -*- coding: utf-8 -*-
import logging
import urllib.parse

import dash
from dash import Input, Output, State, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from app import app
from components import _build_detail_panel, _build_intro_panel
from constants import SUPPORTED_WORKFLOWS
import io

import pandas as pd

from loader import _load_library_matches
from normalizer import _find_first_column, _infer_resolved_workflow
from scorer import _compute_scan_level_confidence


dash_app = dash.Dash(
    name="dashinterface2",
    server=app,
    url_base_pathname="/dashinterface2/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

dash_app.title = "GNPS2-AnnEx"

logger = logging.getLogger(__name__)


def _get_url_param(param_dict, key, default=""):
    return param_dict.get(key, [default])[0]


def _parse_dashboard_query(search):
    try:
        query_dict = urllib.parse.parse_qs((search or "").lstrip("?"))
    except Exception:
        query_dict = {}

    workflow_choice = _get_url_param(query_dict, "workflow", "auto").strip().lower() or "auto"
    if workflow_choice not in SUPPORTED_WORKFLOWS:
        workflow_choice = "auto"

    entropy_param = _get_url_param(query_dict, "entropy", "0").strip().lower()
    use_entropy_similarity = entropy_param not in {"0", "false", "off", "no"}

    return {
        "task_id": _get_url_param(query_dict, "task", "").strip(),
        "workflow_choice": workflow_choice,
        "use_entropy_similarity": use_entropy_similarity,
        "use_entropy_value": ["enabled"] if use_entropy_similarity else [],
        "detail_mode": _get_url_param(query_dict, "detail", "0") == "1",
        "scan": _get_url_param(query_dict, "scan", "").strip(),
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
                            dbc.Label("Task ID", html_for="task-id", className="small fw-semibold text-uppercase text-muted mb-1"),
                            dbc.Input(
                                id="task-id",
                                placeholder="Enter FBMN, Classical Networking, or Library Search task ID",
                                value="",
                                style={"borderRadius": "10px", "borderColor": "rgba(0,0,0,0.12)"},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Workflow", html_for="workflow-choice", className="small fw-semibold text-uppercase text-muted mb-1"),
                            dbc.Select(
                                id="workflow-choice",
                                options=[
                                    {"label": "Auto-detect", "value": "auto"},
                                    {"label": "Classical Networking", "value": "classic"},
                                    {"label": "FBMN", "value": "fbmn"},
                                    {"label": "Library Search", "value": "librarysearch"},
                                ],
                                value="auto",
                                style={"borderRadius": "10px", "borderColor": "rgba(0,0,0,0.12)"},
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("\u00a0", className="d-block small"),
                            dbc.Checklist(
                                id="use-entropy",
                                options=[{"label": "Entropy similarity", "value": "enabled"}],
                                value=[],
                                switch=True,
                                className="mt-1",
                            ),
                        ],
                        md=3,
                        className="d-flex flex-column justify-content-end",
                    ),
                ],
                className="g-3",
            ),
            html.Div(
                [
                    dbc.Button(
                        "Load annotations",
                        id="run-analysis",
                        color="primary",
                        n_clicks=0,
                        style={"borderRadius": "999px", "paddingLeft": "1.5rem", "paddingRight": "1.5rem"},
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
                    "The evidence score (0–100) is computed per scan from four weighted signals.",
                    className="text-muted",
                ),
                html.H6("Score weights", className="mt-3 mb-2 small fw-semibold text-uppercase text-muted"),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem([html.Strong("50% "), "Structure agreement — supporting hits / total hits (same planar structure or positional-isomer family)"]),
                        dbc.ListGroupItem([html.Strong("15% "), "Tanimoto (ECFP4) — mean pairwise structural similarity; fallback to structure agreement when SMILES unavailable"]),
                        dbc.ListGroupItem([html.Strong("25% "), "Spectral similarity — entropy similarity when enabled; fallback to mean cosine otherwise"]),
                        dbc.ListGroupItem([html.Strong("10% "), "Max cosine — best MQScore observed in the scan"]),
                    ],
                    flush=True,
                    className="mb-3",
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
            dbc.Button("Close", id="close-info-modal", color="secondary", outline=True, size="sm")
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
        INFO_MODAL,
        dbc.Row(
            [
                dbc.Col(html.Div(CONTROLS_CARD, id="controls-container"), md=12),
            ],
            style={"marginTop": 24},
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(id="detail-panel"), md=12),
            ],
            style={"marginTop": 12},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        id="loading-status",
                        type="dot",
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
                        type="default",
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
    [Output("task-id", "value"), Output("workflow-choice", "value"), Output("use-entropy", "value")],
    [Input("url", "search")],
)
def determine_task_from_url(search):
    query_state = _parse_dashboard_query(search)
    return (
        query_state["task_id"],
        query_state["workflow_choice"],
        query_state["use_entropy_value"],
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
    Output("detail-panel", "children"),
    [Input("url", "search")],
)
def render_detail_panel(search):
    query_state = _parse_dashboard_query(search)

    if not query_state["detail_mode"]:
        return html.Div()

    task_id = query_state["task_id"]
    scan = query_state["scan"]
    workflow_choice = query_state["workflow_choice"]
    use_entropy_similarity = query_state["use_entropy_similarity"]

    if not task_id or not scan:
        return dbc.Alert("Detail view requires task and scan query parameters.", color="warning")

    try:
        return _build_detail_panel(
            task_id,
            scan,
            workflow_choice,
            use_entropy_similarity=use_entropy_similarity,
        )
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger")


@dash_app.callback(
    [
        Output("matches-store", "data"),
        Output("run-params-store", "data"),
        Output("load-status", "children"),
    ],
    [Input("run-analysis", "n_clicks"), Input("url", "search")],
    [State("task-id", "value"), State("workflow-choice", "value"), State("use-entropy", "value")],
)
def load_library_matches_callback(_n_clicks, search, task_id, workflow_choice, use_entropy_value):
    """Stage 1 — fetch library matches and report what was found."""
    query_state = _parse_dashboard_query(search)
    should_autorun = bool(query_state["task_id"]) and not query_state["detail_mode"] and (_n_clicks or 0) < 1

    if should_autorun:
        task_id = query_state["task_id"]
        workflow_choice = query_state["workflow_choice"]
        use_entropy_similarity = query_state["use_entropy_similarity"]
    else:
        task_id = (task_id or "").strip()
        workflow_choice = (workflow_choice or "auto").strip().lower()
        use_entropy_similarity = isinstance(use_entropy_value, list) and ("enabled" in use_entropy_value)

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

    entropy_note = ""
    if use_entropy_similarity:
        entropy_note = (
            f" Spectral entropy similarity is ON — computing USI comparisons for {n_scans} scan(s). "
            "This may take several minutes for large datasets."
        )

    status = dbc.Alert(
        [
            html.Strong(f"Loaded {n_matches:,} library matches across {n_scans:,} scan(s). "),
            html.Span("Computing confidence scores"),
            html.Span(" (structure agreement, Tanimoto ECFP4, cosine, library class)"),
            html.Span(entropy_note, className="text-muted"),
            html.Span(" — results will appear below.", className="text-muted"),
        ],
        color="info",
        className="mb-0 py-2",
    )

    params = {
        "task_id": task_id,
        "workflow_choice": workflow_choice,
        "resolved_workflow": resolved_workflow,
        "use_entropy_similarity": use_entropy_similarity,
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
    use_entropy_similarity = params["use_entropy_similarity"]

    try:
        matches_df = pd.read_json(io.StringIO(matches_json), orient="split")
        summary_df = _compute_scan_level_confidence(
            matches_df,
            task_id=task_id,
            workflow_choice=resolved_workflow,
            use_entropy_similarity=use_entropy_similarity,
        )
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger")

    if len(summary_df) == 0:
        return dbc.Alert("No library matches were found for this task.", color="warning")

    badge_counts = {
        "Well-supported": int((summary_df["confidence_label"] == "Well-supported").sum()),
        "Ambiguous": int((summary_df["confidence_label"] == "Ambiguous").sum()),
        "Insufficient evidence": int((summary_df["confidence_label"] == "Insufficient evidence").sum()),
    }

    table_df = summary_df.copy()
    table_df["support"] = (
        table_df["supporting_matches"].astype(str) + " / " + table_df["total_matches"].astype(str)
    )
    table_df["compound_link"] = table_df.apply(
        lambda row: (
            f"[{row['top_compound']}](/dashinterface2/?task={urllib.parse.quote(str(task_id))}"
            f"&workflow={urllib.parse.quote(str(workflow_choice))}"
            f"&entropy={'1' if use_entropy_similarity else '0'}"
            f"&scan={urllib.parse.quote(str(row['scan']))}&detail=1)"
        ),
        axis=1,
    )
    table_df = table_df[
        [
            "scan",
            "compound_link",
            "structure_key",
            "support",
            "tanimoto_score",
            "tanimoto_source",
            "entropy_similarity",
            "entropy_source",
            "mean_cosine",
            "max_cosine",
            "confidence_score",
            "confidence_label",
        ]
    ]

    results_table = dash_table.DataTable(
        id="confidence-datatable",
        data=table_df.to_dict("records"),
        columns=[
            {"name": "Scan", "id": "scan"},
            {"name": "Proposed Compound", "id": "compound_link", "presentation": "markdown"},
            {"name": "Representative Structure Key", "id": "structure_key"},
            {"name": "Support (planar structure or isomer family / total)", "id": "support"},
            {"name": "Tanimoto (ECFP4)", "id": "tanimoto_score"},
            {"name": "Tanimoto Source", "id": "tanimoto_source"},
            {"name": "Entropy Similarity", "id": "entropy_similarity"},
            {"name": "Entropy Source", "id": "entropy_source"},
            {"name": "Mean Cosine", "id": "mean_cosine"},
            {"name": "Max Cosine", "id": "max_cosine"},
            {"name": "Score", "id": "confidence_score"},
            {"name": "Evidence Level", "id": "confidence_label"},
        ],
        page_action="native",
        page_current=0,
        page_size=25,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
        style_cell={
            "textAlign": "left",
            "padding": "6px",
            "fontSize": "13px",
            "whiteSpace": "normal",
            "height": "auto",
            "maxWidth": "240px",
        },
        style_cell_conditional=[
            {
                "if": {"column_id": "compound_link"},
                "width": "180px",
                "maxWidth": "180px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            {
                "if": {"column_id": "confidence_score"},
                "width": "90px",
                "maxWidth": "90px",
                "textAlign": "center",
                "fontWeight": "600",
            },
            {
                "if": {"column_id": "confidence_label"},
                "width": "130px",
                "maxWidth": "130px",
                "textAlign": "center",
                "fontWeight": "600",
            },
            {
                "if": {"column_id": "tanimoto_score"},
                "width": "100px",
                "maxWidth": "100px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "tanimoto_source"},
                "width": "155px",
                "maxWidth": "155px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "entropy_similarity"},
                "width": "115px",
                "maxWidth": "115px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "entropy_source"},
                "width": "150px",
                "maxWidth": "150px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "scan"},
                "width": "95px",
                "maxWidth": "95px",
            },
        ],
        style_data_conditional=[
            {
                "if": {
                    "filter_query": '{confidence_label} = "Well-supported"',
                    "column_id": "confidence_label",
                },
                "backgroundColor": "#d1e7dd",
                "color": "#0f5132",
            },
            {
                "if": {
                    "filter_query": '{confidence_label} = "Ambiguous"',
                    "column_id": "confidence_label",
                },
                "backgroundColor": "#fff3cd",
                "color": "#664d03",
            },
            {
                "if": {
                    "filter_query": '{confidence_label} = "Insufficient evidence"',
                    "column_id": "confidence_label",
                },
                "backgroundColor": "#f8d7da",
                "color": "#842029",
            },
        ],
    )

    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Annotation Evidence Summary")),
                    dbc.CardBody(
                        [
                            html.P(f"Task: {task_id} · Entropy scoring: {'enabled' if use_entropy_similarity else 'disabled'}"),
                            dbc.Badge(f"Well-supported: {badge_counts['Well-supported']}", color="success", className="me-2"),
                            dbc.Badge(
                                f"Ambiguous: {badge_counts['Ambiguous']}",
                                color="warning",
                                className="me-2",
                            ),
                            dbc.Badge(f"Insufficient evidence: {badge_counts['Insufficient evidence']}", color="danger"),
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
                            html.Small(
                                "Click a proposed compound to open the detail view with structure images and mirror-plot links.",
                                className="text-muted",
                            ),
                            html.Hr(),
                            results_table,
                        ]
                    ),
                ]
            ),
        ]
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
