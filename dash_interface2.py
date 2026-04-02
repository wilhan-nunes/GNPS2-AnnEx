# -*- coding: utf-8 -*-
import logging
import urllib.parse

import dash
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

from app import app
from components import (
    _build_detail_panel,
    _build_tanimoto_heatmap_figure,
    _render_mol_png_data_uri,
    _search_pubchem_candidates,
)
from bin.normalizer import _find_first_column, _infer_resolved_workflow, _truncate_text
from bin.constants import SUPPORTED_WORKFLOWS
import io

import pandas as pd

from bin.loader import _load_library_matches
from bin.scorer import _compute_scan_level_confidence


dash_app = dash.Dash(
    name="dashinterface2",
    server=app,
    url_base_pathname="/dashinterface2/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

dash_app.title = "GNPS2-AnnEx"

logger = logging.getLogger(__name__)

LAPLACE_ALPHA = 4.0
LAPLACE_BETA = 5.0


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
        dcc.Store(id="scan-data-store", storage_type="memory"),
        dcc.Store(id="pubchem-overrides-store", storage_type="memory"),
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
                    html.Div(
                        [
                            html.Div(id="detail-score-panel"),
                            html.Div(id="detail-compounds-panel"),
                        ],
                        id="detail-panel",
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
                            style={"borderRadius": "14px", "background": "rgba(255,255,255,0.9)"},
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
                        type="dot",
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
    [
        Output("detail-score-panel", "children"),
        Output("detail-compounds-panel", "children"),
        Output("scan-data-store", "data"),
    ],
    [Input("url", "search")],
)
def render_detail_panel(search):
    query_state = _parse_dashboard_query(search)

    if not query_state["detail_mode"]:
        return html.Div(), html.Div(), None

    task_id = query_state["task_id"]
    scan = query_state["scan"]
    workflow_choice = query_state["workflow_choice"]
    use_entropy_similarity = query_state["use_entropy_similarity"]

    if not task_id or not scan:
        return dbc.Alert("Detail view requires task and scan query parameters.", color="warning"), html.Div(), None

    try:
        panel_ui, compounds_data = _build_detail_panel(
            task_id,
            scan,
            workflow_choice,
            use_entropy_similarity=use_entropy_similarity,
        )
        score_panel = panel_ui.children[0] if getattr(panel_ui, "children", None) else panel_ui
        compounds_panel = panel_ui.children[1] if getattr(panel_ui, "children", None) else html.Div()
        return score_panel, compounds_panel, {
            "compounds": compounds_data,
            "task_id": task_id,
            "scan": scan,
            "workflow_choice": workflow_choice,
            "use_entropy_similarity": use_entropy_similarity,
        }
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger"), html.Div(), None


@dash_app.callback(
    Output("detail-score-panel", "children", allow_duplicate=True),
    [
        Input("scan-data-store", "data"),
        Input("pubchem-overrides-store", "data"),
        Input("pubchem-enabled", "value"),
    ],
    prevent_initial_call=True,
)
def refresh_detail_score_panel(scan_data, pubchem_overrides, pubchem_value):
    """Update only the score panel so PubChem selection does not remount compound cards."""
    if not scan_data:
        return html.Div()

    task_id = _truncate_text(str(scan_data.get("task_id", "")), max_length=200)
    scan = _truncate_text(str(scan_data.get("scan", "")), max_length=200)
    workflow_choice = _truncate_text(str(scan_data.get("workflow_choice", "auto")), max_length=40)
    use_entropy_similarity = bool(scan_data.get("use_entropy_similarity", False))
    use_pubchem_for_score = isinstance(pubchem_value, list) and "enabled" in pubchem_value

    if not task_id or not scan:
        return html.Div()

    try:
        panel_ui, _ = _build_detail_panel(
            task_id,
            scan,
            workflow_choice,
            use_entropy_similarity=use_entropy_similarity,
            pubchem_overrides=pubchem_overrides,
            use_pubchem_for_score=use_pubchem_for_score,
        )
        return panel_ui.children[0] if getattr(panel_ui, "children", None) else panel_ui
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger")


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

    status = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        dbc.Badge("Loading", color="primary", className="me-2"),
                        html.Span("Preparing scan-level confidence summary", className="fw-semibold"),
                    ],
                    className="d-flex align-items-center mb-2",
                ),
                html.Div(
                    [
                        dbc.Badge(f"{n_matches:,} matches", color="light", text_color="dark", className="me-2"),
                        dbc.Badge(f"{n_scans:,} scans", color="light", text_color="dark"),
                    ],
                    className="mb-2",
                ),
                html.Div(
                    [
                        html.Span(
                            "Scoring model: structure agreement, Tanimoto ECFP4, cosine.",
                            className="text-muted",
                        ),
                        html.Span(entropy_note, className="text-muted"),
                        html.Span(" Results will appear below.", className="text-muted"),
                    ]
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
        "Consistent evidence": int((summary_df["confidence_label"] == "Consistent evidence").sum()),
        "Inconclusive": int((summary_df["confidence_label"] == "Inconclusive").sum()),
        "Inconsistent evidence": int((summary_df["confidence_label"] == "Inconsistent evidence").sum()),
    }

    table_df = summary_df.copy()
    table_df["adjusted_support_score"] = (
        (table_df["supporting_matches"].astype(float) + LAPLACE_ALPHA)
        / (table_df["total_matches"].astype(float) + LAPLACE_BETA)
    ).round(3)
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
            "precursor_mz",
            "compound_link",
            "support",
            "adjusted_support_score",
            "confidence_score",
            "confidence_label",
            "structure_key",
            "tanimoto_score",
            "mean_cosine",
            "max_cosine",
            "entropy_similarity",
            "tanimoto_source",
            "entropy_source",
            "missing_structures",
        ]
    ]

    results_table = dag.AgGrid(
        id="confidence-datatable",
        rowData=table_df.to_dict("records"),
        columnDefs=[
            {"headerName": "Scan", "field": "scan", "maxWidth": 95},
            {"headerName": "Prec. m/z", "field": "precursor_mz", "maxWidth": 110},
            {
                "headerName": "Proposed Compound",
                "field": "compound_link",
                "cellRenderer": "markdown",
                "minWidth": 180,
            },
            {
                "headerName": "Support (planar structure or isomer family / total)",
                "field": "support",
                "minWidth": 220,
            },
            {
                "headerName": f"Adjusted Support (Laplace α={LAPLACE_ALPHA:g}, β={LAPLACE_BETA:g})",
                "field": "adjusted_support_score",
                "maxWidth": 190,
                "cellStyle": {
                    "styleConditions": [
                        {
                            "condition": "params.value >= 0.9",
                            "style": {"backgroundColor": "#0c2c84", "color": "#f7fbff", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value >= 0.8 && params.value < 0.9",
                            "style": {"backgroundColor": "#225ea8", "color": "#f7fbff", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value >= 0.7 && params.value < 0.8",
                            "style": {"backgroundColor": "#1d91c0", "color": "#f7fbff", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value >= 0.6 && params.value < 0.7",
                            "style": {"backgroundColor": "#41b6c4", "color": "#08306b", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value >= 0.5 && params.value < 0.6",
                            "style": {"backgroundColor": "#7fcdbb", "color": "#08306b", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value < 0.5",
                            "style": {"backgroundColor": "#ccece6", "color": "#0b3d4f", "fontWeight": 600, "textAlign": "center"},
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
                            "style": {"backgroundColor": "#d1e7dd", "color": "#0f5132", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value === 'Inconclusive'",
                            "style": {"backgroundColor": "#fff3cd", "color": "#664d03", "fontWeight": 600, "textAlign": "center"},
                        },
                        {
                            "condition": "params.value === 'Inconsistent evidence'",
                            "style": {"backgroundColor": "#f8d7da", "color": "#842029", "fontWeight": 600, "textAlign": "center"},
                        },
                    ],
                    "defaultStyle": {"fontWeight": 600, "textAlign": "center"},
                },
            },
            {"headerName": "Representative Structure Key", "field": "structure_key", "minWidth": 220},
            {
                "headerName": "Tanimoto (ECFP4)",
                "field": "tanimoto_score",
                "maxWidth": 120,
                "cellStyle": {"textAlign": "center"},
            },
            {"headerName": "Mean Cosine", "field": "mean_cosine", "maxWidth": 120, "cellStyle": {"textAlign": "center"}},
            {"headerName": "Max Cosine", "field": "max_cosine", "maxWidth": 120, "cellStyle": {"textAlign": "center"}},
            {
                "headerName": "Entropy Similarity",
                "field": "entropy_similarity",
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
                "headerName": "Entropy Source",
                "field": "entropy_source",
                "minWidth": 165,
                "cellStyle": {"textAlign": "center"},
            },
            {
                "headerName": "Missing structures (no SMILES/InChIKey)",
                "field": "missing_structures",
                "maxWidth": 145,
                "cellStyle": {"textAlign": "center"},
            },
        ],
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
                                    f" · Entropy scoring: {'enabled' if use_entropy_similarity else 'disabled'}",
                                ]
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
    Output({"type": "pubchem-candidates-div", "index": MATCH}, "children"),
    Input({"type": "pubchem-search-btn", "index": MATCH}, "n_clicks"),
    State({"type": "pubchem-name-input", "index": MATCH}, "value"),
    State({"type": "compound-info-store", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def search_pubchem_for_card(n_clicks, manual_name_query, compound_info):
    """Fetch PubChem candidates for one compound card and render a dropdown picker."""
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
        query_note = f" | query: {matched_name}" if source_label == "name-fuzzy" and matched_name else ""
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
    """Update structure previews on missing-structure cards after PubChem selection."""
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
                            style={"width": "180px", "height": "180px", "objectFit": "contain"},
                        ),
                        html.Small(
                            "Structure from selected PubChem match"
                            if selected_smiles
                            else "Structure from library/InChI",
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
    [Input("url", "search")],
)
def toggle_pubchem_controls(search):
    if _parse_dashboard_query(search)["detail_mode"]:
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

    use_selected_pubchem = isinstance(pubchem_value, list) and "enabled" in pubchem_value
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
        return dbc.Alert("RDKit/Plotly unavailable — cannot build Tanimoto heatmap.", color="warning")

    n_missing = sum(1 for c in compounds_data if not c.get("smiles"))

    if applied_overrides:
        source_note = f"{applied_overrides} structure(s) updated via selected PubChem SMILES."
    elif overrides and not use_selected_pubchem:
        source_note = "PubChem selections are available but currently disabled by the toggle."
    elif n_missing:
        source_note = (
            f"{n_missing} compound(s) have no SMILES — cells are blank. "
            "Type a name and use 'Search PubChem' on the cards above, then enable the toggle below."
        )
    else:
        source_note = "All SMILES sourced from library."

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Pairwise Tanimoto Similarity Heatmap")),
            dbc.CardBody(
                [
                    html.Small(
                        [
                            f"ECFP4 pairwise Tanimoto across {len(compounds_data)} unique compound(s). "
                            "Green = structurally similar, red = dissimilar. Upper triangle omitted. ",
                            html.Span(source_note, className="text-muted"),
                        ]
                    ),
                    html.Hr(),
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                ]
            ),
        ],
        className="mb-3",
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
