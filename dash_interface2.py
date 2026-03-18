# -*- coding: utf-8 -*-
import logging
import urllib.parse

import dash
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc

from app import app
from components import _build_detail_panel, _build_intro_panel
from constants import SUPPORTED_WORKFLOWS
from loader import _load_library_matches
from normalizer import _infer_resolved_workflow
from scorer import _compute_scan_level_confidence


dash_app = dash.Dash(
    name="dashinterface2",
    server=app,
    url_base_pathname="/dashinterface2/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

dash_app.title = "GNPS2 Library Confidence"

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
            "TopKonfidence: GNPS2 Library Annotation Confidence",
            href="/",
        ),
    ],
    color="light",
    dark=False,
    sticky="top",
)


CONTROLS_CARD = dbc.Card(
    [
        dbc.CardHeader(html.H5("Input")),
        dbc.CardBody(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("GNPS2 Task ID"),
                        dbc.Input(
                            id="task-id",
                            placeholder="Enter FBMN, Classical Networking, or Library Search task ID",
                            value="",
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Workflow"),
                        dbc.Select(
                            id="workflow-choice",
                            options=[
                                {"label": "Auto", "value": "auto"},
                                {"label": "Classical Networking", "value": "classic"},
                                {"label": "FBMN", "value": "fbmn"},
                                {"label": "Library Search", "value": "librarysearch"},
                            ],
                            value="auto",
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Checklist(
                    id="use-entropy",
                    options=[{"label": "Use spectral entropy similarity in score", "value": "enabled"}],
                    value=[],
                    switch=True,
                    className="mb-3",
                ),
                dbc.Button("Load Library Matches", id="run-analysis", color="primary", n_clicks=0),
                html.Hr(),
                html.P(
                    "Confidence is computed per scan using the fraction of supporting matches "
                    "for the same planar structure or positional-isomer family, entropy similarity, max cosine, and Library_Class quality.",
                    className="mb-1",
                ),
                html.Small(
                    "Score weights: support 50%, entropy similarity 25% (fallback to mean cosine if unavailable or disabled), max cosine 10%, Library_Class quality 15% (Class 1 > 2 > 3).",
                    className="text-muted d-block",
                ),
                html.Small(
                    "Auto mode now also tries the GNPS2 Library Search `librarymatches` JSON view.",
                    className="text-muted d-block",
                ),
                html.Small(
                    "Stereochemistry is ignored by using planar InChIKey connectivity (or fallbacks).",
                    className="text-muted",
                ),
            ]
        ),
    ]
)


BODY = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
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
                        id="loading-output",
                        type="default",
                        children=html.Div(id="results-panel"),
                    ),
                    md=12,
                ),
            ],
            style={"marginTop": 12},
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
    Output("results-panel", "children"),
    [Input("run-analysis", "n_clicks"), Input("url", "search")],
    [State("task-id", "value"), State("workflow-choice", "value"), State("use-entropy", "value")],
)
def run_confidence_analysis(_n_clicks, search, task_id, workflow_choice, use_entropy_value):
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
        return html.Div()

    if not task_id:
        return _build_intro_panel()

    if (_n_clicks is None or _n_clicks < 1) and not should_autorun:
        return _build_intro_panel()

    try:
        matches_df = _load_library_matches(task_id, workflow_choice)
        resolved_workflow = _infer_resolved_workflow(matches_df, workflow_choice)
        summary_df = _compute_scan_level_confidence(
            matches_df,
            task_id=task_id,
            workflow_choice=resolved_workflow,
            use_entropy_similarity=use_entropy_similarity,
        )
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger")

    if len(summary_df) == 0:
        return dbc.Alert(
            "No library matches were found for this task.",
            color="warning",
        )

    badge_counts = {
        "Reliable": int((summary_df["confidence_label"] == "Reliable").sum()),
        "Less Reliable": int((summary_df["confidence_label"] == "Less Reliable").sum()),
        "Unreliable": int((summary_df["confidence_label"] == "Unreliable").sum()),
    }

    table_df = summary_df.copy()
    table_df["support"] = (
        table_df["supporting_matches"].astype(str) + " / " + table_df["total_matches"].astype(str)
    )
    table_df["class_profile"] = (
        "C1:"
        + table_df["class_1_hits"].astype(str)
        + " | C2:"
        + table_df["class_2_hits"].astype(str)
        + " | C3:"
        + table_df["class_3_hits"].astype(str)
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
            "entropy_similarity",
            "entropy_source",
            "mean_cosine",
            "max_cosine",
            "library_class_quality",
            "class_profile",
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
            {"name": "Support (same planar structure or positional-isomer family / total)", "id": "support"},
            {"name": "Entropy Similarity", "id": "entropy_similarity"},
            {"name": "Entropy Source", "id": "entropy_source"},
            {"name": "Mean Cosine", "id": "mean_cosine"},
            {"name": "Max Cosine", "id": "max_cosine"},
            {"name": "Library Class Quality", "id": "library_class_quality"},
            {"name": "Class Hits (1/2/3)", "id": "class_profile"},
            {"name": "Score", "id": "confidence_score"},
            {"name": "Confidence", "id": "confidence_label"},
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
                "if": {"column_id": "library_class_quality"},
                "width": "120px",
                "maxWidth": "120px",
                "textAlign": "center",
            },
            {
                "if": {"column_id": "class_profile"},
                "width": "130px",
                "maxWidth": "130px",
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
                    "filter_query": '{confidence_label} = "Reliable"',
                    "column_id": "confidence_label",
                },
                "backgroundColor": "#d1e7dd",
                "color": "#0f5132",
            },
            {
                "if": {
                    "filter_query": '{confidence_label} = "Less Reliable"',
                    "column_id": "confidence_label",
                },
                "backgroundColor": "#fff3cd",
                "color": "#664d03",
            },
            {
                "if": {
                    "filter_query": '{confidence_label} = "Unreliable"',
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
                    dbc.CardHeader(html.H5("Confidence Summary")),
                    dbc.CardBody(
                        [
                            html.P(f"Task: {task_id}"),
                            html.P(f"Entropy scoring: {'Enabled' if use_entropy_similarity else 'Disabled'}"),
                            dbc.Badge(f"Reliable: {badge_counts['Reliable']}", color="success", className="me-2"),
                            dbc.Badge(
                                f"Less Reliable: {badge_counts['Less Reliable']}",
                                color="warning",
                                className="me-2",
                            ),
                            dbc.Badge(f"Unreliable: {badge_counts['Unreliable']}", color="danger"),
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


if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", port=5000, debug=True)
