# -*- coding: utf-8 -*-
import urllib.parse

import dash
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from gnpsdata import taskresult
from gnpsdata import workflow_classicnetworking
from gnpsdata import workflow_fbmn

from app import app


dash_app = dash.Dash(
    name="dashinterface2",
    server=app,
    url_base_pathname="/dashinterface2/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

dash_app.title = "GNPS2 Library Confidence"


def _get_url_param(param_dict, key, default=""):
    return param_dict.get(key, [default])[0]


def _find_first_column(df, candidates):
    candidate_map = {str(col).lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in candidate_map:
            return candidate_map[col.lower()]
    return None


def _canonical_structure_key(row):
    inchi_planar = str(row.get("InChIKey-Planar", "")).strip()
    if inchi_planar and inchi_planar.lower() != "nan":
        return inchi_planar

    inchikey = str(row.get("InChIKey", "")).strip()
    if inchikey and inchikey.lower() != "nan":
        return inchikey.split("-")[0]

    smiles = str(row.get("Smiles", "")).strip()
    if smiles and smiles.lower() != "nan":
        # Strip common stereochemistry markers from SMILES as a fallback.
        return (
            smiles.replace("@", "")
            .replace("/", "")
            .replace("\\", "")
            .replace("[", "")
            .replace("]", "")
        )

    inchi = str(row.get("INCHI", "")).strip()
    if inchi and inchi.lower() != "nan":
        return inchi.split("/t")[0].split("/m")[0].split("/s")[0]

    return "UNKNOWN_STRUCTURE"


def _confidence_category(score):
    if score >= 80:
        return "Reliable", "success"
    if score >= 55:
        return "Less Reliable", "warning"
    return "Unreliable", "danger"


def _load_library_matches(task_id, workflow_choice):
    loaders = []

    if workflow_choice == "classic":
        loaders = [
            lambda tid: workflow_classicnetworking.get_librarymatches_dataframe(tid, gnps2=True),
        ]
    elif workflow_choice == "fbmn":
        loaders = [
            lambda tid: workflow_fbmn.get_library_match_dataframe(tid, gnps2=True),
        ]
    else:
        loaders = [
            lambda tid: workflow_classicnetworking.get_librarymatches_dataframe(tid, gnps2=True),
            lambda tid: workflow_fbmn.get_library_match_dataframe(tid, gnps2=True),
        ]

    # Always keep a direct taskresult fallback for compatibility.
    loaders.extend(
        [
            lambda tid: taskresult.get_gnps2_task_resultfile_dataframe(
                tid,
                "nf_output/library/merged_results_with_gnps.tsv",
            ),
            lambda tid: taskresult.get_task_resultview_dataframe(tid, "librarymatches"),
        ]
    )

    errors = []
    for loader in loaders:
        try:
            df = loader(task_id)
            if df is not None and len(df) > 0:
                return df
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        "Could not load library matches for this task ID. "
        "Please verify the task is finished and has library matches. "
        + (" | ".join(errors[-2:]) if errors else "")
    )


def _compute_scan_level_confidence(df):
    scan_col = _find_first_column(df, ["#Scan#", "scan", "Scan"]) 
    cosine_col = _find_first_column(df, ["MQScore", "Cosine", "cosine"])
    compound_col = _find_first_column(df, ["Compound_Name", "compound_name", "Compound Name"])

    if scan_col is None:
        raise ValueError("Could not find scan column (#Scan# / scan / Scan) in library results.")
    if cosine_col is None:
        raise ValueError("Could not find cosine score column (MQScore / Cosine).")

    work_df = df.copy()
    work_df[scan_col] = work_df[scan_col].astype(str)
    work_df["_cosine"] = pd.to_numeric(work_df[cosine_col], errors="coerce").fillna(0.0)
    work_df["_structure_key"] = work_df.apply(_canonical_structure_key, axis=1)
    work_df["_compound"] = (
        work_df[compound_col].astype(str)
        if compound_col is not None
        else "Unknown"
    )

    summary_rows = []
    for scan, scan_df in work_df.groupby(scan_col):
        total_matches = len(scan_df)

        grouped = (
            scan_df.groupby("_structure_key", as_index=False)
            .agg(
                count=("_structure_key", "size"),
                mean_cosine=("_cosine", "mean"),
                max_cosine=("_cosine", "max"),
            )
            .sort_values(["count", "mean_cosine", "max_cosine"], ascending=False)
        )

        best = grouped.iloc[0]
        structure_key = best["_structure_key"]
        structure_hits = scan_df[scan_df["_structure_key"] == structure_key]
        top_compound = structure_hits["_compound"].mode().iloc[0]

        support_fraction = best["count"] / total_matches if total_matches else 0.0
        mean_cosine = float(best["mean_cosine"])
        max_cosine = float(best["max_cosine"])

        confidence_score = 100.0 * (
            0.60 * support_fraction + 0.30 * mean_cosine + 0.10 * max_cosine
        )
        confidence_label, badge_color = _confidence_category(confidence_score)

        summary_rows.append(
            {
                "scan": scan,
                "top_compound": top_compound,
                "structure_key": structure_key,
                "supporting_matches": int(best["count"]),
                "total_matches": int(total_matches),
                "mean_cosine": round(mean_cosine, 4),
                "max_cosine": round(max_cosine, 4),
                "confidence_score": round(confidence_score, 1),
                "confidence_label": confidence_label,
                "badge_color": badge_color,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(["confidence_score", "supporting_matches"], ascending=False)

    return summary_df


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
                            placeholder="Enter FBMN or Classical Networking task ID",
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
                            ],
                            value="auto",
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Button("Load Library Matches", id="run-analysis", color="primary", n_clicks=0),
                html.Hr(),
                html.P(
                    "Confidence is computed per scan using the fraction of supporting matches "
                    "for the same core structure and cosine quality.",
                    className="mb-1",
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
                dbc.Col(CONTROLS_CARD, md=12),
            ],
            style={"marginTop": 24},
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
    [Output("task-id", "value"), Output("workflow-choice", "value")],
    [Input("url", "search")],
)
def determine_task_from_url(search):
    try:
        query_dict = urllib.parse.parse_qs((search or "").lstrip("?"))
    except Exception:
        query_dict = {}

    task_id = _get_url_param(query_dict, "task", "")
    workflow_choice = _get_url_param(query_dict, "workflow", "auto")
    if workflow_choice not in {"auto", "classic", "fbmn"}:
        workflow_choice = "auto"

    return task_id, workflow_choice


@dash_app.callback(
    Output("results-panel", "children"),
    [Input("run-analysis", "n_clicks")],
    [State("task-id", "value"), State("workflow-choice", "value")],
)
def run_confidence_analysis(_n_clicks, task_id, workflow_choice):
    task_id = (task_id or "").strip()

    if not task_id:
        return dbc.Alert("Enter a GNPS2 task ID to start.", color="secondary")

    try:
        matches_df = _load_library_matches(task_id, workflow_choice)
        summary_df = _compute_scan_level_confidence(matches_df)
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
    table_df = table_df[
        [
            "scan",
            "top_compound",
            "structure_key",
            "support",
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
            {"name": "Proposed Compound", "id": "top_compound"},
            {"name": "Core Structure Key", "id": "structure_key"},
            {"name": "Support (same structure / total)", "id": "support"},
            {"name": "Mean Cosine", "id": "mean_cosine"},
            {"name": "Max Cosine", "id": "max_cosine"},
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
                "if": {"column_id": "top_compound"},
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
                    dbc.CardBody([results_table]),
                ]
            ),
        ]
    )


if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", port=5000, debug=True)