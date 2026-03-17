# -*- coding: utf-8 -*-
import base64
from io import BytesIO
import urllib.parse

import dash
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Draw

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


SUPPORTED_WORKFLOWS = {"auto", "classic", "fbmn", "librarysearch"}


def _get_url_param(param_dict, key, default=""):
    return param_dict.get(key, [default])[0]


def _find_first_column(df, candidates):
    candidate_map = {str(col).lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in candidate_map:
            return candidate_map[col.lower()]
    return None


def _normalize_library_matches_dataframe(df):
    if df is None or len(df) == 0:
        return df

    work_df = df.copy()
    canonical_columns = {
        "#Scan#": ["#Scan#", "scan", "Scan"],
        "MQScore": ["MQScore", "Cosine", "cosine"],
        "Compound_Name": ["Compound_Name", "compound_name", "Compound Name", "CompoundName"],
        "SpectrumFile": ["SpectrumFile", "spectrumfile", "Spectrum File"],
        "SpectrumID": ["SpectrumID", "spectrumid", "Spectrum ID"],
        "library_usi": ["library_usi", "LibraryUSI", "Library_USI", "library usi"],
        "Smiles": ["Smiles", "SMILES", "smiles"],
        "InChIKey": ["InChIKey", "inchikey", "INCHIKEY"],
        "InChIKey-Planar": ["InChIKey-Planar", "inchikey-planar", "INCHIKEY-PLANAR"],
        "INCHI": ["INCHI", "InChI", "inchi"],
    }

    for canonical_name, candidates in canonical_columns.items():
        if canonical_name in work_df.columns:
            continue

        matched_column = _find_first_column(work_df, candidates)
        if matched_column is not None:
            work_df = work_df.rename(columns={matched_column: canonical_name})

    return work_df


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


def _get_direct_resultview_dataframe(task_id, view_name):
    response = requests.get(
        "https://gnps2.org/result",
        params={"task": task_id, "viewname": view_name, "json": ""},
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        for key in ["data", "results", "rows"]:
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
        if len(payload) > 0:
            return pd.DataFrame([payload])

    return pd.DataFrame()


def _load_library_matches(task_id, workflow_choice):
    workflow_choice = (workflow_choice or "auto").strip().lower()
    loaders = []

    if workflow_choice == "classic":
        loaders.extend(
            [
                (
                    "classical-networking loader",
                    lambda tid: workflow_classicnetworking.get_librarymatches_dataframe(tid, gnps2=True),
                ),
            ]
        )
    elif workflow_choice == "fbmn":
        loaders.extend(
            [
                (
                    "fbmn loader",
                    lambda tid: workflow_fbmn.get_library_match_dataframe(tid, gnps2=True),
                ),
            ]
        )
    elif workflow_choice == "librarysearch":
        loaders.extend(
            [
                (
                    "direct GNPS2 librarymatches JSON view",
                    lambda tid: _get_direct_resultview_dataframe(tid, "summarylist"),
                ),
            ]
        )
    else:
        loaders.extend(
            [
                (
                    "direct GNPS2 librarymatches JSON view",
                    lambda tid: _get_direct_resultview_dataframe(tid, "librarymatches"),
                ),
                (
                    "classical-networking loader",
                    lambda tid: workflow_classicnetworking.get_librarymatches_dataframe(tid, gnps2=True),
                ),
                (
                    "fbmn loader",
                    lambda tid: workflow_fbmn.get_library_match_dataframe(tid, gnps2=True),
                ),
            ]
        )

    # Keep taskresult-based fallbacks for compatibility across GNPS2 workflows.
    loaders.extend(
        [
            (
                "taskresult merged_results_with_gnps.tsv",
                lambda tid: taskresult.get_gnps2_task_resultfile_dataframe(
                    tid,
                    "nf_output/library/merged_results_with_gnps.tsv",
                ),
            ),
            (
                "taskresult librarymatches view",
                lambda tid: taskresult.get_task_resultview_dataframe(tid, "librarymatches"),
            ),
        ]
    )

    errors = []
    for loader_name, loader in loaders:
        try:
            df = _normalize_library_matches_dataframe(loader(task_id))
            if df is not None and len(df) > 0:
                return df
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    raise ValueError(
        "Could not load library matches for this task ID. "
        "Please verify the task is finished and has library matches. "
        + (" | ".join(errors[-3:]) if errors else "")
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


def _to_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _render_mol_png_data_uri(smiles="", inchi=""):
    smiles = _to_text(smiles)
    inchi = _to_text(inchi)

    mol = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)

    if mol is None and inchi:
        mol = Chem.MolFromInchi(inchi)

    if mol is None:
        return ""

    image = Draw.MolToImage(mol, size=(220, 220))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded_image}"


def _normalize_library_usi(library_usi):
    library_usi = _to_text(library_usi)
    if not library_usi:
        return ""

    if "mzspec:GNPS:GNPS-LIBRARY:accession:" in library_usi:
        return library_usi

    prefix = "mzspec:GNPS:GNPS-LIBRARY:"
    if library_usi.startswith(prefix):
        suffix = library_usi[len(prefix):]
        if suffix.startswith("accession:"):
            return f"{prefix}{suffix}"
        if suffix.startswith("CCMSLIB"):
            return f"{prefix}accession:{suffix}"

    if library_usi.startswith("CCMSLIB"):
        return f"{prefix}accession:{library_usi}"

    return library_usi


def _infer_resolved_workflow(matches_df, workflow_choice):
    workflow_choice = (workflow_choice or "auto").strip().lower()
    if workflow_choice in SUPPORTED_WORKFLOWS and workflow_choice != "auto":
        return workflow_choice

    spectrum_file_col = _find_first_column(matches_df, ["SpectrumFile"])
    if spectrum_file_col is None:
        return "auto"

    observed_names = {
        _to_text(value).split("/")[-1]
        for value in matches_df[spectrum_file_col].dropna().head(25).tolist()
        if _to_text(value)
    }
    generated_names = {"spectra_filtered.mgf", "spectra_reformatted.mgf"}

    if observed_names and observed_names.isdisjoint(generated_names):
        return "librarysearch"

    return "auto"


def _build_query_usi(task_id, scan, workflow_choice, spectrum_file=""):
    if workflow_choice == "librarysearch":
        spectrum_file = _to_text(spectrum_file).lstrip("/")
        if not spectrum_file:
            return ""

        scan = str(scan)
        return f"mzspec:GNPS2:TASK-{task_id}-inputspectra/{spectrum_file}:scan:{scan}"

    scan = str(scan)
    return f"mzspec:GNPS2:TASK-{task_id}-nf_output/clustering/spectra_reformatted.mgf:scan:{scan}"


def _build_mirror_plot_link(query_usi, library_usi):
    query_usi = _to_text(query_usi)
    library_usi = _normalize_library_usi(library_usi)
    if not query_usi or not library_usi:
        return ""

    params = urllib.parse.urlencode({"usi1": query_usi, "usi2": library_usi})
    return f"https://metabolomics-usi.gnps2.org/dashinterface/?{params}"


def _build_detail_panel(task_id, scan, workflow_choice):
    matches_df = _load_library_matches(task_id, workflow_choice)
    resolved_workflow = _infer_resolved_workflow(matches_df, workflow_choice)
    summary_df = _compute_scan_level_confidence(matches_df)

    scan_col = _find_first_column(matches_df, ["#Scan#", "scan", "Scan"])
    if scan_col is None:
        raise ValueError("Could not find scan column in library results.")

    scan_df = matches_df[matches_df[scan_col].astype(str) == str(scan)].copy()
    if len(scan_df) == 0:
        raise ValueError(f"No matches found for scan {scan} in task {task_id}.")

    score_row = summary_df[summary_df["scan"].astype(str) == str(scan)]
    if len(score_row) == 0:
        raise ValueError(f"Could not compute confidence metrics for scan {scan}.")
    score_row = score_row.iloc[0]

    cosine_col = _find_first_column(scan_df, ["MQScore", "Cosine", "cosine"])
    compound_col = _find_first_column(scan_df, ["Compound_Name", "compound_name", "Compound Name"])
    library_usi_col = _find_first_column(scan_df, ["library_usi", "LibraryUSI", "library usi"])
    spectrum_file_col = _find_first_column(scan_df, ["SpectrumFile"])
    smiles_col = _find_first_column(scan_df, ["Smiles", "SMILES", "smiles"])
    inchikey_col = _find_first_column(scan_df, ["InChIKey", "inchikey", "INCHIKEY"])
    inchi_col = _find_first_column(scan_df, ["INCHI", "InChI", "inchi"])

    metric_card = dbc.Card(
        [
            dbc.CardHeader(html.H5(f"Detail View: Task {task_id}, Scan {scan}")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(html.Div([html.Strong("Confidence Badge"), html.Br(), score_row["confidence_label"]]), md=3),
                            dbc.Col(html.Div([html.Strong("Confidence Score"), html.Br(), score_row["confidence_score"]]), md=3),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Strong("Support"),
                                        html.Br(),
                                        f"{score_row['supporting_matches']} / {score_row['total_matches']}",
                                    ]
                                ),
                                md=3,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Strong("Cosine (mean / max)"),
                                        html.Br(),
                                        f"{score_row['mean_cosine']} / {score_row['max_cosine']}",
                                    ]
                                ),
                                md=3,
                            ),
                        ]
                    ),
                    html.Hr(),
                    html.A(
                        "Back to table",
                        href=f"/dashinterface2/?task={urllib.parse.quote(str(task_id))}&workflow={urllib.parse.quote(str(workflow_choice))}",
                    ),
                ]
            ),
        ],
        className="mb-3",
    )

    spectrum_file = ""
    if spectrum_file_col is not None and len(scan_df) > 0:
        spectrum_file = _to_text(scan_df.iloc[0][spectrum_file_col])

    query_usi = _build_query_usi(task_id, scan, resolved_workflow, spectrum_file=spectrum_file)
    scan_df["_cosine"] = pd.to_numeric(scan_df[cosine_col], errors="coerce").fillna(0.0) if cosine_col else 0.0
    scan_df["_compound"] = scan_df[compound_col].astype(str) if compound_col else "Unknown"
    scan_df["_library_usi"] = scan_df[library_usi_col].astype(str) if library_usi_col else ""
    scan_df["_smiles"] = scan_df[smiles_col].astype(str) if smiles_col else ""
    scan_df["_inchikey"] = scan_df[inchikey_col].astype(str) if inchikey_col else ""
    scan_df["_inchi"] = scan_df[inchi_col].astype(str) if inchi_col else ""

    detail_rows = (
        scan_df.sort_values("_cosine", ascending=False)
        .drop_duplicates(subset=["_compound", "_library_usi", "_smiles", "_inchikey"])
        .head(25)
    )

    cards = []
    for _, row in detail_rows.iterrows():
        compound = _to_text(row.get("_compound")) or "Unknown Compound"
        lib_usi = _to_text(row.get("_library_usi"))
        smiles = _to_text(row.get("_smiles"))
        inchikey = _to_text(row.get("_inchikey"))
        inchi = _to_text(row.get("_inchi"))
        cosine = row.get("_cosine", 0.0)

        image_url = _render_mol_png_data_uri(smiles=smiles, inchi=inchi)
        normalized_library_usi = _normalize_library_usi(lib_usi)

        mirror_url = ""
        if normalized_library_usi:
            mirror_url = _build_mirror_plot_link(query_usi, normalized_library_usi)

        card_body = [
            html.H6(compound),
            html.Div(f"Cosine: {round(float(cosine), 4)}"),
            html.Small(
                f"Library USI: {normalized_library_usi}" if normalized_library_usi else "Library USI unavailable",
                className="text-muted",
            ),
            html.Br(),
            html.Small(f"InChIKey: {inchikey}" if inchikey else "InChIKey unavailable", className="text-muted"),
            html.Br(),
            html.Small(
                "SMILES/InChI available" if (smiles or inchi) else "SMILES/InChI unavailable",
                className="text-muted",
            ),
            html.Hr(),
            html.H6("Mirror Plot"),
            html.Div(
                html.A("Open mirror plot", href=mirror_url, target="_blank")
                if mirror_url
                else "Mirror plot link unavailable for this hit"
            ),
        ]

        if image_url:
            card_body.insert(
                1,
                html.Img(
                    src=image_url,
                    style={"width": "180px", "height": "180px", "objectFit": "contain"},
                ),
            )

        cards.append(
            dbc.Col(
                dbc.Card(dbc.CardBody(card_body), className="mb-3"),
                lg=4,
                md=6,
                sm=12,
            )
        )

    compounds_card = dbc.Card(
        [
            dbc.CardHeader(html.H5("Matched Compound Structures")),
            dbc.CardBody(dbc.Row(cards) if cards else html.Div("No compounds to display.")),
        ]
    )

    return html.Div([metric_card, compounds_card])


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
                dbc.Button("Load Library Matches", id="run-analysis", color="primary", n_clicks=0),
                html.Hr(),
                html.P(
                    "Confidence is computed per scan using the fraction of supporting matches "
                    "for the same core structure and cosine quality.",
                    className="mb-1",
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
    if workflow_choice not in SUPPORTED_WORKFLOWS:
        workflow_choice = "auto"

    return task_id, workflow_choice


@dash_app.callback(
    Output("controls-container", "style"),
    [Input("url", "search")],
)
def toggle_controls_for_detail_view(search):
    try:
        query_dict = urllib.parse.parse_qs((search or "").lstrip("?"))
    except Exception:
        query_dict = {}

    detail_mode = _get_url_param(query_dict, "detail", "0") == "1"
    if detail_mode:
        return {"display": "none"}
    return {"display": "block"}


@dash_app.callback(
    Output("detail-panel", "children"),
    [Input("url", "search")],
)
def render_detail_panel(search):
    try:
        query_dict = urllib.parse.parse_qs((search or "").lstrip("?"))
    except Exception:
        query_dict = {}

    detail_mode = _get_url_param(query_dict, "detail", "0") == "1"
    if not detail_mode:
        return html.Div()

    task_id = _get_url_param(query_dict, "task", "").strip()
    scan = _get_url_param(query_dict, "scan", "").strip()
    workflow_choice = _get_url_param(query_dict, "workflow", "auto").strip() or "auto"
    if workflow_choice not in SUPPORTED_WORKFLOWS:
        workflow_choice = "auto"

    if not task_id or not scan:
        return dbc.Alert("Detail view requires task and scan query parameters.", color="warning")

    try:
        return _build_detail_panel(task_id, scan, workflow_choice)
    except Exception as exc:
        return dbc.Alert(str(exc), color="danger")


@dash_app.callback(
    Output("results-panel", "children"),
    [Input("run-analysis", "n_clicks")],
    [State("task-id", "value"), State("workflow-choice", "value")],
)
def run_confidence_analysis(_n_clicks, task_id, workflow_choice):
    task_id = (task_id or "").strip()
    workflow_choice = (workflow_choice or "auto").strip().lower()
    if workflow_choice not in SUPPORTED_WORKFLOWS:
        workflow_choice = "auto"

    if not task_id:
        return dbc.Alert("Enter a GNPS2 task ID to start.", color="secondary")

    if _n_clicks is None or _n_clicks < 1:
        return html.Div()

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
    table_df["compound_link"] = table_df.apply(
        lambda row: (
            f"[{row['top_compound']}](/dashinterface2/?task={urllib.parse.quote(str(task_id))}"
            f"&workflow={urllib.parse.quote(str(workflow_choice))}&scan={urllib.parse.quote(str(row['scan']))}&detail=1)"
        ),
        axis=1,
    )
    table_df = table_df[
        [
            "scan",
            "compound_link",
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
            {"name": "Proposed Compound", "id": "compound_link", "presentation": "markdown"},
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
