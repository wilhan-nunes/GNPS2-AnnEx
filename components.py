# -*- coding: utf-8 -*-
import base64
import logging
import math
import urllib.parse
from io import BytesIO

import dash_bootstrap_components as dbc
import pandas as pd
from dash import html
from rdkit import Chem
from rdkit.Chem import Draw

from bin.loader import _load_library_matches
from bin.normalizer import (
    _annotate_support_groups,
    _build_query_usi,
    _canonical_structure_key,
    _find_first_column,
    _infer_resolved_workflow,
    _normalize_library_usi,
    _to_text,
    _truncate_text,
)
from bin.scorer import _compute_scan_level_confidence

logger = logging.getLogger(__name__)


def _format_workflow_label(workflow_choice):
    workflow_label_map = {
        "auto": "Auto",
        "classic": "Classical Networking",
        "fbmn": "FBMN",
        "librarysearch": "Library Search",
    }
    return workflow_label_map.get(_to_text(workflow_choice).lower(), workflow_choice)


def _format_entropy_source_label(entropy_source):
    source_label_map = {
        "spectral_entropy": "Spectral entropy similarity",
        "mean_cosine_fallback": "Mean cosine fallback",
        "disabled": "Entropy disabled; mean cosine fallback",
    }
    return source_label_map.get(_to_text(entropy_source), "Similarity evidence")


def _format_tanimoto_source_label(tanimoto_source):
    source_label_map = {
        "tanimoto_ecfp4": "Tanimoto (ECFP4, mean pairwise)",
        "single_structure": "Tanimoto (single structure — perfect consensus)",
        "structure_agreement_fallback": "Structure agreement fallback (too few SMILES for Tanimoto)",
        "unavailable": "Unavailable (RDKit error)",
    }
    return source_label_map.get(_to_text(tanimoto_source), "Tanimoto structural similarity")


def _metric_bar_color(value):
    if value >= 0.75:
        return "#1d9a6c"
    if value >= 0.45:
        return "#d48a18"
    return "#d35d47"


def _confidence_palette(confidence_label):
    confidence_label = _to_text(confidence_label)
    palette_map = {
        "Well-supported": {
            "color": "#1d9a6c",
            "badge_background": "#dff4ea",
            "badge_border": "#8dd5b7",
            "tone": "Strong structural evidence",
        },
        "Ambiguous": {
            "color": "#d48a18",
            "badge_background": "#fff1d6",
            "badge_border": "#f0cb7e",
            "tone": "Structural ambiguity present",
        },
        "Insufficient evidence": {
            "color": "#d35d47",
            "badge_background": "#fde2de",
            "badge_border": "#efb0a5",
            "tone": "Weak or inconclusive evidence",
        },
    }
    return palette_map.get(
        confidence_label,
        {
            "color": "#58708b",
            "badge_background": "#e9eef4",
            "badge_border": "#c5d0da",
            "tone": "Evidence summary",
        },
    )


def _describe_svg_arc(cx, cy, radius, start_deg, end_deg):
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)
    start_x = cx + radius * math.cos(start_rad)
    start_y = cy + radius * math.sin(start_rad)
    end_x = cx + radius * math.cos(end_rad)
    end_y = cy + radius * math.sin(end_rad)
    large_arc = 1 if (end_deg - start_deg) > 180 else 0
    return (
        f"M {start_x:.3f} {start_y:.3f} "
        f"A {radius} {radius} 0 {large_arc} 1 {end_x:.3f} {end_y:.3f}"
    )


def _build_reliability_gauge_data_uri(score, color):
    score_fraction = max(0.0, min(1.0, float(score) / 100.0))
    cx = 70
    cy = 65   # raised so arc endpoints (y ≈ 102) sit within the 110-high viewBox
    radius = 52
    # In SVG, sin is positive going DOWN, so:
    #   135° = lower-left  (cos < 0, sin > 0)  → 7-8 o'clock
    #   270° = top         (cos = 0, sin < 0)
    #   405° = 45° = lower-right (cos > 0, sin > 0) → 4-5 o'clock
    # Sweeping 270° clockwise takes the arc up and over the top — standard speedometer shape.
    start_deg = 135
    sweep_deg = 270
    end_deg = start_deg + sweep_deg * score_fraction
    background_arc = _describe_svg_arc(cx, cy, radius, start_deg, start_deg + sweep_deg)
    score_arc = _describe_svg_arc(cx, cy, radius, start_deg, end_deg)
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 110">
      <path d="{background_arc}" fill="none" stroke="#dce6ef" stroke-width="10" stroke-linecap="round"/>
      <path d="{score_arc}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
    </svg>
    """
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def _build_reliability_breakdown_panel(
    task_id,
    scan,
    workflow_choice,
    resolved_workflow,
    score_row,
    use_entropy_similarity,
):
    confidence_score = float(score_row["confidence_score"])
    confidence_label = _to_text(score_row["confidence_label"])
    palette = _confidence_palette(confidence_label)
    supporting_matches = int(score_row["supporting_matches"])
    total_matches = int(score_row["total_matches"])
    support_fraction = (supporting_matches / total_matches) if total_matches else 0.0
    similarity_value = float(score_row["entropy_similarity"])
    entropy_source = _to_text(score_row["entropy_source"])
    tanimoto_score = float(score_row["tanimoto_score"])
    tanimoto_source = _to_text(score_row["tanimoto_source"])
    max_cosine = float(score_row["max_cosine"])
    mean_cosine = float(score_row["mean_cosine"])
    top_compound = _to_text(score_row["top_compound"]) or "Unknown Compound"
    structure_key = _to_text(score_row["structure_key"])
    gauge_src = _build_reliability_gauge_data_uri(confidence_score, palette["color"])

    similarity_formula = (
        "entropy similarity × 25%"
        if entropy_source == "spectral_entropy"
        else "mean cosine fallback × 25%"
    )
    similarity_secondary = (
        f"Mean cosine: {mean_cosine:.4f}"
        if entropy_source == "spectral_entropy"
        else "Used because entropy was unavailable or disabled"
    )

    tanimoto_label = _format_tanimoto_source_label(tanimoto_source)
    tanimoto_secondary = (
        "Mean pairwise Tanimoto (ECFP4, r=2, 2048-bit) across unique SMILES in this scan"
        if tanimoto_source == "tanimoto_ecfp4"
        else "Fell back to hit-count structure agreement (fewer than 2 unique valid SMILES)"
        if tanimoto_source == "structure_agreement_fallback"
        else "All hits agree on one structure — no pairs to compare"
        if tanimoto_source == "single_structure"
        else "RDKit fingerprint computation unavailable"
    )

    metrics = [
        {
            "label": "Structure agreement",
            "formula": "same planar structure or positional-isomer family / total hits × 50%",
            "weight": 0.50,
            "value": support_fraction,
            "raw_primary": f"{supporting_matches} / {total_matches} hits support the same planar structure or positional-isomer family",
            "raw_secondary": _truncate_text(structure_key, max_length=48),
        },
        {
            "label": "Tanimoto structural similarity",
            "formula": f"mean pairwise ECFP4 Tanimoto × 15% · source: {tanimoto_label}",
            "weight": 0.15,
            "value": tanimoto_score,
            "raw_primary": f"{tanimoto_score:.4f} · {tanimoto_label}",
            "raw_secondary": tanimoto_secondary,
        },
        {
            "label": "Spectral similarity",
            "formula": similarity_formula,
            "weight": 0.25,
            "value": similarity_value,
            "raw_primary": f"{similarity_value:.4f} from {_format_entropy_source_label(entropy_source).lower()}",
            "raw_secondary": similarity_secondary,
        },
        {
            "label": "Best cosine score",
            "formula": "max cosine × 10%",
            "weight": 0.10,
            "value": max_cosine,
            "raw_primary": f"{max_cosine:.4f} strongest library hit",
            "raw_secondary": f"Mean cosine across supporting hits: {mean_cosine:.4f}",
        },
    ]

    metadata_fields = [
        ("Task ID", task_id),
        ("Scan", scan),
        ("Workflow", _format_workflow_label(resolved_workflow or workflow_choice)),
        ("Spectral sim. source", _format_entropy_source_label(entropy_source)),
        ("Tanimoto source", tanimoto_label),
        ("Structure support", f"{supporting_matches} / {total_matches}"),
    ]

    metric_rows = []
    for metric in metrics:
        value = max(0.0, min(1.0, float(metric["value"])))
        color = _metric_bar_color(value)
        contribution = value * float(metric["weight"]) * 100.0
        metric_rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            html.Strong(metric["label"]),
                            html.Span(metric["formula"]),
                        ],
                        className="td-component",
                    ),
                    html.Td(f"{int(metric['weight'] * 100)}%", className="td-weight"),
                    html.Td(
                        html.Div(
                            [
                                html.Div(
                                    html.Div(
                                        className="bar-fill",
                                        style={
                                            "width": f"{value * 100:.0f}%",
                                            "background": color,
                                        },
                                    ),
                                    className="bar-bg",
                                ),
                                html.Span(f"{value:.3f}", className="bar-val", style={"color": color}),
                            ],
                            className="bar-cell",
                        )
                    ),
                    html.Td(
                        [
                            html.Span(metric["raw_primary"], className="td-raw-primary"),
                            html.Span(metric["raw_secondary"], className="td-raw-secondary"),
                        ],
                        className="td-raw",
                    ),
                    html.Td(f"{contribution:.1f}", className="td-contrib", style={"color": color}),
                ]
            )
        )

    metric_rows.append(
        html.Tr(
            [
                html.Td(html.Span("Annotation evidence score", className="total-label")),
                html.Td(),
                html.Td(),
                html.Td(),
                html.Td(
                    f"{confidence_score:.1f}",
                    className="total-score",
                    style={"color": palette["color"]},
                ),
            ],
            className="total-row",
        )
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(label, className="query-label"),
                            html.Span(value, className="query-val"),
                        ],
                        className="query-field",
                    )
                    for label, value in metadata_fields
                ],
                className="query-bar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(src=gauge_src, className="gauge-svg", alt="Reliability score gauge"),
                            html.Div(f"{confidence_score:.1f}", className="gauge-score", style={"color": palette["color"]}),
                            html.Div("annotation evidence score", className="gauge-sublabel"),
                            html.Div(
                                confidence_label,
                                className="gauge-badge",
                                style={
                                    "color": palette["color"],
                                    "background": palette["badge_background"],
                                    "borderColor": palette["badge_border"],
                                },
                            ),
                            html.Div(palette["tone"], className="gauge-reliability", style={"color": palette["color"]}),
                            html.Div(className="gauge-divider"),
                            html.Div(top_compound, className="compound-name"),
                            html.Div(
                                f"Representative structure key: {_truncate_text(structure_key, max_length=42)}",
                                className="gauge-context",
                            ),
                        ],
                        className="gauge-card",
                    ),
                    html.Div(
                        html.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Metric"),
                                            html.Th("Weight"),
                                            html.Th("Normalised value"),
                                            html.Th("Raw evidence"),
                                            html.Th("Contribution"),
                                        ]
                                    )
                                ),
                                html.Tbody(metric_rows),
                            ]
                        ),
                        className="table-card",
                    ),
                ],
                className="score-layout",
            ),
            html.Div(
                [
                    html.Span(
                        [
                            html.Span(className="legend-dot reliable-dot"),
                            "Well-supported: score \u2265 80",
                        ]
                    ),
                    html.Span(
                        [
                            html.Span(className="legend-dot moderate-dot"),
                            "Ambiguous: 55 to 79.9",
                        ]
                    ),
                    html.Span(
                        [
                            html.Span(className="legend-dot low-dot"),
                            "Insufficient evidence: score < 55",
                        ]
                    ),
                    html.Span(
                        "Spectral similarity uses spectral entropy when available, otherwise mean cosine. "
                        "Tanimoto uses ECFP4 mean pairwise similarity when ≥ 2 unique SMILES are present, "
                        "otherwise falls back to the hit-count structure agreement fraction.",
                        className="legend-note",
                    ),
                ],
                className="legend",
            ),
            html.Div(
                html.A(
                    "Back to scan table",
                    href=(
                        f"/dashinterface2/?task={urllib.parse.quote(str(task_id))}"
                        f"&workflow={urllib.parse.quote(str(workflow_choice))}"
                        f"&entropy={'1' if use_entropy_similarity else '0'}"
                    ),
                    className="detail-back-link",
                ),
                className="detail-actions",
            ),
        ],
        className="msms-detail mb-3",
    )


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


def _build_mirror_plot_link(query_usi, library_usi):
    query_usi = _to_text(query_usi)
    library_usi = _normalize_library_usi(library_usi)
    if not query_usi or not library_usi:
        return ""

    params = urllib.parse.urlencode({"usi1": query_usi, "usi2": library_usi})
    return f"https://metabolomics-usi.gnps2.org/dashinterface/?{params}"


def _build_detail_panel(task_id, scan, workflow_choice, use_entropy_similarity=True):
    matches_df = _load_library_matches(task_id, workflow_choice)
    resolved_workflow = _infer_resolved_workflow(matches_df, workflow_choice)
    summary_df = _compute_scan_level_confidence(
        matches_df,
        task_id=task_id,
        workflow_choice=resolved_workflow,
        scan_filter=scan,
        use_entropy_similarity=use_entropy_similarity,
    )

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
    library_name_col = _find_first_column(scan_df, ["LibraryName", "library_name", "Library Name"])
    spectrum_file_col = _find_first_column(scan_df, ["SpectrumFile"])
    smiles_col = _find_first_column(scan_df, ["Smiles", "SMILES", "smiles"])
    inchikey_col = _find_first_column(scan_df, ["InChIKey", "inchikey", "INCHIKEY"])
    inchi_col = _find_first_column(scan_df, ["INCHI", "InChI", "inchi"])

    spectrum_file = ""
    if spectrum_file_col is not None and len(scan_df) > 0:
        spectrum_file = _to_text(scan_df.iloc[0][spectrum_file_col])

    query_usi = _build_query_usi(task_id, scan, resolved_workflow, spectrum_file=spectrum_file)
    scan_df["_cosine"] = pd.to_numeric(scan_df[cosine_col], errors="coerce").fillna(0.0) if cosine_col else 0.0
    scan_df["_compound"] = scan_df[compound_col].astype(str) if compound_col else "Unknown"
    scan_df["_library_usi"] = scan_df[library_usi_col].astype(str) if library_usi_col else ""
    scan_df["_library_name"] = scan_df[library_name_col].astype(str) if library_name_col else ""
    scan_df["_smiles"] = scan_df[smiles_col].astype(str) if smiles_col else ""
    scan_df["_inchikey"] = scan_df[inchikey_col].astype(str) if inchikey_col else ""
    scan_df["_inchi"] = scan_df[inchi_col].astype(str) if inchi_col else ""
    scan_df["_structure_key"] = scan_df.apply(_canonical_structure_key, axis=1)
    scan_df = _annotate_support_groups(scan_df)

    detail_rows = (
        scan_df.sort_values("_cosine", ascending=False)
        .drop_duplicates(subset=["_compound", "_library_usi", "_smiles", "_inchikey"])
        .head(25)
    )

    metric_card = _build_reliability_breakdown_panel(
        task_id,
        scan,
        workflow_choice,
        resolved_workflow,
        score_row,
        use_entropy_similarity,
    )

    cards = []
    for _, row in detail_rows.iterrows():
        compound = _to_text(row.get("_compound")) or "Unknown Compound"
        lib_usi = _to_text(row.get("_library_usi"))
        lib_name = _to_text(row.get("_library_name"))
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
                f"Library: {lib_name}" if lib_name else "Library name unavailable",
                className="text-muted",
            ),
            html.Br(),
            html.Small(
                f"Library USI: {normalized_library_usi}" if normalized_library_usi else "Library USI unavailable",
                className="text-muted",
            ),
            html.Br(),
            html.Small(f"InChIKey: {inchikey}" if (inchikey != "None" and inchikey)  else "InChIKey unavailable", className="text-muted"),
            html.Br(),
            html.Small(
                "SMILES/InChI available" if (smiles or inchi) else "SMILES/InChI unavailable",
                className="text-muted",
            ),
            html.Hr(),
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


def _build_intro_panel():
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Start With a GNPS2 Task")),
            dbc.CardBody(
                [
                    html.P(
                        "Enter a completed GNPS2 task ID above to load library matches and score each scan.",
                        className="mb-2",
                    ),
                    html.Ul(
                        [
                            html.Li("Supported workflows: FBMN, Classical Networking, and Library Search."),
                            html.Li("Each scan is scored from structure agreement, Tanimoto structural similarity, spectral similarity, and best cosine score."),
                            html.Li("Open a proposed compound to inspect the detailed evidence and mirror-plot links."),
                        ],
                        className="mb-2",
                    ),
                    html.Small(
                        "If you start from the landing page, results open automatically after you submit a task ID.",
                        className="text-muted",
                    ),
                ]
            ),
        ]
    )
