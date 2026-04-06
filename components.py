# -*- coding: utf-8 -*-
import base64
import json
import logging
import math
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from io import BytesIO

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
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
from bin.scorer import _compute_scan_level_confidence, _confidence_category
from bin.tanimoto import _smiles_to_morgan_fp, scan_tanimoto

logger = logging.getLogger(__name__)


def _search_pubchem_candidates(inchikey: str, name: str, max_results: int = 8) -> list:
    """
    Search PubChem for compound candidates.

    Tries InChIKey first (exact match). If that fails, performs a fuzzy
    name lookup by combining PubChem autocomplete suggestions with
    name-based property queries.
    Returns list of {"cid": int|str, "iupac_name": str, "smiles": str, "source": str, "matched_name": str}.
    """

    invalid_names = {"", "unknown", "unknown compound", "n/a", "none", "nan"}

    def _fetch_json(url: str) -> dict:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TopKonfidence/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read())
        except Exception:
            logger.debug("PubChem fetch failed for URL %s", url)
            return {}

    def _fetch_properties(url_suffix: str, source_label: str, matched_name: str = "") -> list:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            + url_suffix
            + "/property/IsomericSMILES,SMILES,IUPACName/JSON"
        )
        data = _fetch_json(url)
        return [
            {
                "cid": prop.get("CID", "?"),
                "iupac_name": prop.get("IUPACName") or f"CID {prop.get('CID', '?')}",
                "smiles": prop.get("IsomericSMILES") or prop.get("SMILES") or "",
                "source": source_label,
                "matched_name": matched_name,
            }
            for prop in data.get("PropertyTable", {}).get("Properties", [])
        ]

    def _fetch_autocomplete_names(query: str, limit: int = 8) -> list:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/"
            + urllib.parse.quote(query)
            + f"/JSON?limit={int(limit)}"
        )
        data = _fetch_json(url)
        terms = data.get("dictionary_terms", {}).get("compound", [])
        if not isinstance(terms, list):
            return []

        cleaned = [str(term).strip() for term in terms if str(term).strip()]
        cleaned.sort(
            key=lambda term: SequenceMatcher(None, query.lower(), term.lower()).ratio(),
            reverse=True,
        )
        return cleaned[:limit]

    def _dedupe_candidates(candidates: list) -> list:
        seen_keys: set = set()
        deduped = []
        for candidate in candidates:
            cid = str(candidate.get("cid", "")).strip()
            key = cid if cid else (candidate.get("smiles", ""), candidate.get("iupac_name", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(candidate)
            if len(deduped) >= max_results:
                break
        return deduped

    results: list = []

    inchikey = (inchikey or "").strip()
    if inchikey and inchikey.lower() not in {"none", "nan"}:
        results = _fetch_properties("inchikey/" + urllib.parse.quote(inchikey), "InChIKey")

    if not results:
        name = (name or "").strip()
        if name and name.lower() not in invalid_names:
            query_names = [name]
            for candidate_name in _fetch_autocomplete_names(name, limit=max_results):
                if candidate_name.lower() != name.lower():
                    query_names.append(candidate_name)

            for candidate_name in query_names:
                source = "name" if candidate_name == name else "name-fuzzy"
                results.extend(
                    _fetch_properties(
                        "name/" + urllib.parse.quote(candidate_name),
                        source,
                        matched_name=candidate_name,
                    )
                )
                if len(_dedupe_candidates(results)) >= max_results:
                    break

    return _dedupe_candidates(results)


def _fetch_pubchem_smiles(compound_name: str) -> str:
    """Fetch the first available SMILES from PubChem by compound name. Returns '' on failure."""
    candidates = _search_pubchem_candidates("", compound_name, max_results=1)
    return candidates[0]["smiles"] if candidates else ""


def _build_tanimoto_heatmap_figure(compounds_data: list, use_pubchem: bool = False):
    """
    Build a Plotly lower-triangle heatmap of pairwise Tanimoto (ECFP4) similarity.

    Parameters
    ----------
    compounds_data : list of {"name": str, "smiles": str}
    use_pubchem    : if True, fetch SMILES from PubChem for compounds that lack one.

    Returns a plotly go.Figure, or None if RDKit/Plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go
        from rdkit.Chem import DataStructs
    except ImportError:
        return None

    compounds = [dict(c) for c in compounds_data]  # shallow copy

    for c in compounds:
        smiles = c.get("smiles") or ""
        if not smiles and use_pubchem:
            fetched = _fetch_pubchem_smiles(c["name"])
            c["smiles"] = fetched
            c["smiles_source"] = "pubchem" if fetched else "unavailable"
        elif smiles:
            c["smiles_source"] = "library"
        else:
            c["smiles_source"] = "unavailable"

    fps = [_smiles_to_morgan_fp(c["smiles"]) if c.get("smiles") else None for c in compounds]

    n = len(compounds)
    labels = [_truncate_text(c["name"], max_length=28) for c in compounds]
    sources = [c.get("smiles_source", "unavailable") for c in compounds]

    # Build lower-triangle matrix (i >= j); upper triangle stays None → blank
    z = [[None] * n for _ in range(n)]
    hover_text = [[""] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            li, lj = labels[i], labels[j]
            if i == j:
                z[i][j] = 1.0
                hover_text[i][j] = f"<b>{li}</b><br>Self: 1.0000"
            elif fps[i] is not None and fps[j] is not None:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                z[i][j] = round(sim, 4)
                hover_text[i][j] = (
                    f"<b>{li}</b> vs <b>{lj}</b><br>"
                    f"Tanimoto: {sim:.4f}<br>"
                    f"SMILES sources: {sources[i]} / {sources[j]}"
                )
            else:
                z[i][j] = None
                missing = [lbl for lbl, fp in ((li, fps[i]), (lj, fps[j])) if fp is None]
                hover_text[i][j] = (
                    f"<b>{li}</b> vs <b>{lj}</b><br>"
                    f"N/A — no SMILES for: {', '.join(missing)}"
                )

    colorscale = [[0.0, "#f80000"], [0.5, "#d48a18"], [1.0, "#1d9a6c"]]
    cell_px = 38
    height = max(320, n * cell_px + 150)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Tanimoto", thickness=14, len=0.7),
            showscale=True,
        )
    )
    fig.update_layout(
        title=dict(text="Pairwise Tanimoto Similarity (ECFP4)", font=dict(size=13)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=11), side="bottom"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


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
        "tanimoto_ecfp4_pubchem": "Tanimoto (ECFP4, mean pairwise; SMILES via PubChem)",
        "single_structure": "Tanimoto (single structure — perfect consensus)",
        "single_structure_pubchem": "Tanimoto (single structure — perfect consensus; SMILES via PubChem)",
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
        "Consistent evidence": {
            "color": "#1d9a6c",
            "badge_background": "#dff4ea",
            "badge_border": "#8dd5b7",
            "tone": "Strong structural evidence",
        },
        "Inconclusive": {
            "color": "#d48a18",
            "badge_background": "#fff1d6",
            "badge_border": "#f0cb7e",
            "tone": "Inconclusive evidence",
        },
        "Inconsistent evidence": {
            "color": "#d35d47",
            "badge_background": "#fde2de",
            "badge_border": "#efb0a5",
            "tone": "Inconsistent evidence across matches",
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
        else "Mean pairwise Tanimoto (ECFP4, r=2, 2048-bit); missing SMILES resolved via PubChem"
        if tanimoto_source == "tanimoto_ecfp4_pubchem"
        else "All hits agree on one structure — no pairs to compare"
        if tanimoto_source == "single_structure"
        else "Single structure (SMILES resolved via PubChem) — perfect consensus"
        if tanimoto_source == "single_structure_pubchem"
        else "Fell back to hit-count structure agreement (fewer than 2 unique valid SMILES)"
        if tanimoto_source == "structure_agreement_fallback"
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
                            "Consistent evidence: score \u2265 80",
                        ]
                    ),
                    html.Span(
                        [
                            html.Span(className="legend-dot moderate-dot"),
                            "Inconclusive: 55 to 79.9",
                        ]
                    ),
                    html.Span(
                        [
                            html.Span(className="legend-dot low-dot"),
                            "Inconsistent evidence: score < 55",
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


MASST_API_URL = "https://masst-records.gnps2.org/masst_records.json?sql="


def _fetch_masst_context(inchikey_first_blocks: list) -> pd.DataFrame:
    """
    Query the MASST records API for biological context of the given InChIKey first blocks.

    Joins masst_table → library_table → redu_table to retrieve SampleType and
    UBERONBodyPartName counts. Returns an empty DataFrame on failure.
    """
    blocks = [b for b in inchikey_first_blocks if b and len(b) >= 5]
    if not blocks:
        return pd.DataFrame(), ""

    in_clause = ", ".join(f"'{b}'" for b in blocks)
    sql = f"""
WITH filtered AS (
  SELECT m.mri_id_int, m.spectrum_id_int
  FROM masst_table m
  WHERE m.spectrum_id_int IN (
    SELECT spectrum_id_int FROM library_table
    WHERE InChIKey_smiles_firstBlock IN ({in_clause})
  )
),
best_per_mri AS (
  SELECT mri_id_int, COUNT(DISTINCT spectrum_id_int) AS unique_spectra
  FROM filtered GROUP BY mri_id_int
)
SELECT rd.SampleType, rd.UBERONBodyPartName,
       COUNT(*) AS num_samples,
       SUM(b.unique_spectra) AS total_spectra
FROM best_per_mri b
JOIN redu_table rd ON b.mri_id_int = rd.mri_id_int
WHERE rd.SampleType IS NOT NULL AND rd.SampleType != ''
GROUP BY rd.SampleType, rd.UBERONBodyPartName
ORDER BY num_samples DESC
LIMIT 100
""".strip()

    url = MASST_API_URL + urllib.parse.quote(sql)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TopKonfidence/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        if not isinstance(data, dict) or not data.get("ok") or not data.get("rows"):
            return pd.DataFrame(), sql
        columns = data["columns"]
        rows = data["rows"]
        return pd.DataFrame(rows, columns=columns), sql
    except Exception:
        logger.debug("MASST context fetch failed for InChIKeys %s", blocks)
        return pd.DataFrame(), sql


def _build_masst_context_card(inchikey_first_blocks: list):
    """
    Build a Dash card showing biological context (SampleType + body part) from MASST/REDU.
    Returns None if no data is available.
    """
    df, sql = _fetch_masst_context(inchikey_first_blocks)

    sql_debug = html.Details(
        [
            html.Summary("SQL query", style={"cursor": "pointer", "fontSize": "0.8rem"}),
            html.Pre(sql, style={"fontSize": "0.75rem", "whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
        ],
        className="mt-2",
    )

    if df.empty:
        return dbc.Card(
            [
                dbc.CardHeader(html.H5("Biological Context (MASST/REDU)")),
                dbc.CardBody([
                    html.Div(
                        "No MASST/REDU biological context found for these structures.",
                        className="text-muted small",
                    ),
                    sql_debug,
                ]),
            ],
            className="mt-3",
        )

    # Aggregate by SampleType for the table
    sample_type_counts = (
        df.groupby("SampleType", as_index=False)["num_samples"]
        .sum()
        .sort_values("num_samples", ascending=False)
        .head(15)
    )

    sample_type_table_rows = [
        html.Tr([
            html.Th("Sample Type", style={"width": "70%"}),
            html.Th("Samples", style={"width": "30%", "textAlign": "right"}),
        ])
    ] + [
        html.Tr([
            html.Td(row["SampleType"]),
            html.Td(str(int(row["num_samples"])), style={"textAlign": "right"}),
        ])
        for _, row in sample_type_counts.iterrows()
    ]

    # Top body parts table (non-empty UBERONBodyPartName)
    body_part_df = df[df["UBERONBodyPartName"].notna() & (df["UBERONBodyPartName"] != "")]
    body_part_counts = (
        body_part_df.groupby("UBERONBodyPartName", as_index=False)["num_samples"]
        .sum()
        .sort_values("num_samples", ascending=False)
        .head(10)
    )

    table_rows = [
        html.Tr([
            html.Th("Body Part", style={"width": "70%"}),
            html.Th("Samples", style={"width": "30%", "textAlign": "right"}),
        ])
    ] + [
        html.Tr([
            html.Td(row["UBERONBodyPartName"]),
            html.Td(str(int(row["num_samples"])), style={"textAlign": "right"}),
        ])
        for _, row in body_part_counts.iterrows()
    ]

    total_samples = int(df["num_samples"].sum())
    total_spectra = int(df["total_spectra"].sum()) if "total_spectra" in df.columns else 0
    summary = html.Small(
        f"Found in {total_samples:,} REDU samples across {total_spectra:,} spectra.",
        className="text-muted d-block mb-2",
    )

    body_children = [summary]
    if not sample_type_counts.empty:
        body_children.append(html.H6("Top sample types", className="mb-1"))
        body_children.append(
            dbc.Table(
                html.Tbody(sample_type_table_rows),
                bordered=False,
                striped=True,
                size="sm",
                className="mb-0",
            )
        )
    if not body_part_counts.empty:
        body_children.append(html.H6("Top body parts", className="mt-3 mb-1"))
        body_children.append(
            dbc.Table(
                html.Tbody(table_rows),
                bordered=False,
                striped=True,
                size="sm",
                className="mb-0",
            )
        )

    inchikey_labels = html.Small(
        "Queried InChIKey first blocks: " + ", ".join(inchikey_first_blocks),
        className="text-muted d-block mt-2",
    )
    body_children.append(inchikey_labels)
    body_children.append(sql_debug)

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Biological Context (MASST/REDU)")),
            dbc.CardBody(body_children),
        ],
        className="mt-3",
    )


def _build_detail_panel(
    task_id,
    scan,
    workflow_choice,
    use_entropy_similarity=True,
    pubchem_overrides=None,
    use_pubchem_for_score=False,
):
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
    scan_df["_smiles"] = scan_df["_smiles"].apply(
        lambda v: "" if _to_text(v).strip().lower() in {"none", "nan"} else _to_text(v).strip()
    )
    scan_df["_inchikey"] = scan_df["_inchikey"].apply(
        lambda v: "" if _to_text(v).strip().lower() in {"none", "nan"} else _to_text(v).strip()
    )
    scan_df["_inchi"] = scan_df["_inchi"].apply(
        lambda v: "" if _to_text(v).strip().lower() in {"none", "nan"} else _to_text(v).strip()
    )
    scan_df["_structure_key"] = scan_df.apply(_canonical_structure_key, axis=1)
    scan_df = _annotate_support_groups(scan_df)

    def _detail_group_key(row):
        structure_key = _to_text(row.get("_structure_key"))
        if structure_key and structure_key != "UNKNOWN_STRUCTURE":
            return f"structure::{structure_key}"
        fallback_parts = [
            _to_text(row.get("_compound")),
            _to_text(row.get("_library_usi")),
            _to_text(row.get("_smiles")),
            _to_text(row.get("_inchikey")),
            _to_text(row.get("_inchi")),
        ]
        return "fallback::" + "||".join(fallback_parts)

    def _ordered_unique_texts(values):
        ordered = []
        seen = set()
        for value in values:
            text = _to_text(value)
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    detail_rows = scan_df.sort_values("_cosine", ascending=False).copy()
    detail_rows["_detail_group_key"] = detail_rows.apply(_detail_group_key, axis=1)

    table_rows = []
    compounds_data: list = []
    card_signatures: list = []
    overrides = dict(pubchem_overrides or {}) if isinstance(pubchem_overrides, dict) else {}

    grouped_detail_rows = []
    for _, group_df in detail_rows.groupby("_detail_group_key", sort=False):
        grouped_detail_rows.append(group_df.sort_values("_cosine", ascending=False))
        if len(grouped_detail_rows) >= 25:
            break

    max_group_matches = max((len(group_df) for group_df in grouped_detail_rows), default=0)
    max_group_cosine = max((float(group_df["_cosine"].max()) for group_df in grouped_detail_rows), default=0.0)

    for card_idx, group_df in enumerate(grouped_detail_rows):
        compound_names = _ordered_unique_texts(group_df["_compound"].tolist())
        library_names = _ordered_unique_texts(group_df["_library_name"].tolist())
        library_usis = _ordered_unique_texts(group_df["_library_usi"].tolist())
        smiles_candidates = _ordered_unique_texts(group_df["_smiles"].tolist())
        inchikey_candidates = _ordered_unique_texts(group_df["_inchikey"].tolist())
        inchi_candidates = _ordered_unique_texts(group_df["_inchi"].tolist())

        compound = compound_names[0] if compound_names else "Unknown Compound"
        smiles = smiles_candidates[0] if smiles_candidates else ""
        inchikey = inchikey_candidates[0] if inchikey_candidates else ""
        inchi = inchi_candidates[0] if inchi_candidates else ""
        cosine = float(group_df["_cosine"].max()) if len(group_df) else 0.0
        image_url = _render_mol_png_data_uri(smiles=smiles, inchi=inchi)
        normalized_library_usis = [_normalize_library_usi(usi) for usi in library_usis]
        normalized_library_usis = [usi for usi in normalized_library_usis if usi]

        mirror_links = []
        for link_idx, normalized_library_usi in enumerate(normalized_library_usis[:5]):
            mirror_links.append(
                html.A(
                    f"Mirror plot {link_idx + 1}",
                    href=_build_mirror_plot_link(query_usi, normalized_library_usi),
                    target="_blank",
                    className="d-inline-block me-2 mb-1",
                )
            )
        if len(normalized_library_usis) > 5:
            mirror_links.append(
                html.Small(
                    f"+{len(normalized_library_usis) - 5} more library match(es)",
                    className="text-muted d-block",
                )
            )

        can_search_pubchem = not smiles

        if can_search_pubchem:
            preview_children = (
                html.Img(
                    src=image_url,
                    style={"width": "180px", "height": "180px", "objectFit": "contain"},
                )
                if image_url
                else html.Small(
                    "No structure image yet. Select a PubChem match to display one.",
                    className="text-muted d-block",
                )
            )

            structure_preview = html.Div(
                id={"type": "pubchem-structure-preview", "index": card_idx},
                children=preview_children,
                className="mb-0",
            )

            pubchem_section = html.Div(
                [
                    dcc.Store(
                        id={"type": "compound-info-store", "index": card_idx},
                        data={"name": compound, "inchikey": inchikey, "smiles": smiles, "inchi": inchi},
                    ),
                    html.Small("PubChem structure lookup", className="text-muted fw-semibold d-block mb-1"),
                    dbc.Input(
                        id={"type": "pubchem-name-input", "index": card_idx},
                        type="text",
                        placeholder="Type a compound name to fuzzy-search PubChem",
                        className="mb-2",
                    ),
                    dbc.Button(
                        "Search PubChem for structure",
                        id={"type": "pubchem-search-btn", "index": card_idx},
                        size="sm",
                        color="info",
                        outline=True,
                        n_clicks=0,
                    ),
                    html.Div(id={"type": "pubchem-candidates-div", "index": card_idx}, className="mt-2"),
                ]
            )
        else:
            structure_preview = html.Div(
                html.Img(
                    src=image_url,
                    style={"width": "180px", "height": "180px", "objectFit": "contain"},
                )
                if image_url
                else html.Small("No structure image available.", className="text-muted d-block"),
                id={"type": "pubchem-structure-preview", "index": card_idx},
                className="mb-0",
            )
            pubchem_section = html.Div(
                [
                    dcc.Store(
                        id={"type": "compound-info-store", "index": card_idx},
                        data={"name": compound, "inchikey": inchikey, "smiles": smiles, "inchi": inchi},
                    ),
                    html.Div(id={"type": "pubchem-candidates-div", "index": card_idx}, className="d-none"),
                ]
            )

        alias_note = ""
        if len(compound_names) > 1:
            alias_note = ", ".join(compound_names[1:4])
            if len(compound_names) > 4:
                alias_note += f", +{len(compound_names) - 4} more"

        libraries_note = ", ".join(library_names[:4]) if library_names else ""
        if len(library_names) > 4:
            libraries_note += f", +{len(library_names) - 4} more"

        structure_text = smiles if smiles else inchi if inchi else ""
        if len(structure_text) > 90:
            structure_text = structure_text[:87].rstrip() + "..."

        summary_cell = html.Div(
            [
                html.Div(
                    [
                        html.Strong(compound),
                        dbc.Badge(f"{len(group_df)} match(es)", color="secondary", className="ms-2"),
                        dbc.Badge(
                            "Most matches",
                            color="success",
                            className="ms-2",
                        )
                        if len(group_df) == max_group_matches and max_group_matches > 0
                        else None,
                        dbc.Badge(
                            "Top cosine",
                            color="info",
                            className="ms-2",
                        )
                        if math.isclose(cosine, max_group_cosine, rel_tol=1e-9, abs_tol=1e-9)
                        else None,
                    ],
                    className="mb-1",
                ),
                html.Small(f"Best cosine: {cosine:.4f}", className="d-block"),
                html.Small(
                    f"InChIKey: {inchikey}" if inchikey else "InChIKey unavailable",
                    className="text-muted d-block",
                ),
                html.Small(
                    "SMILES/InChI available" if structure_text else "SMILES/InChI unavailable",
                    className="text-muted d-block mt-1",
                ),
                html.Small(structure_text, className="text-mono d-block"),
                html.Small(
                    f"Also annotated as: {alias_note}" if alias_note else "Single compound label in this group",
                    className="text-muted d-block mt-2",
                ),
                html.Small(
                    f"Libraries: {libraries_note}" if libraries_note else "Library name unavailable",
                    className="text-muted d-block",
                ),
            ]
        )

        action_cell_children = [
            html.Small(
                f"{len(normalized_library_usis)} unique library match(es)"
                if normalized_library_usis
                else "Library USI unavailable",
                className="text-muted d-block mb-1",
            ),
            html.Div(mirror_links if mirror_links else "Mirror plot links unavailable for this structure"),
        ]
        if can_search_pubchem:
            action_cell_children.extend([html.Hr(className="my-2"), pubchem_section])
        else:
            action_cell_children.append(html.Div(pubchem_section, className="d-none"))

        table_rows.append(
            html.Tr(
                [
                    html.Td(structure_preview, style={"width": "210px", "verticalAlign": "top"}),
                    html.Td(summary_cell, style={"verticalAlign": "top", "minWidth": "320px"}),
                    html.Td(html.Div(action_cell_children), style={"width": "320px", "verticalAlign": "top"}),
                ]
            )
        )

        card_signatures.append(
            {
                "card_idx": card_idx,
                "row_indices": group_df.index.tolist(),
            }
        )

        heatmap_name = compound
        if len(compound_names) > 1:
            heatmap_name = f"{compound} (+{len(compound_names) - 1})"
        compounds_data.append({"card_idx": card_idx, "name": heatmap_name, "smiles": smiles, "inchikey": inchikey})

    score_row_for_panel = score_row.copy()
    if use_pubchem_for_score and overrides:
        adjusted_scan_df = scan_df.copy()
        applied_overrides = 0

        for signature in card_signatures:
            override_smiles = _to_text(overrides.get(str(signature["card_idx"]))).strip()
            if not override_smiles:
                continue

            mask = adjusted_scan_df.index.isin(signature["row_indices"])
            if mask.any():
                adjusted_scan_df.loc[mask, "_smiles"] = override_smiles
                applied_overrides += 1

        if applied_overrides > 0:
            supporting_matches = int(score_row_for_panel["supporting_matches"])
            total_matches = int(score_row_for_panel["total_matches"])
            support_fraction = (supporting_matches / total_matches) if total_matches else 0.0

            tanimoto_score, tanimoto_source = scan_tanimoto(adjusted_scan_df["_smiles"].tolist(), support_fraction)
            if tanimoto_source == "tanimoto_ecfp4":
                tanimoto_source = "tanimoto_ecfp4_pubchem"
            elif tanimoto_source == "single_structure":
                tanimoto_source = "single_structure_pubchem"

            entropy_similarity = float(score_row_for_panel["entropy_similarity"])
            max_cosine = float(score_row_for_panel["max_cosine"])
            confidence_score = 100.0 * (
                0.50 * support_fraction
                + 0.15 * float(tanimoto_score)
                + 0.25 * entropy_similarity
                + 0.10 * max_cosine
            )
            confidence_score = min(100.0, max(0.0, confidence_score))
            confidence_label, badge_color = _confidence_category(confidence_score)

            score_row_for_panel["tanimoto_score"] = round(float(tanimoto_score), 4)
            score_row_for_panel["tanimoto_source"] = tanimoto_source
            score_row_for_panel["confidence_score"] = round(float(confidence_score), 1)
            score_row_for_panel["confidence_label"] = confidence_label
            score_row_for_panel["badge_color"] = badge_color

    metric_card = _build_reliability_breakdown_panel(
        task_id,
        scan,
        workflow_choice,
        resolved_workflow,
        score_row_for_panel,
        use_entropy_similarity,
    )

    compounds_card = dbc.Card(
        [
            dbc.CardHeader(html.H5("Matched Compound Structures")),
            dbc.CardBody(
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Structure"),
                                    html.Th("Match Summary"),
                                    html.Th("Library Evidence / PubChem"),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    bordered=False,
                    hover=True,
                    responsive=True,
                    className="align-middle mb-0",
                )
                if table_rows
                else html.Div("No compounds to display.")
            ),
        ]
    )

    # Collect unique first blocks from all inchikeys in this scan's matches
    inchikey_first_blocks = sorted({
        ik.split("-")[0]
        for ik in scan_df["_inchikey"].tolist()
        if ik and "-" in ik and len(ik.split("-")[0]) >= 5
    })
    masst_card = _build_masst_context_card(inchikey_first_blocks)

    return html.Div([metric_card, compounds_card, masst_card]), compounds_data
