# -*- coding: utf-8 -*-
import base64
from functools import lru_cache
from io import BytesIO
import math
import urllib.parse

import dash
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from gnpsdata import taskresult
from gnpsdata import workflow_classicnetworking
from gnpsdata import workflow_fbmn

from app import app, cache


dash_app = dash.Dash(
    name="dashinterface2",
    server=app,
    url_base_pathname="/dashinterface2/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

dash_app.title = "GNPS2 Library Confidence"


SUPPORTED_WORKFLOWS = {"auto", "classic", "fbmn", "librarysearch"}
ENTROPY_USI_TIMEOUT_SEC = 8
LIBRARY_RESULTS_CACHE_TIMEOUT_SEC = 86400


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
        "Library_Class": ["Library_Class", "library_class", "Library Class", "LibraryClass", "libraryclass"],
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


@lru_cache(maxsize=8192)
def _build_positional_isomer_group_key(inchi="", smiles=""):
    inchi = _to_text(inchi)
    smiles = _to_text(smiles)

    mol = None
    if inchi:
        try:
            mol = Chem.MolFromInchi(inchi)
        except Exception:
            mol = None
    if mol is None and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None

    if mol is not None:
        try:
            mol = Chem.Mol(mol)
            Chem.RemoveStereochemistry(mol)
            mol = Chem.RemoveHs(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = ""
            sidechain_fragments = []
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                replaced = Chem.ReplaceCore(mol, scaffold)
                if replaced is not None and replaced.GetNumAtoms() > 0:
                    for fragment in Chem.GetMolFrags(replaced, asMols=True, sanitizeFrags=False):
                        try:
                            Chem.SanitizeMol(fragment)
                        except Exception:
                            pass
                        sidechain_fragments.append(_canonicalize_fragment_smiles(fragment))

            if scaffold_smiles:
                fragment_key = "|".join(sorted(sidechain_fragments))
                return f"POSISO|{formula}|{scaffold_smiles}|{fragment_key}"
        except Exception:
            pass

    return ""


def _confidence_category(score):
    if score >= 80:
        return "Reliable", "success"
    if score >= 55:
        return "Less Reliable", "warning"
    return "Unreliable", "danger"


def _annotate_support_groups(df):
    work_df = df.copy()
    work_df["_positional_group_key"] = work_df.apply(
        lambda row: _build_positional_isomer_group_key(
            row.get("INCHI", ""),
            row.get("Smiles", ""),
        ),
        axis=1,
    )

    exact_to_positional = {}
    if "_structure_key" in work_df.columns:
        positional_rows = work_df[work_df["_positional_group_key"] != ""]
        if len(positional_rows) > 0:
            exact_to_positional = (
                positional_rows.groupby("_structure_key")["_positional_group_key"]
                .agg(lambda series: series.mode().iloc[0] if len(series.mode()) > 0 else series.iloc[0])
                .to_dict()
            )

    work_df["_support_group_key"] = work_df["_structure_key"].map(exact_to_positional)
    if "_positional_group_key" in work_df.columns:
        work_df["_support_group_key"] = work_df["_support_group_key"].fillna(work_df["_positional_group_key"])
    work_df["_support_group_key"] = work_df["_support_group_key"].fillna("")
    work_df["_support_group_key"] = work_df["_support_group_key"].replace("", pd.NA)
    work_df["_support_group_key"] = work_df["_support_group_key"].fillna(work_df["_structure_key"])

    return work_df


def _parse_library_class(value):
    text = _to_text(value)
    if not text:
        return None

    numeric_value = None
    try:
        numeric_value = int(float(text))
    except Exception:
        for character in text:
            if character in {"1", "2", "3"}:
                numeric_value = int(character)
                break

    if numeric_value in {1, 2, 3}:
        return numeric_value
    return None


def _library_class_to_quality_score(library_class_value):
    # GNPS library classes: 1 = highest confidence source, 3 = lower confidence source.
    class_quality_map = {
        1: 1.00,
        2: 0.65,
        3: 0.30,
    }
    if library_class_value in class_quality_map:
        return class_quality_map[library_class_value]
    # Neutral fallback when class is missing/unparseable.
    return 0.50


@lru_cache(maxsize=4096)
def _get_entropy_similarity_from_usis(query_usi, library_usi):
    query_usi = _to_text(query_usi)
    library_usi = _normalize_library_usi(library_usi)
    if not query_usi or not library_usi:
        return None

    try:
        from spectral_entropy_v2 import (
            DEFAULT_MAX_PEAK_NUM,
            DEFAULT_MS2_TOLERANCE_DA,
            DEFAULT_NOISE_THRESHOLD,
            entropy_similarity_pair,
            fetch_spectrum,
        )
    except BaseException:
        return None

    try:
        query_spec = fetch_spectrum(query_usi, timeout=ENTROPY_USI_TIMEOUT_SEC)
        library_spec = fetch_spectrum(library_usi, timeout=ENTROPY_USI_TIMEOUT_SEC)
        similarity_payload = entropy_similarity_pair(
            query_spec.get("peaks", []),
            library_spec.get("peaks", []),
            ms2_tolerance_da=DEFAULT_MS2_TOLERANCE_DA,
            noise_threshold=DEFAULT_NOISE_THRESHOLD,
            max_peak_num=DEFAULT_MAX_PEAK_NUM,
        )
        entropy_similarity = similarity_payload.get("entropy_similarity")
        if entropy_similarity is None:
            return None
        entropy_similarity = float(entropy_similarity)
        return max(0.0, min(1.0, entropy_similarity))
    except Exception:
        return None


@lru_cache(maxsize=128)
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


@cache.memoize(timeout=LIBRARY_RESULTS_CACHE_TIMEOUT_SEC)
def _load_library_matches_cached(task_id, workflow_choice):
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


def _load_library_matches(task_id, workflow_choice):
    task_id = (task_id or "").strip()
    workflow_choice = (workflow_choice or "auto").strip().lower() or "auto"
    matches_df = _load_library_matches_cached(task_id, workflow_choice)
    if matches_df is None:
        return matches_df
    return matches_df.copy()


def _compute_scan_level_confidence(
    df,
    task_id="",
    workflow_choice="auto",
    scan_filter=None,
    use_entropy_similarity=True,
):
    scan_col = _find_first_column(df, ["#Scan#", "scan", "Scan"])
    cosine_col = _find_first_column(df, ["MQScore", "Cosine", "cosine"])
    compound_col = _find_first_column(df, ["Compound_Name", "compound_name", "Compound Name"])
    library_usi_col = _find_first_column(df, ["library_usi", "LibraryUSI", "library usi"])
    spectrum_file_col = _find_first_column(df, ["SpectrumFile"])
    library_class_col = _find_first_column(
        df,
        ["Library_Class", "library_class", "Library Class", "LibraryClass", "libraryclass"],
    )

    if scan_col is None:
        raise ValueError("Could not find scan column (#Scan# / scan / Scan) in library results.")
    if cosine_col is None:
        raise ValueError("Could not find cosine score column (MQScore / Cosine).")

    work_df = df.copy()
    work_df[scan_col] = work_df[scan_col].astype(str)
    if scan_filter is not None:
        work_df = work_df[work_df[scan_col] == str(scan_filter)].copy()

    work_df["_cosine"] = pd.to_numeric(work_df[cosine_col], errors="coerce").fillna(0.0)
    work_df["_structure_key"] = work_df.apply(_canonical_structure_key, axis=1)
    work_df = _annotate_support_groups(work_df)
    work_df["_compound"] = (
        work_df[compound_col].astype(str)
        if compound_col is not None
        else "Unknown"
    )
    work_df["_library_usi"] = (
        work_df[library_usi_col].astype(str)
        if library_usi_col is not None
        else ""
    )
    work_df["_spectrum_file"] = (
        work_df[spectrum_file_col].astype(str)
        if spectrum_file_col is not None
        else ""
    )
    work_df["_library_class"] = (
        work_df[library_class_col].apply(_parse_library_class)
        if library_class_col is not None
        else None
    )
    work_df["_library_class_quality"] = work_df["_library_class"].apply(_library_class_to_quality_score)
    resolved_workflow = _infer_resolved_workflow(work_df, workflow_choice)

    summary_rows = []
    for scan, scan_df in work_df.groupby(scan_col):
        total_matches = len(scan_df)

        grouped = (
            scan_df.groupby("_support_group_key", as_index=False)
            .agg(
                count=("_support_group_key", "size"),
                mean_cosine=("_cosine", "mean"),
                max_cosine=("_cosine", "max"),
            )
            .sort_values(["count", "mean_cosine", "max_cosine"], ascending=False)
        )

        best = grouped.iloc[0]
        support_group_key = best["_support_group_key"]
        structure_hits = scan_df[scan_df["_support_group_key"] == support_group_key]
        representative_row = structure_hits.sort_values("_cosine", ascending=False).iloc[0]
        structure_key = representative_row.get("_structure_key", "UNKNOWN_STRUCTURE")
        top_compound = structure_hits["_compound"].mode().iloc[0]

        support_fraction = best["count"] / total_matches if total_matches else 0.0
        mean_cosine = float(best["mean_cosine"])
        max_cosine = float(best["max_cosine"])
        class_quality = float(structure_hits["_library_class_quality"].mean()) if len(structure_hits) else 0.5
        class_counts = structure_hits["_library_class"].value_counts(dropna=True).to_dict()
        class_1_hits = int(class_counts.get(1, 0))
        class_2_hits = int(class_counts.get(2, 0))
        class_3_hits = int(class_counts.get(3, 0))
        entropy_similarity = None
        entropy_source = "disabled"
        if use_entropy_similarity:
            representative_library_usi = _normalize_library_usi(representative_row.get("_library_usi"))
            representative_spectrum_file = _to_text(representative_row.get("_spectrum_file"))
            query_usi = _build_query_usi(task_id, scan, resolved_workflow, spectrum_file=representative_spectrum_file)
            entropy_similarity = _get_entropy_similarity_from_usis(query_usi, representative_library_usi)
            entropy_source = "spectral_entropy"
        if entropy_similarity is None:
            entropy_similarity = mean_cosine
            if use_entropy_similarity:
                entropy_source = "mean_cosine_fallback"

        confidence_score = 100.0 * (
            0.50 * support_fraction
            + 0.25 * entropy_similarity
            + 0.10 * max_cosine
            + 0.15 * class_quality
        )
        confidence_label, badge_color = _confidence_category(confidence_score)

        summary_rows.append(
            {
                "scan": scan,
                "top_compound": top_compound,
                "structure_key": structure_key,
                "support_group_key": support_group_key,
                "supporting_matches": int(best["count"]),
                "total_matches": int(total_matches),
                "mean_cosine": round(mean_cosine, 4),
                "max_cosine": round(max_cosine, 4),
                "entropy_similarity": round(float(entropy_similarity), 4),
                "entropy_source": entropy_source,
                "library_class_quality": round(class_quality, 4),
                "class_1_hits": class_1_hits,
                "class_2_hits": class_2_hits,
                "class_3_hits": class_3_hits,
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


def _truncate_text(text, max_length=36):
    text = _to_text(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


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


def _metric_bar_color(value):
    if value >= 0.75:
        return "#1d9a6c"
    if value >= 0.45:
        return "#d48a18"
    return "#d35d47"


def _confidence_palette(confidence_label):
    confidence_label = _to_text(confidence_label)
    palette_map = {
        "Reliable": {
            "color": "#1d9a6c",
            "badge_background": "#dff4ea",
            "badge_border": "#8dd5b7",
            "tone": "Strong supporting evidence",
        },
        "Less Reliable": {
            "color": "#d48a18",
            "badge_background": "#fff1d6",
            "badge_border": "#f0cb7e",
            "tone": "Mixed supporting evidence",
        },
        "Unreliable": {
            "color": "#d35d47",
            "badge_background": "#fde2de",
            "badge_border": "#efb0a5",
            "tone": "Weak supporting evidence",
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


def _canonicalize_fragment_smiles(fragment):
    fragment = Chem.Mol(fragment)
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(fragment, canonical=True)


def _build_reliability_gauge_data_uri(score, color):
    score_fraction = max(0.0, min(1.0, float(score) / 100.0))
    cx = 70
    cy = 78
    radius = 52
    start_deg = 210
    sweep_deg = 240
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
    detail_rows,
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
    class_quality = float(score_row["library_class_quality"])
    max_cosine = float(score_row["max_cosine"])
    mean_cosine = float(score_row["mean_cosine"])
    class_1_hits = int(score_row["class_1_hits"])
    class_2_hits = int(score_row["class_2_hits"])
    class_3_hits = int(score_row["class_3_hits"])
    top_compound = _to_text(score_row["top_compound"]) or "Unknown Compound"
    structure_key = _to_text(score_row["structure_key"])
    support_group_key = _to_text(score_row.get("support_group_key"))
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
        {
            "label": "Library class quality",
            "formula": "mean mapped class score × 15%",
            "weight": 0.15,
            "value": class_quality,
            "raw_primary": f"{class_quality:.4f} quality score",
            "raw_secondary": f"C1 {class_1_hits} · C2 {class_2_hits} · C3 {class_3_hits}",
        },
    ]

    metadata_fields = [
        ("Task ID", task_id),
        ("Scan", scan),
        ("Workflow", _format_workflow_label(resolved_workflow or workflow_choice)),
        ("Similarity source", _format_entropy_source_label(entropy_source)),
        ("Structure support", f"{supporting_matches} / {total_matches}"),
        ("Class profile", f"C1 {class_1_hits} · C2 {class_2_hits} · C3 {class_3_hits}"),
    ]

    annotation_pills = []
    seen_compounds = set()
    top_compound_included = False
    for rank, row in enumerate(detail_rows.to_dict("records"), start=1):
        compound = _to_text(row.get("_compound")) or "Unknown Compound"
        if compound in seen_compounds:
            continue
        seen_compounds.add(compound)
        is_primary = compound == top_compound
        supports_consensus = _to_text(row.get("_support_group_key")) == support_group_key
        pill_class = "tab"
        if is_primary:
            pill_class += " active"
            top_compound_included = True
        elif supports_consensus:
            pill_class += " supporting"
        annotation_pills.append(
            html.Span(
                f"Rank #{rank} · {_truncate_text(compound, max_length=30)}",
                className=pill_class,
                title=compound,
            )
        )
        if len(annotation_pills) >= 6:
            break

    if not top_compound_included:
        annotation_pills.insert(
            0,
            html.Span(
                f"Consensus · {_truncate_text(top_compound, max_length=30)}",
                className="tab active",
                title=top_compound,
            ),
        )

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
                html.Td(html.Span("Composite reliability score", className="total-label")),
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
            html.Div(annotation_pills, className="tabs"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(src=gauge_src, className="gauge-svg", alt="Reliability score gauge"),
                            html.Div(f"{confidence_score:.1f}", className="gauge-score", style={"color": palette["color"]}),
                            html.Div("composite reliability score", className="gauge-sublabel"),
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
                            "Reliable: score >= 80",
                        ]
                    ),
                    html.Span(
                        [
                            html.Span(className="legend-dot moderate-dot"),
                            "Less Reliable: 55 to 79.9",
                        ]
                    ),
                    html.Span(
                        [
                            html.Span(className="legend-dot low-dot"),
                            "Unreliable: score < 55",
                        ]
                    ),
                    html.Span(
                        "Similarity evidence uses spectral entropy when available; otherwise mean cosine is used.",
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
        detail_rows,
        use_entropy_similarity,
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
                            html.Li("Each scan is scored from structure agreement, spectral similarity, and library class quality."),
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
