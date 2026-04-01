# -*- coding: utf-8 -*-
import logging
from functools import lru_cache

import pandas as pd

from .constants import ENTROPY_USI_TIMEOUT_SEC
from .normalizer import (
    _annotate_support_groups,
    _build_query_usi,
    _canonical_structure_key,
    _find_first_column,
    _infer_resolved_workflow,
    _normalize_library_usi,
    _to_text,
)
from .tanimoto import scan_tanimoto

logger = logging.getLogger(__name__)


def _confidence_category(score):
    if score >= 80:
        return "Well-supported", "success"
    if score >= 55:
        return "Ambiguous", "warning"
    return "Insufficient evidence", "danger"



@lru_cache(maxsize=4096)
def _get_entropy_similarity_from_usis(query_usi, library_usi):
    query_usi = _to_text(query_usi)
    library_usi = _normalize_library_usi(library_usi)
    if not query_usi or not library_usi:
        return None

    try:
        from .spectral_entropy_v2 import (
            DEFAULT_MAX_PEAK_NUM,
            DEFAULT_MS2_TOLERANCE_DA,
            DEFAULT_NOISE_THRESHOLD,
            entropy_similarity_pair,
            fetch_spectrum,
        )
    except Exception:
        logger.exception("Failed to import spectral_entropy_v2")
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
        logger.warning("Entropy similarity failed for query=%s library=%s", query_usi, library_usi, exc_info=True)
        return None


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
    smiles_col = _find_first_column(df, ["Smiles", "SMILES", "smiles"])
    inchikey_col = _find_first_column(df, ["InChIKey", "inchikey", "INCHIKEY"])
    precursor_mz_col = _find_first_column(df, ["SpecMZ", "Precursor_MZ", "precursor_mz"])

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
    resolved_workflow = _infer_resolved_workflow(work_df, workflow_choice)

    summary_rows = []
    for scan, scan_df in work_df.groupby(scan_col):
        total_matches = len(scan_df)

        if smiles_col is not None:
            smiles_missing = (
                scan_df[smiles_col].astype(str).str.strip().str.lower().isin({"", "none", "nan", "n/a", "na", "null"})
            )
        else:
            smiles_missing = pd.Series(True, index=scan_df.index)

        if inchikey_col is not None:
            inchikey_missing = (
                scan_df[inchikey_col].astype(str).str.strip().str.lower().isin({"", "none", "nan", "n/a", "na", "null"})
            )
        else:
            inchikey_missing = pd.Series(True, index=scan_df.index)

        missing_structures = int((smiles_missing & inchikey_missing).sum())

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
        _mode = structure_hits["_compound"].mode()
        top_compound = _mode.iloc[0] if len(_mode) > 0 else ""

        support_fraction = best["count"] / total_matches if total_matches else 0.0
        mean_cosine = float(best["mean_cosine"])
        max_cosine = float(best["max_cosine"])

        # Tanimoto structural similarity (ECFP4, mean pairwise across unique library SMILES).
        # PubChem lookups are handled only on user request in detail view.
        scan_smiles = scan_df[smiles_col].tolist() if smiles_col is not None else []
        tanimoto_score, tanimoto_source = scan_tanimoto(scan_smiles, support_fraction)

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

        # Weights: structure_agreement 50%, tanimoto 15%, spectral_similarity 25%,
        #          max_cosine 10%.
        confidence_score = 100.0 * (
            0.50 * support_fraction
            + 0.15 * tanimoto_score
            + 0.25 * entropy_similarity
            + 0.10 * max_cosine
        )
        confidence_score = min(100.0, max(0.0, confidence_score))
        confidence_label, badge_color = _confidence_category(confidence_score)

        precursor_mz = None
        if precursor_mz_col is not None:
            raw_mz = pd.to_numeric(scan_df[precursor_mz_col], errors="coerce").dropna()
            if not raw_mz.empty:
                precursor_mz = round(float(raw_mz.iloc[0]), 4)

        summary_rows.append(
            {
                "scan": scan,
                "precursor_mz": precursor_mz,
                "top_compound": top_compound,
                "structure_key": structure_key,
                "support_group_key": support_group_key,
                "supporting_matches": int(best["count"]),
                "total_matches": int(total_matches),
                "missing_structures": missing_structures,
                "mean_cosine": round(mean_cosine, 4),
                "max_cosine": round(max_cosine, 4),
                "entropy_similarity": round(float(entropy_similarity), 4),
                "entropy_source": entropy_source,
                "tanimoto_score": round(float(tanimoto_score), 4),
                "tanimoto_source": tanimoto_source,
                "confidence_score": round(confidence_score, 1),
                "confidence_label": confidence_label,
                "badge_color": badge_color,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(["confidence_score", "supporting_matches"], ascending=False)

    return summary_df
