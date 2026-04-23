# -*- coding: utf-8 -*-
import logging
import math
from functools import lru_cache

import pandas as pd

from .constants import ENTROPY_USI_TIMEOUT_SEC, K_SHARED_PEAKS, PPM_SCALE, MZ_PRECISION_FLOOR
from .normalizer import (
    _annotate_support_groups,
    _build_query_usi,
    _canonical_structure_key,
    _find_first_column,
    _infer_resolved_workflow,
    _normalize_library_usi,
    _to_text,
)
from .tanimoto import _canonical_smiles, compute_cross_group_tanimoto, scan_tanimoto
from .wcrs import Subgroup, wcrs
from .mgf_loader import get_query_entropy_from_mgf, _entropy_label

logger = logging.getLogger(__name__)

# Weights are pre-normalised (sum = 1.0).
# Nominal intent: structure 50 : tanimoto 15 : spectral 25 : cosine 10 : shared_peaks 10
# Divided by 110 to keep the sum exactly 1.
DEFAULT_SCORE_WEIGHTS = {
    "structure_agreement": 50 / 110,
    "tanimoto": 15 / 110,
    "spectral_similarity": 25 / 110,
    "max_cosine": 10 / 110,
    "shared_peaks": 10 / 110,
}


def _confidence_category(score):
    if score >= 80:
        return "Consistent evidence", "success"
    if score >= 55:
        return "Inconclusive", "warning"
    return "Inconsistent evidence", "danger"


def _normalize_score_weights(score_weights=None):
    normalized_weights = {}
    if not isinstance(score_weights, dict):
        return DEFAULT_SCORE_WEIGHTS.copy()

    for key, default_value in DEFAULT_SCORE_WEIGHTS.items():
        raw_value = score_weights.get(key, default_value)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = default_value
        if not math.isfinite(value) or value < 0:
            value = default_value
        normalized_weights[key] = value

    total_weight = sum(normalized_weights.values())
    if total_weight <= 0:
        return DEFAULT_SCORE_WEIGHTS.copy()

    return {
        key: value / total_weight
        for key, value in normalized_weights.items()
    }


def _compute_confidence_score(
    support_fraction,
    tanimoto_score,
    spectral_similarity,
    max_cosine,
    shared_peaks_score=0.0,
    score_weights=None,
):
    normalized_score_weights = _normalize_score_weights(score_weights)
    confidence_score = 100.0 * (
        normalized_score_weights["structure_agreement"] * float(support_fraction)
        + normalized_score_weights["tanimoto"] * float(tanimoto_score)
        + normalized_score_weights["spectral_similarity"] * float(spectral_similarity)
        + normalized_score_weights["max_cosine"] * float(max_cosine)
        + normalized_score_weights.get("shared_peaks", 0.0) * float(shared_peaks_score)
    )
    return min(100.0, max(0.0, confidence_score))



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


@lru_cache(maxsize=4096)
def _get_query_entropy_from_usi(query_usi: str):
    """
    Fallback: fetch a single query spectrum from the USI API and compute its
    entropy S.  Used when the MGF-based loader is unavailable (no task_id,
    download failed, scan not found in MGF).  Disk-cached via
    spectral_entropy_v2.fetch_spectrum.
    """
    query_usi = _to_text(query_usi)
    if not query_usi:
        return None
    try:
        from .spectral_entropy_v2 import (
            DEFAULT_MAX_PEAK_NUM,
            DEFAULT_MS2_TOLERANCE_DA,
            DEFAULT_NOISE_THRESHOLD,
            fetch_spectrum,
            spectral_entropy_for_spectrum,
        )
        spec = fetch_spectrum(query_usi, timeout=ENTROPY_USI_TIMEOUT_SEC)
        S = spectral_entropy_for_spectrum(
            spec.get("peaks", []),
            ms2_tolerance_da=DEFAULT_MS2_TOLERANCE_DA,
            noise_threshold=DEFAULT_NOISE_THRESHOLD,
            max_peak_num=DEFAULT_MAX_PEAK_NUM,
        )
        return round(S, 3), _entropy_label(S)
    except Exception:
        logger.debug("Query entropy USI fetch failed for %s", query_usi, exc_info=True)
        return None


def _compute_scan_level_confidence(
    df,
    task_id="",
    workflow_choice="auto",
    scan_filter=None,
    use_entropy_similarity=True,
    score_weights=None,
    ppm_scale=None,
    mz_precision_floor=None,
):
    scan_col = _find_first_column(df, ["#Scan#", "scan", "Scan"])
    cosine_col = _find_first_column(df, ["MQScore", "Cosine", "cosine"])
    compound_col = _find_first_column(df, ["Compound_Name", "compound_name", "Compound Name"])
    library_usi_col = _find_first_column(df, ["library_usi", "LibraryUSI", "library usi"])
    spectrum_file_col = _find_first_column(df, ["SpectrumFile"])
    smiles_col = _find_first_column(df, ["Smiles", "SMILES", "smiles"])
    inchikey_col = _find_first_column(df, ["InChIKey", "inchikey", "INCHIKEY"])
    precursor_mz_col = _find_first_column(df, ["SpecMZ", "Precursor_MZ", "precursor_mz"])
    shared_peaks_col = _find_first_column(df, ["SharedPeaks", "shared_peaks"])
    mz_error_col = _find_first_column(df, ["MZErrorPPM", "mz_error_ppm"])

    if scan_col is None:
        raise ValueError("Could not find scan column (#Scan# / scan / Scan) in library results.")
    if cosine_col is None:
        raise ValueError("Could not find cosine score column (MQScore / Cosine).")

    work_df = df.copy()
    work_df[scan_col] = work_df[scan_col].astype(str)
    if scan_filter is not None:
        work_df = work_df[work_df[scan_col] == str(scan_filter)].copy()

    work_df["_cosine"] = pd.to_numeric(work_df[cosine_col], errors="coerce").fillna(0.0)
    work_df["_shared_peaks"] = (
        pd.to_numeric(work_df[shared_peaks_col], errors="coerce").fillna(0.0)
        if shared_peaks_col is not None else 0.0
    )
    work_df["_abs_ppm"] = (
        pd.to_numeric(work_df[mz_error_col], errors="coerce").abs()
        if mz_error_col is not None else float("nan")
    )
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
    normalized_score_weights = _normalize_score_weights(score_weights)

    try:
        _ppm_scale = float(ppm_scale) if ppm_scale is not None else PPM_SCALE
        if not (math.isfinite(_ppm_scale) and _ppm_scale > 0):
            _ppm_scale = PPM_SCALE
    except (TypeError, ValueError):
        _ppm_scale = PPM_SCALE
    try:
        _mz_precision_floor = float(mz_precision_floor) if mz_precision_floor is not None else MZ_PRECISION_FLOOR
        if not (math.isfinite(_mz_precision_floor) and 0.0 <= _mz_precision_floor <= 1.0):
            _mz_precision_floor = MZ_PRECISION_FLOOR
    except (TypeError, ValueError):
        _mz_precision_floor = MZ_PRECISION_FLOOR

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
                std_cosine=("_cosine", "std"),
                max_cosine=("_cosine", "max"),
                mean_shared_peaks=("_shared_peaks", "mean"),
                median_abs_ppm=("_abs_ppm", "median"),
            )
            .sort_values(["count", "mean_cosine", "max_cosine"], ascending=False)
        )

        # Extract one representative SMILES per support group (highest-cosine hit).
        group_rep_smiles: dict = {}
        if smiles_col is not None:
            for gkey, gdf in scan_df.groupby("_support_group_key"):
                top_hit = gdf.sort_values("_cosine", ascending=False).iloc[0]
                raw_smi = _to_text(top_hit.get(smiles_col, ""))
                if raw_smi and raw_smi.lower() not in {"nan", "none"}:
                    canonical = _canonical_smiles(raw_smi)
                    group_rep_smiles[gkey] = canonical if canonical else raw_smi
                else:
                    group_rep_smiles[gkey] = None

        # Build WCRS subgroups from per-group hit counts.
        wcrs_subgroups = [
            Subgroup(id=str(row["_support_group_key"]), size=int(row["count"]))
            for _, row in grouped.iterrows()
        ]

        # WCRS: determine tanimoto_score as d̄ᵢ of the most representative group.
        wcrs_winner_key = None
        if len(wcrs_subgroups) >= 2 and any(v is not None for v in group_rep_smiles.values()):
            cross_sim = compute_cross_group_tanimoto(group_rep_smiles)
            wcrs_results, wcrs_winner_key = wcrs(wcrs_subgroups, cross_sim, alpha=1.0)
            winner_wcrs = next(r for r in wcrs_results if r.id == wcrs_winner_key)
            tanimoto_score: float = winner_wcrs.avg_similarity
            tanimoto_source: str = "tanimoto_wcrs"
        elif len(wcrs_subgroups) == 1 and any(v is not None for v in group_rep_smiles.values()):
            tanimoto_score = 1.0
            tanimoto_source = "single_structure"
        else:
            # No usable SMILES — fall back to count-based support fraction.
            preliminary_fraction = grouped.iloc[0]["count"] / total_matches if total_matches else 0.0
            scan_smiles = scan_df[smiles_col].tolist() if smiles_col is not None else []
            tanimoto_score, tanimoto_source = scan_tanimoto(scan_smiles, preliminary_fraction)

        # Resolve winner group: WCRS winner when available, else count-based best.
        if wcrs_winner_key is not None:
            mask = grouped["_support_group_key"] == wcrs_winner_key
            best = grouped[mask].iloc[0] if mask.any() else grouped.iloc[0]
        else:
            best = grouped.iloc[0]

        # Preserve count-based top compound for reference in the output.
        _count_hits = scan_df[scan_df["_support_group_key"] == grouped.iloc[0]["_support_group_key"]]
        _count_mode = _count_hits["_compound"].mode()
        count_based_top_compound = _count_mode.iloc[0] if len(_count_mode) > 0 else ""

        support_group_key = best["_support_group_key"]
        structure_hits = scan_df[scan_df["_support_group_key"] == support_group_key]
        representative_row = structure_hits.sort_values("_cosine", ascending=False).iloc[0]
        structure_key = representative_row.get("_structure_key", "UNKNOWN_STRUCTURE")
        _mode = structure_hits["_compound"].mode()
        top_compound = _mode.iloc[0] if len(_mode) > 0 else ""

        support_fraction = best["count"] / total_matches if total_matches else 0.0
        mean_cosine = float(best["mean_cosine"])
        max_cosine = float(best["max_cosine"])

        # SharedPeaks: soft normalisation x/(x+K) → [0, 1]
        raw_shared_peaks = float(best["mean_shared_peaks"]) if not math.isnan(float(best["mean_shared_peaks"])) else 0.0
        shared_peaks_score = raw_shared_peaks / (raw_shared_peaks + K_SHARED_PEAKS)

        # MZErrorPPM: multiplicative precision factor, floored to avoid over-penalising low-res instruments
        raw_abs_ppm = float(best["median_abs_ppm"])
        if math.isnan(raw_abs_ppm) or mz_error_col is None:
            mz_precision = 1.0  # no data → no penalty
        else:
            mz_precision = max(_mz_precision_floor, math.exp(-raw_abs_ppm / _ppm_scale))

        # Always build query_usi (needed for entropy similarity and USI fallback).
        representative_spectrum_file = _to_text(representative_row.get("_spectrum_file"))
        query_usi = _build_query_usi(task_id, scan, resolved_workflow, spectrum_file=representative_spectrum_file)

        # Query entropy: MGF (batch, ~3 s/task) → USI API fallback (~0.3 s/scan).
        query_entropy = None
        query_entropy_label = "N/A"
        q_entropy_result = get_query_entropy_from_mgf(task_id, scan)
        if q_entropy_result is None:
            q_entropy_result = _get_query_entropy_from_usi(query_usi)
        if q_entropy_result is not None:
            query_entropy, query_entropy_label = q_entropy_result

        # Cosine consistency: 1 − clamp(std/mean, 0, 1). Single-hit groups get 1.0.
        _std_cosine = float(best["std_cosine"])
        if math.isnan(_std_cosine) or int(best["count"]) < 2:
            cosine_consistency = 1.0
        else:
            cv = _std_cosine / mean_cosine if mean_cosine > 0 else 1.0
            cosine_consistency = max(0.0, 1.0 - min(1.0, cv))
        entropy_similarity = cosine_consistency
        entropy_source = "cosine_consistency"

        confidence_score = _compute_confidence_score(
            support_fraction,
            tanimoto_score,
            entropy_similarity,
            max_cosine,
            shared_peaks_score=shared_peaks_score,
            score_weights=normalized_score_weights,
        )
        confidence_score = min(100.0, max(0.0, confidence_score * mz_precision))
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
                "count_based_top_compound": count_based_top_compound,
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
                "shared_peaks": round(raw_shared_peaks, 2),
                "shared_peaks_score": round(shared_peaks_score, 4),
                "median_abs_ppm_error": round(raw_abs_ppm, 3) if not math.isnan(raw_abs_ppm) else None,
                "mz_precision": round(mz_precision, 4),
                "confidence_score": round(confidence_score, 1),
                "confidence_label": confidence_label,
                "badge_color": badge_color,
                "query_entropy": round(query_entropy, 3) if query_entropy is not None else None,
                "query_entropy_label": query_entropy_label,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(["confidence_score", "supporting_matches"], ascending=False)

    return summary_df
