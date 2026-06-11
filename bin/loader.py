# -*- coding: utf-8 -*-
import io
import logging
from functools import lru_cache

import pandas as pd
import requests

from gnpsdata import taskresult
from gnpsdata import taskinfo
from gnpsdata import workflow_classicnetworking
from gnpsdata import workflow_fbmn

from app import cache
from .constants import LIBRARY_RESULTS_CACHE_TIMEOUT_SEC
from .normalizer import _normalize_library_matches_dataframe

logger = logging.getLogger(__name__)


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


@cache.memoize(timeout=LIBRARY_RESULTS_CACHE_TIMEOUT_SEC)
def _load_retention_times_cached(task_id):
    try:
        quant_df = workflow_fbmn.get_quantification_dataframe(task_id, gnps2=True)
        if quant_df is None or quant_df.empty:
            return None
        if "row ID" not in quant_df.columns or "row retention time" not in quant_df.columns:
            logger.warning("Quant table missing expected RT columns for task %s", task_id)
            return None
        rt_df = (
            quant_df[["row ID", "row retention time"]]
            .drop_duplicates("row ID")
            .rename(columns={"row ID": "scan", "row retention time": "retention_time"})
        )
        rt_df["scan"] = rt_df["scan"].astype(str)
        return rt_df.to_json(orient="split")
    except Exception as exc:
        logger.warning("Could not load retention times for task %s: %s", task_id, exc)
        return None


def load_retention_times(task_id):
    task_id = (task_id or "").strip()
    if not task_id:
        return None
    raw = _load_retention_times_cached(task_id)
    if raw is None:
        return None
    return pd.read_json(io.StringIO(raw), orient="split")


@cache.memoize(timeout=LIBRARY_RESULTS_CACHE_TIMEOUT_SEC)
def _load_task_information_cached(task_id):
    task_id = (task_id or "").strip()
    if not task_id:
        return {}

    try:
        task_details = taskinfo.get_task_information(task_id, gnps2=True)
    except Exception as exc:
        logger.warning("Could not load task information for %s: %s", task_id, exc)
        return {}

    if isinstance(task_details, dict):
        return task_details

    return {"value": task_details}


@cache.memoize(timeout=LIBRARY_RESULTS_CACHE_TIMEOUT_SEC)
def _compute_scores_cached(task_id, workflow_choice, resolved_workflow, score_weights_tuple, ppm_scale=None, mz_precision_floor=None):
    """Memoized wrapper around _compute_scan_level_confidence.

    Cache key: task_id + workflow_choice + resolved_workflow + score weights + ppm params.
    TTL matches the library-match cache so results stay consistent.
    """
    from .scorer import _compute_scan_level_confidence  # local import avoids circular dep

    matches_df = _load_library_matches(task_id, workflow_choice)
    if matches_df is None or matches_df.empty:
        return None

    score_weights = dict(score_weights_tuple)
    summary_df = _compute_scan_level_confidence(
        matches_df,
        task_id=task_id,
        workflow_choice=resolved_workflow,
        score_weights=score_weights,
        ppm_scale=ppm_scale,
        mz_precision_floor=mz_precision_floor,
    )
    if summary_df is None or len(summary_df) == 0:
        return None
    return summary_df.to_json(orient="split")


def _load_task_information(task_id):
    task_id = (task_id or "").strip()
    if not task_id:
        return {}

    task_details = _load_task_information_cached(task_id)
    if isinstance(task_details, dict):
        return dict(task_details)

    return {}
