# -*- coding: utf-8 -*-
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


def _load_task_information(task_id):
    task_id = (task_id or "").strip()
    if not task_id:
        return {}

    task_details = _load_task_information_cached(task_id)
    if isinstance(task_details, dict):
        return dict(task_details)

    return {}
