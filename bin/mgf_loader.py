# -*- coding: utf-8 -*-
"""
MGF-based spectrum loader for query entropy calculation.

Downloads the input MGF for a GNPS2 task once, caches it on disk under
temp/mgf_cache/, then parses it into a scan→peaks dict held in-process via
lru_cache.  This lets _compute_scan_level_confidence compute spectral entropy
for all query spectra with a single ~3 s download+parse instead of one USI
API call (~0.3 s each) per scored scan.

Public API
----------
get_query_entropy_from_mgf(task_id, scan) → (S, label) | None
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MGF_CACHE_DIR = Path(
    os.environ.get("TOPKONFIDENCE_MGF_CACHE_DIR", "temp/mgf_cache")
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _entropy_label(S: float) -> str:
    if S < 0.5:
        return "noisy / low quality"
    if S < 1.0:
        return "moderate quality"
    if S < 1.75:
        return "clean / good quality"
    return "very diffuse / uniform"


def _mgf_cache_path(task_id: str) -> Path:
    return MGF_CACHE_DIR / f"{task_id}.mgf"


# ── download ──────────────────────────────────────────────────────────────────

def _download_task_mgf(task_id: str) -> Path | None:
    """
    Download the input MGF for task_id to the local cache directory.
    Returns the local path on success, None on any failure.
    Skips download if the file is already cached.
    """
    if not task_id:
        return None

    local_path = _mgf_cache_path(task_id)
    if local_path.exists():
        logger.debug("MGF cache hit for task %s", task_id)
        return local_path

    try:
        from gnpsdata import taskinfo, taskresult
    except ImportError:
        logger.warning("gnpsdata not installed; MGF download unavailable")
        return None

    try:
        info = taskinfo.get_task_information(task_id, gnps2=True)
        input_path = info.get("submission_parameters", {}).get("inputspectra", "")
        if not input_path:
            logger.debug("No inputspectra param found for task %s", task_id)
            return None

        mgf_filename = os.path.basename(input_path)
        if not mgf_filename.lower().endswith(".mgf"):
            logger.debug("inputspectra for task %s is not an MGF: %s", task_id, mgf_filename)
            return None

        MGF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MGF for task %s (%s) → %s", task_id, mgf_filename, local_path)
        taskresult.download_gnps2_task_resultfile(
            task_id,
            f"inputspectra/{mgf_filename}",
            str(local_path),
        )
        logger.info("MGF download complete for task %s", task_id)
        return local_path

    except Exception:
        logger.warning("MGF download failed for task %s", task_id, exc_info=True)
        # Remove a partial download so the next attempt retries cleanly.
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


# ── parse ─────────────────────────────────────────────────────────────────────

def _parse_mgf(path: Path) -> dict:
    """
    Parse an MGF file into {scan_str: [[mz, intensity], ...]} dict.
    Scan key is the SCANS= field value as a string.
    """
    spectra: dict = {}
    current_scan: str | None = None
    current_peaks: list = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                current_scan = None
                current_peaks = []
            elif line == "END IONS":
                if current_scan is not None:
                    spectra[current_scan] = current_peaks
            elif line.startswith("SCANS="):
                current_scan = line.split("=", 1)[1].strip()
            elif "=" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current_peaks.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass

    logger.debug("Parsed %d spectra from %s", len(spectra), path)
    return spectra


# ── in-process cache ──────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _get_task_spectra(task_id: str) -> dict | None:
    """
    Download (if needed) and parse the input MGF for task_id.
    Returns {scan_str: peaks} or None on failure.
    Result is memoised for the lifetime of the process.
    """
    path = _download_task_mgf(task_id)
    if path is None:
        return None
    try:
        return _parse_mgf(path)
    except Exception:
        logger.warning("MGF parse failed for task %s", task_id, exc_info=True)
        return None


# ── public API ────────────────────────────────────────────────────────────────

def get_query_entropy_from_mgf(task_id: str, scan) -> tuple | None:
    """
    Compute spectral entropy S for scan from the task's local MGF.

    Parameters
    ----------
    task_id : GNPS2 task ID string
    scan    : scan number (str or int)

    Returns
    -------
    (S: float, label: str) or None if the spectrum cannot be found/computed.
    """
    spectra = _get_task_spectra(task_id)
    if spectra is None:
        return None

    peaks = spectra.get(str(scan))
    if not peaks:
        return None

    try:
        from .spectral_entropy_v2 import (
            DEFAULT_MAX_PEAK_NUM,
            DEFAULT_MS2_TOLERANCE_DA,
            DEFAULT_NOISE_THRESHOLD,
            spectral_entropy_for_spectrum,
        )
        S = spectral_entropy_for_spectrum(
            peaks,
            ms2_tolerance_da=DEFAULT_MS2_TOLERANCE_DA,
            noise_threshold=DEFAULT_NOISE_THRESHOLD,
            max_peak_num=DEFAULT_MAX_PEAK_NUM,
        )
        return round(S, 3), _entropy_label(S)
    except Exception:
        logger.warning("Entropy computation failed for task %s scan %s", task_id, scan, exc_info=True)
        return None
