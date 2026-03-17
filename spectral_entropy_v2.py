"""
Spectral entropy + similarity calculator for MS/MS spectra using the ms-entropy package.

Fetches two spectra via the GNPS2 Metabolomics USI Resolver API and computes:
  - Spectral entropy          for each spectrum (via ms_entropy.calculate_spectral_entropy)
  - Unweighted entropy similarity  between the two spectra
  - Entropy similarity (weighted)  between the two spectra

These metrics match those shown in the MSViewer web app (https://yuanyue.li/msviewer/).

API endpoint : https://metabolomics-usi.gnps2.org/json/?usi1=<USI>
USI format   : mzspec:GNPS:GNPS-LIBRARY:accession:<ACCESSION_ID>

Usage (CLI):
    python spectral_entropy_v2.py
    python spectral_entropy_v2.py \\
        --usi1 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622" \\
        --usi2 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001548"
    python spectral_entropy_v2.py \\
        --usi1 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622" \\
        --usi2 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001548" \\
        --tolerance-da 0.02

Dash integration:
    from spectral_entropy_v2 import compute_entropy_similarity_payload
    payload = compute_entropy_similarity_payload(usi1, usi2)
    # payload is a flat dict; keys described in compute_entropy_similarity_payload docstring.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests not installed.  Run: pip install requests")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not installed.  Run: pip install numpy")
    sys.exit(1)

try:
    import ms_entropy as me
except ImportError:
    print("ms-entropy not installed.  Run: pip install ms-entropy")
    sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

GNPS_USI_BASE = "https://metabolomics-usi.gnps2.org"

EXAMPLE_USIS = {
    "quercetin_3_glucoside": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622",
    "caffeine":              "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001548",
    "rutin":                 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579358",
}

# Default ms-entropy parameters — identical to MSViewer defaults
DEFAULT_MS2_TOLERANCE_DA  = 0.02   # Da fragment matching tolerance
DEFAULT_NOISE_THRESHOLD   = 0.01   # fraction of base peak; peaks below this are removed
DEFAULT_MAX_PEAK_NUM      = 100    # maximum peaks kept per spectrum (ms-entropy default)
DEFAULT_CACHE_TTL_SEC     = 24 * 60 * 60  # 1 day

CACHE_DIR = Path(
    os.environ.get(
        "TOPKONFIDENCE_CACHE_DIR",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "topkonfidence" / "spectral_entropy_v2",
    )
)


def _cache_file_for_usi(usi: str) -> Path:
    digest = hashlib.sha256(usi.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"spectrum_{digest}.json"


def _load_cached_spectrum(usi: str, cache_ttl_sec: int) -> dict | None:
    cache_file = _cache_file_for_usi(usi)
    if not cache_file.exists():
        return None

    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        fetched_at = float(raw.get("fetched_at", 0))
        payload = raw.get("payload")
        if time.time() - fetched_at > cache_ttl_sec:
            return None
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None

    return None


def _write_cached_spectrum(usi: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_file_for_usi(usi)
        temp_file = cache_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload}, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(temp_file, cache_file)
    except Exception:
        # Cache failures should never block primary functionality.
        pass


# ── API ───────────────────────────────────────────────────────────────────────

def fetch_spectrum(
    usi: str,
    timeout: int = 15,
    cache_ttl_sec: int = DEFAULT_CACHE_TTL_SEC,
) -> dict:
    """
    Fetch a spectrum from the GNPS2 USI resolver.

    Returns a dict with keys:
        usi           : str   – echoed back
        peaks         : list  – [[mz, intensity], ...] raw from API
        precursor_mz  : float
        compound_name : str
        adduct        : str
        instrument    : str
        n_peaks       : int   – raw peak count before any filtering

    Raises:
        requests.HTTPError  on non-2xx HTTP response
        requests.Timeout    if the server takes too long
    """
    usi = (usi or "").strip()
    if not usi:
        raise ValueError("USI is empty.")

    print(f'retrieving usi: "{usi}"')

    cached = _load_cached_spectrum(usi, cache_ttl_sec=cache_ttl_sec)
    if cached is not None:
        print("status: success (cache hit)")
        return cached

    try:
        url = f"{GNPS_USI_BASE}/json/"
        response = requests.get(url, params={"usi1": usi}, timeout=timeout)
        response.raise_for_status()
        raw = response.json()
    except Exception as exc:
        print(f"status: failed ({exc.__class__.__name__}: {exc})")
        raise

    annotations = raw.get("annotations", [{}])
    meta        = annotations[0] if annotations else {}
    peaks       = raw.get("peaks", [])

    payload = {
        "usi":           usi,
        "peaks":         peaks,
        "precursor_mz":  float(raw.get("precursor_mz") or meta.get("Precursor_MZ", 0)),
        "compound_name": meta.get("Compound_Name", "Unknown"),
        "adduct":        meta.get("Adduct", "Unknown"),
        "instrument":    meta.get("Instrument", "Unknown"),
        "n_peaks":       len(peaks),
    }

    _write_cached_spectrum(usi, payload)
    print("status: success")
    return payload


def _to_np_peaks(peaks: list) -> np.ndarray:
    """Convert a list of [mz, intensity] pairs to a float32 numpy array."""
    if not peaks:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[float(p[0]), float(p[1])] for p in peaks], dtype=np.float32)


# ── Per-spectrum entropy ───────────────────────────────────────────────────────

def spectral_entropy_for_spectrum(
    peaks: list,
    ms2_tolerance_da: float = DEFAULT_MS2_TOLERANCE_DA,
    noise_threshold:  float = DEFAULT_NOISE_THRESHOLD,
    max_peak_num:     int   = DEFAULT_MAX_PEAK_NUM,
) -> float:
    """
    Compute spectral entropy for a single spectrum using ms-entropy.

    Internally calls ms_entropy.calculate_spectral_entropy with clean_spectrum=True,
    which matches the preprocessing used by MSViewer.

    Parameters
    ----------
    peaks            : list of [mz, intensity] pairs (raw, not normalised)
    ms2_tolerance_da : fragment merging tolerance in Da (default 0.02)
    noise_threshold  : peaks with intensity < noise_threshold × base-peak are removed
    max_peak_num     : maximum number of peaks kept

    Returns
    -------
    float – spectral entropy in nats. Range [0, ln(N)].
    """
    arr = _to_np_peaks(peaks)
    if arr.shape[0] == 0:
        return 0.0
    return float(
        me.calculate_spectral_entropy(
            arr,
            clean_spectrum=True,
            min_ms2_difference_in_da=ms2_tolerance_da,
            noise_threshold=noise_threshold,
            max_peak_num=max_peak_num,
        )
    )


# ── Pairwise similarity ────────────────────────────────────────────────────────

def entropy_similarity_pair(
    peaks_a: list,
    peaks_b: list,
    ms2_tolerance_da: float = DEFAULT_MS2_TOLERANCE_DA,
    noise_threshold:  float = DEFAULT_NOISE_THRESHOLD,
    max_peak_num:     int   = DEFAULT_MAX_PEAK_NUM,
) -> dict:
    """
    Compute both entropy similarity metrics between two spectra.

    Parameters
    ----------
    peaks_a / peaks_b : list of [mz, intensity] pairs (raw)
    ms2_tolerance_da  : fragment matching tolerance in Da
    noise_threshold   : relative intensity floor for noise removal
    max_peak_num      : maximum peaks per spectrum

    Returns
    -------
    dict with keys:
        unweighted_entropy_similarity : float  in [0, 1]
        entropy_similarity            : float  in [0, 1]  (weighted / main metric)
    """
    arr_a = _to_np_peaks(peaks_a)
    arr_b = _to_np_peaks(peaks_b)

    shared_kwargs = dict(
        ms2_tolerance_in_da=ms2_tolerance_da,
        clean_spectra=True,
        noise_threshold=noise_threshold,
        max_peak_num=max_peak_num,
    )

    unweighted = float(
        me.calculate_unweighted_entropy_similarity(arr_a, arr_b, **shared_kwargs)
    )
    weighted = float(
        me.calculate_entropy_similarity(arr_a, arr_b, **shared_kwargs)
    )

    return {
        "unweighted_entropy_similarity": round(unweighted, 6),
        "entropy_similarity":            round(weighted,   6),
    }


# ── Quality label ─────────────────────────────────────────────────────────────

def entropy_quality_label(S: float) -> tuple[str, str]:
    """
    Map spectral entropy to a quality tier and indicator.
    Thresholds from Li et al., Nature Methods 2021.

    Returns (label, symbol)
    """
    if S < 0.5:
        return "noisy / low quality",    "⚠"
    if S < 1.0:
        return "moderate quality",       "~"
    if S < 1.75:
        return "clean / good quality",   "✓"
    return     "very diffuse / uniform", "?"


# ── High-level payload (Dash-ready) ───────────────────────────────────────────

def compute_entropy_similarity_payload(
    usi1: str,
    usi2: str,
    ms2_tolerance_da: float = DEFAULT_MS2_TOLERANCE_DA,
    noise_threshold:  float = DEFAULT_NOISE_THRESHOLD,
    max_peak_num:     int   = DEFAULT_MAX_PEAK_NUM,
) -> dict:
    """
    Fetch two spectra and compute all entropy metrics.

    Designed for direct use in Dash callbacks. Returns a single flat dict.

    Parameters
    ----------
    usi1 / usi2      : GNPS2 Universal Spectrum Identifiers
    ms2_tolerance_da : MS2 fragment matching tolerance in Da (default 0.02,
                       same as MSViewer)
    noise_threshold  : relative intensity floor (default 0.01, same as MSViewer)
    max_peak_num     : max peaks per spectrum kept (default 100)

    Returns
    -------
    dict – all keys listed below.

    ── Spectrum 1 ──────────────────────────────────────────────────────────────
    usi1                          str    input USI
    compound_name_1               str
    precursor_mz_1                float
    adduct_1                      str
    instrument_1                  str
    n_peaks_raw_1                 int    peak count before filtering
    spectral_entropy_1            float  nats; output of ms_entropy
    spectral_entropy_1_quality    str    quality label
    peaks_1                       list   [[mz, intensity], ...] raw

    ── Spectrum 2 ──────────────────────────────────────────────────────────────
    usi2                          str
    compound_name_2               str
    precursor_mz_2                float
    adduct_2                      str
    instrument_2                  str
    n_peaks_raw_2                 int
    spectral_entropy_2            float
    spectral_entropy_2_quality    str
    peaks_2                       list

    ── Pairwise similarity ──────────────────────────────────────────────────────
    unweighted_entropy_similarity float  range [0, 1]; matches MSViewer
    entropy_similarity            float  range [0, 1]; matches MSViewer (main metric)

    ── Parameters used ─────────────────────────────────────────────────────────
    ms2_tolerance_da              float
    noise_threshold               float
    max_peak_num                  int
    """
    spec1 = fetch_spectrum(usi1)
    spec2 = fetch_spectrum(usi2)

    S1 = spectral_entropy_for_spectrum(
        spec1["peaks"], ms2_tolerance_da, noise_threshold, max_peak_num
    )
    S2 = spectral_entropy_for_spectrum(
        spec2["peaks"], ms2_tolerance_da, noise_threshold, max_peak_num
    )

    similarity = entropy_similarity_pair(
        spec1["peaks"], spec2["peaks"], ms2_tolerance_da, noise_threshold, max_peak_num
    )

    label1, _ = entropy_quality_label(S1)
    label2, _ = entropy_quality_label(S2)

    return {
        # ── Spectrum 1 ─────────────────────────────────────────────────────
        "usi1":                       spec1["usi"],
        "compound_name_1":            spec1["compound_name"],
        "precursor_mz_1":             spec1["precursor_mz"],
        "adduct_1":                   spec1["adduct"],
        "instrument_1":               spec1["instrument"],
        "n_peaks_raw_1":              spec1["n_peaks"],
        "spectral_entropy_1":         round(S1, 6),
        "spectral_entropy_1_quality": label1,
        "peaks_1":                    spec1["peaks"],

        # ── Spectrum 2 ─────────────────────────────────────────────────────
        "usi2":                       spec2["usi"],
        "compound_name_2":            spec2["compound_name"],
        "precursor_mz_2":             spec2["precursor_mz"],
        "adduct_2":                   spec2["adduct"],
        "instrument_2":               spec2["instrument"],
        "n_peaks_raw_2":              spec2["n_peaks"],
        "spectral_entropy_2":         round(S2, 6),
        "spectral_entropy_2_quality": label2,
        "peaks_2":                    spec2["peaks"],

        # ── Pairwise similarity ────────────────────────────────────────────
        **similarity,

        # ── Parameters used ───────────────────────────────────────────────
        "ms2_tolerance_da":  ms2_tolerance_da,
        "noise_threshold":   noise_threshold,
        "max_peak_num":      max_peak_num,
    }


# ── CLI report ────────────────────────────────────────────────────────────────

def print_report(payload: dict) -> None:
    """Print a human-readable summary of the payload to stdout."""
    W   = 64
    sep = "─" * W
    def row(label, value): print(f"  {label:<32} {value}")

    print(f"\n{sep}")
    print(f"  Spectral Entropy Similarity Report  (ms-entropy v{me.__version__})")
    print(sep)

    for n, suffix in [(1, "1"), (2, "2")]:
        S      = payload[f"spectral_entropy_{n}"]
        label  = payload[f"spectral_entropy_{n}_quality"]
        _, sym = entropy_quality_label(S)
        print(f"\n  ── Spectrum {n} {'─'*(W-14)}")
        row("Compound",           payload[f"compound_name_{n}"])
        row("USI",                payload[f"usi{n}"][-40:])
        row("Adduct",             payload[f"adduct_{n}"])
        row(f"Precursor m/z",     f"{payload[f'precursor_mz_{n}']:.4f}")
        row("Instrument",         payload[f"instrument_{n}"])
        row("Raw peaks",          payload[f"n_peaks_raw_{n}"])
        row("Spectral entropy S", f"{S:.4f}  nats")
        row("Quality",            f"{sym}  {label}")

    print(f"\n{sep}")
    print(f"  ── Pairwise similarity {'─'*(W-24)}")
    row("Unweighted entropy similarity", f"{payload['unweighted_entropy_similarity']:.4f}")
    row("Entropy similarity (weighted)", f"{payload['entropy_similarity']:.4f}")
    row("MS2 tolerance",                f"{payload['ms2_tolerance_da']} Da")
    row("Noise threshold",              f"{payload['noise_threshold']} × base peak")
    row("Max peaks",                    payload["max_peak_num"])
    print(f"\n  Note: values match MSViewer (https://yuanyue.li/msviewer/)")
    print(f"{sep}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute spectral entropy and entropy similarity for two GNPS2 spectra "
            "using the ms-entropy package (Li et al., Nat Methods 2021/2023)."
        )
    )
    parser.add_argument(
        "--usi1",
        default=EXAMPLE_USIS["quercetin_3_glucoside"],
        help="USI of the first (query) spectrum.",
    )
    parser.add_argument(
        "--usi2",
        default=EXAMPLE_USIS["caffeine"],
        help="USI of the second (reference) spectrum.",
    )
    parser.add_argument(
        "--tolerance-da",
        type=float,
        default=DEFAULT_MS2_TOLERANCE_DA,
        metavar="DA",
        help=f"MS2 fragment matching tolerance in Da (default: {DEFAULT_MS2_TOLERANCE_DA}).",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=DEFAULT_NOISE_THRESHOLD,
        metavar="FRAC",
        help=f"Remove peaks below FRAC × base peak (default: {DEFAULT_NOISE_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=DEFAULT_MAX_PEAK_NUM,
        metavar="N",
        help=f"Maximum peaks per spectrum (default: {DEFAULT_MAX_PEAK_NUM}).",
    )
    args = parser.parse_args()

    print(f"Fetching spectrum 1: {args.usi1}")
    print(f"Fetching spectrum 2: {args.usi2}")

    try:
        payload = compute_entropy_similarity_payload(
            usi1=args.usi1,
            usi2=args.usi2,
            ms2_tolerance_da=args.tolerance_da,
            noise_threshold=args.noise_threshold,
            max_peak_num=args.max_peaks,
        )
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print_report(payload)


if __name__ == "__main__":
    main()
