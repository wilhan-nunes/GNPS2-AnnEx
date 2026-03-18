"""
Spectral entropy calculator for MS/MS spectra
Fetches peaks from the GNPS2 Metabolomics USI Resolver API.

API endpoint: https://metabolomics-usi.gnps2.org/json/?usi1=<USI>
USI format:   mzspec:GNPS:GNPS-LIBRARY:accession:<ACCESSION_ID>

Usage:
    python spectral_entropy.py
    python spectral_entropy.py --usi "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622"
    python spectral_entropy.py --usi "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622" --top 20
"""

import argparse
import math
import sys
from typing import Optional

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ── API ───────────────────────────────────────────────────────────────────────

GNPS_USI_BASE = "https://metabolomics-usi.gnps2.org"

EXAMPLE_USIS = {
    "0821_triethylphosphate": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622",
    "hoidamide B":              "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001548",
    "Theophylline":                 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579358",
}


def fetch_spectrum(usi: str, timeout: int = 15) -> dict:
    """
    Fetch a spectrum from the GNPS2 USI resolver.

    Returns a dict with keys:
        peaks         : list of [mz, intensity] pairs
        precursor_mz  : float
        compound_name : str
        adduct        : str
        n_peaks       : int
        instrument    : str
        usi           : str (echoed back)

    Raises:
        requests.HTTPError   on non-2xx response
        requests.Timeout     if the server takes too long
        KeyError             if the response is missing expected fields
    """
    url = f"{GNPS_USI_BASE}/json/"
    response = requests.get(url, params={"usi1": usi}, timeout=timeout)
    response.raise_for_status()
    raw = response.json()

    # The /json/ endpoint nests spectrum data under "peaks" as [[mz, intensity], ...]
    # Metadata lives at the top level or inside "annotations" (list of dicts)
    annotations = raw.get("annotations", [{}])
    meta = annotations[0] if annotations else {}

    peaks = raw.get("peaks", [])

    return {
        "usi":           usi,
        "peaks":         peaks,                                    # [[mz, intensity], ...]
        "precursor_mz":  float(raw.get("precursor_mz") or meta.get("Precursor_MZ", 0)),
        "compound_name": meta.get("Compound_Name", "Unknown"),
        "adduct":        meta.get("Adduct", "Unknown"),
        "instrument":    meta.get("Instrument", "Unknown"),
        "n_peaks":       len(peaks),
    }


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_peaks(
    peaks: list,
    top_n: Optional[int] = None,
    min_intensity_frac: float = 0.01,
) -> tuple[list[float], list[float]]:
    """
    Clean and filter peaks.

    Steps:
      1. Remove peaks below min_intensity_frac × base peak intensity.
      2. Optionally keep only the top_n most intense peaks.
      3. Normalise intensities to [0, 1] relative to the base peak.

    Returns:
        mzs         : list of m/z values (float)
        intensities : list of normalised intensities (float, max = 1.0)
    """
    if not peaks:
        return [], []

    mzs_raw   = [float(p[0]) for p in peaks]
    ints_raw  = [float(p[1]) for p in peaks]
    max_int   = max(ints_raw)

    # filter low-intensity noise
    threshold = min_intensity_frac * max_int
    filtered  = [(mz, i) for mz, i in zip(mzs_raw, ints_raw) if i >= threshold]

    # sort by intensity descending, optionally limit
    filtered.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        filtered = filtered[:top_n]

    # re-sort by m/z for display
    filtered.sort(key=lambda x: x[0])

    mzs        = [p[0] for p in filtered]
    intensities = [p[1] / max_int for p in filtered]   # normalised to base peak = 1.0

    return mzs, intensities


# ── Entropy ───────────────────────────────────────────────────────────────────

def spectral_entropy(intensities: list[float]) -> float:
    """
    Compute spectral entropy S = -∑ pᵢ · ln(pᵢ)
    where pᵢ = Iᵢ / ∑Iᵢ  (probability distribution over peaks).

    Returns S in nats (natural log). Range: [0, ln(N)].
    """
    total = sum(intensities)
    if total == 0:
        return 0.0
    ps = [i / total for i in intensities]
    return -sum(p * math.log(p) for p in ps if p > 0)


def per_peak_contributions(intensities: list[float]) -> list[float]:
    """Return the per-peak entropy contribution -pᵢ ln(pᵢ) for each peak."""
    total = sum(intensities)
    if total == 0:
        return [0.0] * len(intensities)
    ps = [i / total for i in intensities]
    return [-p * math.log(p) if p > 0 else 0.0 for p in ps]


def entropy_quality_label(S: float) -> tuple[str, str]:
    """
    Map entropy value to a quality label and an emoji indicator.
    Thresholds from Li et al., Nature Methods 2021.

    Returns (label, indicator)
    """
    if S < 0.5:
        return "noisy / low quality",     "⚠"
    if S < 1.0:
        return "moderate quality",        "~"
    if S < 1.75:
        return "clean / good quality",    "✓"
    return     "very diffuse / uniform",  "?"


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(spectrum: dict, mzs: list, intensities: list) -> None:
    """Print a human-readable entropy report to stdout."""
    n         = len(intensities)
    S         = spectral_entropy(intensities)
    max_S     = math.log(n) if n > 1 else 0.0
    contribs  = per_peak_contributions(intensities)
    label, indicator = entropy_quality_label(S)
    total_raw = sum(intensities)
    ps        = [i / total_raw for i in intensities] if total_raw else [0] * n

    W = 62
    sep = "─" * W

    print(f"\n{'─'*W}")
    print(f"  Spectral Entropy Report")
    print(sep)
    print(f"  Compound   : {spectrum['compound_name']}")
    print(f"  USI        : {spectrum['usi']}")
    print(f"  Adduct     : {spectrum['adduct']}")
    print(f"  Precursor  : {spectrum['precursor_mz']:.4f} m/z")
    print(f"  Instrument : {spectrum['instrument']}")
    print(f"  Raw peaks  : {spectrum['n_peaks']}")
    print(f"  Used peaks : {n}  (after preprocessing)")
    print(sep)
    print(f"  S  = {S:.4f}  nats")
    print(f"  ln(N) = ln({n}) = {max_S:.4f}   (theoretical maximum)")
    print(f"  S / ln(N) = {S/max_S:.3f}   (fractional entropy)")
    print(f"  Quality    : {indicator}  {label}")
    print(sep)

    # per-peak table
    print(f"  {'m/z':>9}   {'norm. I':>8}   {'pᵢ':>7}   {'−pᵢ ln pᵢ':>10}")
    print(f"  {'─'*9}   {'─'*8}   {'─'*7}   {'─'*10}")
    for mz, i, p, c in zip(mzs, intensities, ps, contribs):
        print(f"  {mz:9.4f}   {i:8.4f}   {p:7.4f}   {c:10.4f}")
    print(f"  {'─'*9}   {'─'*8}   {sum(ps):7.4f}   {S:10.4f}  ← S")
    print(sep)

    # ASCII bar chart
    print(f"\n  Spectrum (normalised, base peak = 1.0):\n")
    bar_width = 30
    for mz, intensity in zip(mzs, intensities):
        bar = "█" * int(intensity * bar_width)
        print(f"  {mz:8.2f}  |{bar:<{bar_width}}  {intensity:.3f}")
    print()


# ── Python dict output (for Dash integration) ─────────────────────────────────

def compute_entropy_payload(usi: str, top_n: int = 50) -> dict:
    """
    High-level function: fetch spectrum, preprocess, compute entropy.
    Returns a single dict ready to pass into a Dash callback or pandas DataFrame.

    Example return value:
    {
        "usi":             "mzspec:GNPS:...",
        "compound_name":   "Quercetin 3-glucoside",
        "precursor_mz":    465.1029,
        "adduct":          "[M+H]+",
        "n_peaks_raw":     23,
        "n_peaks_used":    18,
        "entropy":         1.432,
        "entropy_max":     2.890,
        "entropy_frac":    0.496,
        "quality_label":   "clean / good quality",
        "mzs":             [121.03, 153.02, ...],
        "intensities":     [0.08, 0.45, ...],   # normalised
        "probabilities":   [0.02, 0.11, ...],   # pᵢ
        "contributions":   [0.04, 0.24, ...],   # −pᵢ ln pᵢ
    }
    """
    spectrum   = fetch_spectrum(usi)
    mzs, ints  = preprocess_peaks(spectrum["peaks"], top_n=top_n)
    n          = len(mzs)
    S          = spectral_entropy(ints)
    max_S      = math.log(n) if n > 1 else 0.0
    contribs   = per_peak_contributions(ints)
    total      = sum(ints)
    ps         = [i / total for i in ints] if total else [0.0] * n
    label, _   = entropy_quality_label(S)

    return {
        "usi":           spectrum["usi"],
        "compound_name": spectrum["compound_name"],
        "precursor_mz":  spectrum["precursor_mz"],
        "adduct":        spectrum["adduct"],
        "instrument":    spectrum["instrument"],
        "n_peaks_raw":   spectrum["n_peaks"],
        "n_peaks_used":  n,
        "entropy":       round(S, 6),
        "entropy_max":   round(max_S, 6),
        "entropy_frac":  round(S / max_S, 4) if max_S > 0 else 0.0,
        "quality_label": label,
        "mzs":           mzs,
        "intensities":   ints,
        "probabilities": ps,
        "contributions": contribs,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute spectral entropy for a GNPS2 spectrum via USI."
    )
    parser.add_argument(
        "--usi",
        default=EXAMPLE_USIS["quercetin_3_glucoside"],
        help="Universal Spectrum Identifier (USI). Defaults to quercetin 3-glucoside.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        metavar="N",
        help="Keep only the top N most intense peaks (default: 50).",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.01,
        metavar="FRAC",
        help="Remove peaks below FRAC × base peak intensity (default: 0.01).",
    )
    args = parser.parse_args()

    print(f"Fetching: {args.usi}")
    try:
        spectrum = fetch_spectrum(args.usi)
    except Exception as e:
        print(f"Error fetching spectrum: {e}")
        sys.exit(1)

    mzs, intensities = preprocess_peaks(
        spectrum["peaks"],
        top_n=args.top,
        min_intensity_frac=args.min_intensity,
    )

    if not mzs:
        print("No peaks found after preprocessing.")
        sys.exit(1)

    print_report(spectrum, mzs, intensities)


if __name__ == "__main__":
    main()
