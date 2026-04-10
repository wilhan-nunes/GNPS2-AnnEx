# -*- coding: utf-8 -*-
"""
Tanimoto structural similarity for GNPS2 library annotation confidence.

Computes the mean pairwise Tanimoto similarity (ECFP4, Morgan radius=2, 2048 bits)
across the unique valid SMILES present in a scan's library matches.

Fallback behaviour
------------------
When no valid SMILES are present in a scan the function falls back to the
hit-count-based structure agreement fraction so the scorer always has a value.
When exactly one unique valid SMILES is present the scan is structurally
unambiguous and a perfect score of 1.0 is returned ("single_structure").

Source tags returned by scan_tanimoto
--------------------------------------
"tanimoto_ecfp4"              – computed from ≥ 2 unique valid SMILES
"single_structure"            – only one valid SMILES; all hits agree → 1.0
"structure_agreement_fallback"– fewer than MIN_STRUCTURES_FOR_TANIMOTO unique
                                 valid SMILES; uses the count-based fraction
"unavailable"                 – RDKit import failed (should not happen in practice)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of unique valid SMILES required to compute a pairwise Tanimoto.
# With exactly 1 valid SMILES the scan is unambiguous (score = 1.0, "single_structure").
# With 0 valid SMILES we cannot compute anything and fall back to support_fraction.
MIN_SMILES_FOR_PAIRWISE = 2


def _canonical_smiles(smiles: str):
    """Return RDKit canonical SMILES, or None if the molecule cannot be parsed."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _smiles_to_morgan_fp(smiles: str):
    """Return an ECFP4 Morgan fingerprint (2048-bit) for *smiles*, or None."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        logger.debug("Could not generate Morgan fingerprint for SMILES: %s", smiles)
        return None


def compute_mean_pairwise_tanimoto(unique_smiles: list) -> tuple:
    """
    Compute the mean pairwise Tanimoto similarity across *unique_smiles*.

    Parameters
    ----------
    unique_smiles : list of str
        Already-deduplicated, non-empty SMILES strings.

    Returns
    -------
    (score, source) : (float, str)
        score  – mean pairwise Tanimoto in [0, 1]
        source – "tanimoto_ecfp4" or "single_structure" (see module docstring)
    """
    try:
        from rdkit.Chem import DataStructs
    except ImportError:
        logger.warning("RDKit not available; cannot compute Tanimoto similarity.")
        return 0.5, "unavailable"

    fps = []
    for smi in unique_smiles:
        fp = _smiles_to_morgan_fp(smi)
        if fp is not None:
            fps.append(fp)

    if len(fps) == 0:
        return 0.5, "unavailable"

    if len(fps) == 1:
        # One unique structure: perfect structural consensus among all hits.
        return 1.0, "single_structure"

    sims = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps))
        for j in range(i + 1, len(fps))
    ]
    return round(float(np.mean(sims)), 4), "tanimoto_ecfp4"


def scan_tanimoto(smiles_iterable, support_fraction: float) -> tuple:
    """
    Return a Tanimoto-based structural agreement score for one scan.

    Collects unique non-empty SMILES from *smiles_iterable*.  If fewer than
    MIN_STRUCTURES_FOR_TANIMOTO are found, the hit-count-based
    *support_fraction* is returned instead with source "structure_agreement_fallback".

    Parameters
    ----------
    smiles_iterable : iterable of str
        SMILES values for all library matches in the scan (may contain NaN/None).
    support_fraction : float
        Fallback value: fraction of hits supporting the dominant structure.

    Returns
    -------
    (score, source) : (float, str)
    """
    seen: set = set()
    unique_smiles: list = []
    for raw in smiles_iterable:
        smi = str(raw).strip() if raw is not None else ""
        if not smi or smi.lower() in {"nan", "none"}:
            continue
        canonical = _canonical_smiles(smi)
        key = canonical if canonical is not None else smi
        if key not in seen:
            seen.add(key)
            unique_smiles.append(key)

    if len(unique_smiles) == 0:
        # No valid SMILES available at all — fall back to count-based agreement.
        return support_fraction, "structure_agreement_fallback"

    return compute_mean_pairwise_tanimoto(unique_smiles)
