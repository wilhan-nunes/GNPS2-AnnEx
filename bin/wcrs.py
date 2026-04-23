# -*- coding: utf-8 -*-
"""
Weighted Centroid Representativeness Score (WCRS).

Given subgroups with sizes and pairwise cross-group similarities, determines
which subgroup is most representative of the total evidence.

Algorithm
---------
For each subgroup i with size nᵢ, total N = Σ nⱼ:
  d̄ᵢ = (nᵢ·1.0 + Σⱼ≠ᵢ nⱼ·sim(i,j)) / N   weighted avg similarity to all evidence
  Sᵢ  = (nᵢ/N)^alpha · d̄ᵢ                   raw score
  Ŝᵢ  = Sᵢ / Σ Sⱼ                            normalized (sums to 1.0)

Winner = argmax(Ŝᵢ); ties broken by input order (first occurrence wins).
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Subgroup:
    id: str
    size: int


@dataclass
class SubgroupResult:
    id: str
    size: int
    size_weight: float
    avg_similarity: float   # d̄ᵢ — used as tanimoto_score in the confidence formula
    raw_score: float
    normalized_score: float


def wcrs(
    subgroups: list,
    similarities: dict,
    alpha: float = 1.0,
) -> tuple:
    """
    Compute the Weighted Centroid Representativeness Score for each subgroup.

    Parameters
    ----------
    subgroups : list of Subgroup
        Zero-size subgroups are filtered before scoring (warning logged).
    similarities : dict
        Keys are (id_a, id_b) tuples — only one direction needed per pair;
        both (a, b) and (b, a) are checked. Missing pairs default to 0.0.
    alpha : float
        Size weight exponent. 0 = ignore size; 1 = balanced; >1 = size dominates.

    Returns
    -------
    (results, winner_id)
        results   – list of SubgroupResult, one per valid subgroup
        winner_id – id of the subgroup with the highest normalized_score

    Raises
    ------
    ValueError
        If no subgroups remain after filtering zero-size entries.
    """
    valid = [sg for sg in subgroups if sg.size > 0]
    excluded = len(subgroups) - len(valid)
    if excluded:
        logger.warning("wcrs: excluded %d zero-size subgroup(s)", excluded)

    if not valid:
        raise ValueError("wcrs: no subgroups with size > 0")

    if len(valid) == 1:
        sg = valid[0]
        result = SubgroupResult(
            id=sg.id,
            size=sg.size,
            size_weight=1.0,
            avg_similarity=1.0,
            raw_score=1.0,
            normalized_score=1.0,
        )
        return [result], sg.id

    N = sum(sg.size for sg in valid)
    size_map = {sg.id: sg.size for sg in valid}

    def _sim(a: str, b: str) -> float:
        val = similarities.get((a, b))
        if val is not None:
            return float(val)
        val = similarities.get((b, a))
        if val is not None:
            return float(val)
        logger.warning("wcrs: missing similarity for pair (%s, %s); treating as 0.0", a, b)
        return 0.0

    raw_scores = []
    for sg in valid:
        cross = sum(
            size_map[other.id] * _sim(sg.id, other.id)
            for other in valid
            if other.id != sg.id
        )
        d_bar = (sg.size * 1.0 + cross) / N
        size_weight = (sg.size / N) ** alpha
        raw = size_weight * d_bar
        raw_scores.append((sg, size_weight, d_bar, raw))

    total = sum(r[3] for r in raw_scores)

    results = [
        SubgroupResult(
            id=sg.id,
            size=sg.size,
            size_weight=round(sw, 6),
            avg_similarity=round(d, 6),
            raw_score=round(raw, 6),
            normalized_score=round(raw / total, 6),
        )
        for sg, sw, d, raw in raw_scores
    ]

    winner = max(results, key=lambda r: r.normalized_score).id
    return results, winner
