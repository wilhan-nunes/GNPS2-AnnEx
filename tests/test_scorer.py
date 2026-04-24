import pandas as pd
import pytest

from bin.scorer import (
    DEFAULT_SCORE_WEIGHTS,
    _confidence_category,
    _compute_scan_level_confidence,
    _normalize_score_weights,
)

REQUIRED_COLUMNS = {
    "scan",
    "confidence_score",
    "confidence_label",
    "badge_color",
    "total_matches",
    "supporting_matches",
    "max_cosine",
    "tanimoto_score",
    "tanimoto_source",
    "entropy_similarity",
    "entropy_source",
    "median_abs_ppm_error",
    "shared_peaks_score",
}


# ── _confidence_category ──────────────────────────────────────────────────────


class TestConfidenceCategory:
    @pytest.mark.parametrize("score,label,badge", [
        (100.0, "Consistent evidence", "success"),
        (80.0,  "Consistent evidence", "success"),
        (79.9,  "Inconclusive", "warning"),
        (55.0,  "Inconclusive", "warning"),
        (54.9,  "Inconsistent evidence", "danger"),
        (0.0,   "Inconsistent evidence", "danger"),
    ])
    def test_boundary_thresholds(self, score, label, badge):
        result_label, result_badge = _confidence_category(score)
        assert result_label == label
        assert result_badge == badge


class TestScoreWeightHelpers:
    def test_percentage_weights_normalize_to_defaults(self):
        # Nominal percentages 50:15:25:10:10 should normalize to DEFAULT_SCORE_WEIGHTS.
        normalized = _normalize_score_weights({
            "structure_agreement": 50,
            "tanimoto": 15,
            "spectral_similarity": 25,
            "max_cosine": 10,
            "shared_peaks": 10,
        })
        assert normalized == pytest.approx(DEFAULT_SCORE_WEIGHTS, rel=1e-6)

    def test_zero_weights_fall_back_to_defaults(self):
        normalized = _normalize_score_weights({
            "structure_agreement": 0,
            "tanimoto": 0,
            "spectral_similarity": 0,
            "max_cosine": 0,
            "shared_peaks": 0,
        })
        assert normalized == pytest.approx(DEFAULT_SCORE_WEIGHTS)


# ── _compute_scan_level_confidence ────────────────────────────────────────────


class TestComputeScanLevelConfidence:

    # ── output shape ──────────────────────────────────────────────────────────

    def test_small_csv_returns_one_row(self, small_df):
        result = _compute_scan_level_confidence(small_df, use_entropy_similarity=False)
        assert len(result) == 1

    def test_small_csv_correct_scan_id(self, small_df):
        result = _compute_scan_level_confidence(small_df, use_entropy_similarity=False)
        assert result.iloc[0]["scan"] == "16050"

    def test_small_csv_total_matches(self, small_df):
        result = _compute_scan_level_confidence(small_df, use_entropy_similarity=False)
        assert result.iloc[0]["total_matches"] == 5

    def test_required_columns_present(self, small_df):
        result = _compute_scan_level_confidence(small_df, use_entropy_similarity=False)
        assert REQUIRED_COLUMNS.issubset(set(result.columns))

    # ── score bounds and labels ───────────────────────────────────────────────

    def test_score_in_valid_range(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df, use_entropy_similarity=False
        )
        assert (result["confidence_score"] >= 0).all()
        assert (result["confidence_score"] <= 100).all()

    def test_label_consistent_with_score(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df, use_entropy_similarity=False
        )
        for _, row in result.iterrows():
            expected_label, expected_badge = _confidence_category(row["confidence_score"])
            assert row["confidence_label"] == expected_label
            assert row["badge_color"] == expected_badge

    # ── scan_filter ───────────────────────────────────────────────────────────

    def test_scan_filter_isolates_single_scan(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df, use_entropy_similarity=False, scan_filter="10"
        )
        assert len(result) == 1
        assert result.iloc[0]["scan"] == "10"

    def test_scan_filter_no_match_returns_empty(self, minimal_df):
        result = _compute_scan_level_confidence(
            minimal_df, use_entropy_similarity=False, scan_filter="9999"
        )
        assert len(result) == 0

    # ── error cases ───────────────────────────────────────────────────────────

    def test_missing_scan_column_raises(self):
        df = pd.DataFrame({"MQScore": [0.9]})
        with pytest.raises(ValueError, match="scan column"):
            _compute_scan_level_confidence(df, use_entropy_similarity=False)

    def test_missing_cosine_column_raises(self):
        df = pd.DataFrame({"#Scan#": ["1"]})
        with pytest.raises(ValueError, match="cosine"):
            _compute_scan_level_confidence(df, use_entropy_similarity=False)

    # ── structure agreement logic ─────────────────────────────────────────────

    def test_full_consensus_gives_supporting_equals_total(self, single_scan_df):
        """3 hits with identical structure → supporting_matches == total_matches."""
        result = _compute_scan_level_confidence(
            single_scan_df, use_entropy_similarity=False
        )
        row = result.iloc[0]
        assert row["supporting_matches"] == 3
        assert row["total_matches"] == 3

    def test_full_consensus_tanimoto_is_single_structure(self, single_scan_df):
        result = _compute_scan_level_confidence(
            single_scan_df, use_entropy_similarity=False
        )
        assert result.iloc[0]["tanimoto_source"] == "single_structure"

    def test_unknown_structures_not_grouped(self):
        """Hits with no structure info must each get a unique sentinel key
        so they do not artificially inflate the support count."""
        df = pd.DataFrame({
            "#Scan#": ["5", "5"],
            "MQScore": [0.8, 0.6],
        })
        result = _compute_scan_level_confidence(df, use_entropy_similarity=False)
        row = result.iloc[0]
        assert row["total_matches"] == 2
        assert row["supporting_matches"] == 1  # only the best-cosine hit counts

    def test_partial_structure_agreement(self, mixed_structure_df):
        """Scan 10 has 2/3 hits agreeing → supporting_matches == 2."""
        result = _compute_scan_level_confidence(
            mixed_structure_df, use_entropy_similarity=False
        )
        scan_10 = result[result["scan"] == "10"].iloc[0]
        assert scan_10["total_matches"] == 3
        assert scan_10["supporting_matches"] == 2

    # ── entropy disabled path ─────────────────────────────────────────────────

    def test_spectral_source_is_cosine_consistency(self, small_df):
        result = _compute_scan_level_confidence(small_df, use_entropy_similarity=False)
        assert result.iloc[0]["entropy_source"] == "cosine_consistency"

    # ── sorting ───────────────────────────────────────────────────────────────

    def test_results_sorted_by_confidence_descending(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df, use_entropy_similarity=False
        )
        scores = result["confidence_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_percentage_style_weights_match_default_scores(self, mixed_structure_df):
        baseline = _compute_scan_level_confidence(
            mixed_structure_df, use_entropy_similarity=False
        )
        reweighted = _compute_scan_level_confidence(
            mixed_structure_df,
            use_entropy_similarity=False,
            score_weights={
                "structure_agreement": 50,
                "tanimoto": 15,
                "spectral_similarity": 25,
                "max_cosine": 10,
                "shared_peaks": 10,
            },
        )
        pd.testing.assert_series_equal(
            baseline["confidence_score"],
            reweighted["confidence_score"],
            check_names=False,
        )

    def test_support_only_weighting_uses_structure_agreement_score(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df,
            use_entropy_similarity=False,
            score_weights={
                "structure_agreement": 1.0,
                "tanimoto": 0.0,
                "spectral_similarity": 0.0,
                "max_cosine": 0.0,
                "shared_peaks": 0.0,
            },
        )
        scan_10 = result[result["scan"] == "10"].iloc[0]
        assert scan_10["confidence_score"] == pytest.approx(66.7, abs=0.1)


# ── WCRS integration ──────────────────────────────────────────────────────────


class TestWcrsIntegration:
    def test_two_group_scan_uses_wcrs_source(self, two_group_scan_df):
        result = _compute_scan_level_confidence(two_group_scan_df, use_entropy_similarity=False)
        row = result.iloc[0]
        assert row["tanimoto_source"] == "tanimoto_wcrs"
        assert 0.0 <= row["tanimoto_score"] <= 1.0

    def test_single_group_still_single_structure(self, single_scan_df):
        result = _compute_scan_level_confidence(single_scan_df, use_entropy_similarity=False)
        assert result.iloc[0]["tanimoto_source"] == "single_structure"

    def test_no_smiles_falls_back_gracefully(self):
        df = __import__("pandas").DataFrame({
            "#Scan#": ["7", "7"],
            "MQScore": [0.9, 0.8],
        })
        result = _compute_scan_level_confidence(df, use_entropy_similarity=False)
        assert result.iloc[0]["tanimoto_source"] == "structure_agreement_fallback"

    def test_count_based_top_compound_present(self, two_group_scan_df):
        result = _compute_scan_level_confidence(two_group_scan_df, use_entropy_similarity=False)
        assert "count_based_top_compound" in result.columns

    def test_wcrs_score_within_bounds(self, mixed_structure_df):
        result = _compute_scan_level_confidence(mixed_structure_df, use_entropy_similarity=False)
        assert (result["tanimoto_score"] >= 0.0).all()
        assert (result["tanimoto_score"] <= 1.0).all()

    def test_wcrs_does_not_break_confidence_score_range(self, mixed_structure_df):
        result = _compute_scan_level_confidence(mixed_structure_df, use_entropy_similarity=False)
        assert (result["confidence_score"] >= 0.0).all()
        assert (result["confidence_score"] <= 100.0).all()

    def test_cosine_consistency_single_hit_is_one(self):
        """A support group with exactly one hit has no variance → consistency = 1.0."""
        import pandas as pd
        from tests.conftest import CAFFEINE_SMILES
        df = pd.DataFrame({
            "#Scan#": ["77"],
            "MQScore": [0.92],
            "Smiles": [CAFFEINE_SMILES],
            "Compound_Name": ["Caffeine"],
            "InChIKey-Planar": ["RYYVLZVUVIJVGH"],
        })
        result = _compute_scan_level_confidence(df, use_entropy_similarity=False)
        assert result.iloc[0]["entropy_similarity"] == pytest.approx(1.0)
        assert result.iloc[0]["entropy_source"] == "cosine_consistency"

    def test_cosine_consistency_in_bounds(self, mixed_structure_df):
        result = _compute_scan_level_confidence(mixed_structure_df, use_entropy_similarity=False)
        assert (result["entropy_similarity"] >= 0.0).all()
        assert (result["entropy_similarity"] <= 1.0).all()
