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
    "cosine_consistency",
    "cosine_consistency_source",
    "median_abs_ppm_error",
    "shared_peaks_score",
    "shared_peaks_source",
    "query_peak_count",
    "library_diversity_score",
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
        # Nominal percentages 40:15:20:15:5:5 (= DEFAULT_SCORE_WEIGHTS × 100) should
        # normalize back to DEFAULT_SCORE_WEIGHTS.
        normalized = _normalize_score_weights({
            "structure_agreement": 40,
            "tanimoto": 15,
            "cosine_consistency": 20,
            "max_cosine": 15,
            "shared_peaks": 5,
            "library_diversity": 5,
        })
        assert normalized == pytest.approx(DEFAULT_SCORE_WEIGHTS, rel=1e-6)

    def test_zero_weights_fall_back_to_defaults(self):
        normalized = _normalize_score_weights({
            "structure_agreement": 0,
            "tanimoto": 0,
            "cosine_consistency": 0,
            "max_cosine": 0,
            "shared_peaks": 0,
            "library_diversity": 0,
        })
        assert normalized == pytest.approx(DEFAULT_SCORE_WEIGHTS)


# ── _compute_scan_level_confidence ────────────────────────────────────────────


class TestComputeScanLevelConfidence:

    # ── output shape ──────────────────────────────────────────────────────────

    def test_small_csv_returns_one_row(self, small_df):
        result = _compute_scan_level_confidence(small_df)
        assert len(result) == 1

    def test_small_csv_correct_scan_id(self, small_df):
        result = _compute_scan_level_confidence(small_df)
        assert result.iloc[0]["scan"] == "16050"

    def test_small_csv_total_matches(self, small_df):
        result = _compute_scan_level_confidence(small_df)
        assert result.iloc[0]["total_matches"] == 5

    def test_required_columns_present(self, small_df):
        result = _compute_scan_level_confidence(small_df)
        assert REQUIRED_COLUMNS.issubset(set(result.columns))

    # ── score bounds and labels ───────────────────────────────────────────────

    def test_score_in_valid_range(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df,         )
        assert (result["confidence_score"] >= 0).all()
        assert (result["confidence_score"] <= 100).all()

    def test_label_consistent_with_score(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df,         )
        for _, row in result.iterrows():
            expected_label, expected_badge = _confidence_category(row["confidence_score"])
            assert row["confidence_label"] == expected_label
            assert row["badge_color"] == expected_badge

    # ── scan_filter ───────────────────────────────────────────────────────────

    def test_scan_filter_isolates_single_scan(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df, scan_filter="10"
        )
        assert len(result) == 1
        assert result.iloc[0]["scan"] == "10"

    def test_scan_filter_no_match_returns_empty(self, minimal_df):
        result = _compute_scan_level_confidence(
            minimal_df, scan_filter="9999"
        )
        assert len(result) == 0

    # ── error cases ───────────────────────────────────────────────────────────

    def test_missing_scan_column_raises(self):
        df = pd.DataFrame({"MQScore": [0.9]})
        with pytest.raises(ValueError, match="scan column"):
            _compute_scan_level_confidence(df)

    def test_missing_cosine_column_raises(self):
        df = pd.DataFrame({"#Scan#": ["1"]})
        with pytest.raises(ValueError, match="cosine"):
            _compute_scan_level_confidence(df)

    # ── structure agreement logic ─────────────────────────────────────────────

    def test_full_consensus_gives_supporting_equals_total(self, single_scan_df):
        """3 hits with identical structure → supporting_matches == total_matches."""
        result = _compute_scan_level_confidence(
            single_scan_df,         )
        row = result.iloc[0]
        assert row["supporting_matches"] == 3
        assert row["total_matches"] == 3

    def test_full_consensus_tanimoto_is_single_structure(self, single_scan_df):
        result = _compute_scan_level_confidence(
            single_scan_df,         )
        assert result.iloc[0]["tanimoto_source"] == "single_structure"

    def test_unknown_structures_not_grouped(self):
        """Hits with no structure info must each get a unique sentinel key
        so they do not artificially inflate the support count."""
        df = pd.DataFrame({
            "#Scan#": ["5", "5"],
            "MQScore": [0.8, 0.6],
        })
        result = _compute_scan_level_confidence(df)
        row = result.iloc[0]
        assert row["total_matches"] == 2
        assert row["supporting_matches"] == 1  # only the best-cosine hit counts

    def test_partial_structure_agreement(self, mixed_structure_df):
        """Scan 10 has 2/3 hits agreeing → supporting_matches == 2."""
        result = _compute_scan_level_confidence(
            mixed_structure_df,         )
        scan_10 = result[result["scan"] == "10"].iloc[0]
        assert scan_10["total_matches"] == 3
        assert scan_10["supporting_matches"] == 2

    # ── cosine consistency path ───────────────────────────────────────────────

    def test_spectral_source_is_cosine_consistency(self, small_df):
        result = _compute_scan_level_confidence(small_df)
        assert result.iloc[0]["cosine_consistency_source"] == "cosine_consistency"

    # ── sorting ───────────────────────────────────────────────────────────────

    def test_results_sorted_by_confidence_descending(self, mixed_structure_df):
        result = _compute_scan_level_confidence(
            mixed_structure_df,         )
        scores = result["confidence_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_percentage_style_weights_match_default_scores(self, mixed_structure_df):
        # Passing explicit weights matching DEFAULT_SCORE_WEIGHTS ratios (40:15:20:15:5:5)
        # should produce identical scores to passing no weights at all.
        baseline = _compute_scan_level_confidence(
            mixed_structure_df,         )
        reweighted = _compute_scan_level_confidence(
            mixed_structure_df,
            score_weights={
                "structure_agreement": 40,
                "tanimoto": 15,
                "cosine_consistency": 20,
                "max_cosine": 15,
                "shared_peaks": 5,
                "library_diversity": 5,
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
            score_weights={
                "structure_agreement": 1.0,
                "tanimoto": 0.0,
                "cosine_consistency": 0.0,
                "max_cosine": 0.0,
                "shared_peaks": 0.0,
                "library_diversity": 0.0,
            },
        )
        scan_10 = result[result["scan"] == "10"].iloc[0]
        # Bayesian-shrunk support_fraction: (2+1)/(3+2) = 0.60 → score = 60.0
        assert scan_10["confidence_score"] == pytest.approx(60.0, abs=0.1)


# ── WCRS integration ──────────────────────────────────────────────────────────


class TestWcrsIntegration:
    def test_two_group_scan_uses_wcrs_source(self, two_group_scan_df):
        result = _compute_scan_level_confidence(two_group_scan_df)
        row = result.iloc[0]
        assert row["tanimoto_source"] == "tanimoto_wcrs"
        assert 0.0 <= row["tanimoto_score"] <= 1.0

    def test_single_group_still_single_structure(self, single_scan_df):
        result = _compute_scan_level_confidence(single_scan_df)
        assert result.iloc[0]["tanimoto_source"] == "single_structure"

    def test_no_smiles_falls_back_gracefully(self):
        df = __import__("pandas").DataFrame({
            "#Scan#": ["7", "7"],
            "MQScore": [0.9, 0.8],
        })
        result = _compute_scan_level_confidence(df)
        assert result.iloc[0]["tanimoto_source"] == "structure_agreement_fallback"

    def test_count_based_top_compound_present(self, two_group_scan_df):
        result = _compute_scan_level_confidence(two_group_scan_df)
        assert "count_based_top_compound" in result.columns

    def test_wcrs_score_within_bounds(self, mixed_structure_df):
        result = _compute_scan_level_confidence(mixed_structure_df)
        assert (result["tanimoto_score"] >= 0.0).all()
        assert (result["tanimoto_score"] <= 1.0).all()

    def test_wcrs_does_not_break_confidence_score_range(self, mixed_structure_df):
        result = _compute_scan_level_confidence(mixed_structure_df)
        assert (result["confidence_score"] >= 0.0).all()
        assert (result["confidence_score"] <= 100.0).all()

    def test_cosine_consistency_single_hit_is_zero(self):
        """A support group with exactly one hit has no variance — consistency is 0.0 (undefined).
        max_cosine carries the spectral evidence for single-hit scans."""
        import pandas as pd
        from tests.conftest import CAFFEINE_SMILES
        df = pd.DataFrame({
            "#Scan#": ["77"],
            "MQScore": [0.92],
            "Smiles": [CAFFEINE_SMILES],
            "Compound_Name": ["Caffeine"],
            "InChIKey-Planar": ["RYYVLZVUVIJVGH"],
        })
        result = _compute_scan_level_confidence(df)
        assert result.iloc[0]["cosine_consistency"] == pytest.approx(0.0)
        assert result.iloc[0]["cosine_consistency_source"] == "single_hit_undefined"

    def test_cosine_consistency_in_bounds(self, mixed_structure_df):
        result = _compute_scan_level_confidence(mixed_structure_df)
        assert (result["cosine_consistency"] >= 0.0).all()
        assert (result["cosine_consistency"] <= 1.0).all()


# ── shared peaks normalization ────────────────────────────────────────────────


class TestSharedPeaksNormalization:
    def _df(self, shared_peaks=6):
        return pd.DataFrame({
            "#Scan#": ["1"],
            "MQScore": [0.9],
            "SharedPeaks": [shared_peaks],
        })

    def _patch(self, monkeypatch, peak_count):
        monkeypatch.setattr("bin.scorer.get_filtered_query_peak_count_from_mgf", lambda t, s, **kw: peak_count)
        monkeypatch.setattr("bin.scorer.get_query_entropy_from_mgf", lambda t, s: None)
        monkeypatch.setattr("bin.scorer._get_query_entropy_from_usi", lambda u: None)

    def test_coverage_fraction_path(self, monkeypatch):
        self._patch(monkeypatch, 10)
        result = _compute_scan_level_confidence(self._df(shared_peaks=3), task_id="fake")
        row = result.iloc[0]
        assert row["shared_peaks_source"] == "coverage_fraction"
        assert row["shared_peaks_score"] == pytest.approx(3 / 10, rel=1e-4)
        assert row["query_peak_count"] == 10

    def test_coverage_fraction_caps_at_one(self, monkeypatch):
        self._patch(monkeypatch, 2)
        result = _compute_scan_level_confidence(self._df(shared_peaks=5), task_id="fake")
        assert result.iloc[0]["shared_peaks_score"] == pytest.approx(1.0)

    def test_perfect_coverage_scores_one(self, monkeypatch):
        self._patch(monkeypatch, 3)
        result = _compute_scan_level_confidence(self._df(shared_peaks=3), task_id="fake")
        row = result.iloc[0]
        assert row["shared_peaks_score"] == pytest.approx(1.0)
        assert row["shared_peaks_source"] == "coverage_fraction"

    def test_soft_normalization_fallback_when_none(self, monkeypatch):
        self._patch(monkeypatch, None)
        result = _compute_scan_level_confidence(self._df(shared_peaks=6), task_id="fake")
        row = result.iloc[0]
        assert row["shared_peaks_source"] == "soft_normalization"
        assert row["shared_peaks_score"] == pytest.approx(6 / (6 + 5), rel=1e-4)
        assert row["query_peak_count"] is None

    def test_soft_normalization_fallback_when_zero(self, monkeypatch):
        self._patch(monkeypatch, 0)
        result = _compute_scan_level_confidence(self._df(shared_peaks=6), task_id="fake")
        row = result.iloc[0]
        assert row["shared_peaks_source"] == "soft_normalization"
        assert row["shared_peaks_score"] == pytest.approx(6 / 11, rel=1e-4)

    def test_new_columns_present(self, monkeypatch):
        self._patch(monkeypatch, 20)
        result = _compute_scan_level_confidence(self._df(shared_peaks=4), task_id="fake")
        assert "shared_peaks_source" in result.columns
        assert "query_peak_count" in result.columns
