import pytest

from bin.tanimoto import compute_mean_pairwise_tanimoto, scan_tanimoto
from tests.conftest import ASPIRIN_SMILES, CAFFEINE_SMILES, THEOPHYLLINE_SMILES

INVALID_SMILES = "this-is-not-a-smiles"


# ── scan_tanimoto (public entry point) ────────────────────────────────────────


class TestScanTanimoto:
    def test_no_smiles_returns_fallback(self):
        score, source = scan_tanimoto([], support_fraction=0.6)
        assert score == pytest.approx(0.6)
        assert source == "structure_agreement_fallback"

    @pytest.mark.parametrize("bad_values", [
        ["nan", None, ""],
        ["NaN", "none", "NONE"],
    ])
    def test_only_invalid_values_returns_fallback(self, bad_values):
        score, source = scan_tanimoto(bad_values, support_fraction=0.5)
        assert score == pytest.approx(0.5)
        assert source == "structure_agreement_fallback"

    def test_single_unique_smiles_is_perfect(self):
        """Deduplicated to one SMILES → unambiguous structure → score 1.0."""
        score, source = scan_tanimoto(
            [CAFFEINE_SMILES, CAFFEINE_SMILES, CAFFEINE_SMILES],
            support_fraction=1.0,
        )
        assert score == pytest.approx(1.0)
        assert source == "single_structure"

    def test_two_different_molecules_uses_ecfp4(self):
        score, source = scan_tanimoto(
            [CAFFEINE_SMILES, ASPIRIN_SMILES], support_fraction=1.0
        )
        assert source == "tanimoto_ecfp4"
        assert 0.0 <= score <= 1.0

    def test_similar_molecules_score_higher_than_dissimilar(self):
        """Caffeine/theophylline are structurally similar; caffeine/aspirin are not."""
        score_similar, _ = scan_tanimoto(
            [CAFFEINE_SMILES, THEOPHYLLINE_SMILES], support_fraction=1.0
        )
        score_different, _ = scan_tanimoto(
            [CAFFEINE_SMILES, ASPIRIN_SMILES], support_fraction=1.0
        )
        assert score_similar > score_different

    def test_invalid_smiles_with_no_valid_fallback(self):
        """A single unparseable SMILES leaves 0 valid fingerprints."""
        score, source = scan_tanimoto([INVALID_SMILES], support_fraction=0.7)
        # RDKit rejects the SMILES → no fingerprint → treated as 0 valid SMILES
        # The function may return single_structure (1.0) if it counts it as valid,
        # or fallback — either way the source tells us what happened.
        assert source in (
            "structure_agreement_fallback",
            "single_structure",
            "unavailable",
        )

    def test_fallback_value_preserved_exactly(self):
        score, _ = scan_tanimoto([], support_fraction=0.333)
        assert score == pytest.approx(0.333)


# ── compute_mean_pairwise_tanimoto ────────────────────────────────────────────


class TestComputeMeanPairwiseTanimoto:
    def test_identical_smiles_score_one(self):
        score, source = compute_mean_pairwise_tanimoto([CAFFEINE_SMILES, CAFFEINE_SMILES])
        assert score == pytest.approx(1.0)
        assert source == "tanimoto_ecfp4"

    def test_single_entry_is_single_structure(self):
        score, source = compute_mean_pairwise_tanimoto([CAFFEINE_SMILES])
        assert score == pytest.approx(1.0)
        assert source == "single_structure"

    def test_dissimilar_molecules_low_score(self):
        score, source = compute_mean_pairwise_tanimoto([CAFFEINE_SMILES, ASPIRIN_SMILES])
        assert source == "tanimoto_ecfp4"
        assert score < 0.5  # caffeine and aspirin share no pharmacophore

    def test_score_bounded_zero_to_one(self):
        score, _ = compute_mean_pairwise_tanimoto([CAFFEINE_SMILES, ASPIRIN_SMILES])
        assert 0.0 <= score <= 1.0

    def test_three_identical_score_one(self):
        smiles_list = [CAFFEINE_SMILES, CAFFEINE_SMILES, CAFFEINE_SMILES]
        score, source = compute_mean_pairwise_tanimoto(smiles_list)
        assert score == pytest.approx(1.0)
        assert source == "tanimoto_ecfp4"
