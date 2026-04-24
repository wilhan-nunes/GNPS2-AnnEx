import logging

import pytest

from bin.wcrs import Subgroup, SubgroupResult, wcrs


# ── reference fixture (dogs/wolves/cats from spec) ────────────────────────────

@pytest.fixture
def animal_subgroups():
    return [
        Subgroup(id="dogs",   size=5),
        Subgroup(id="wolves", size=3),
        Subgroup(id="cats",   size=2),
    ]


@pytest.fixture
def animal_similarities():
    return {
        ("dogs",   "wolves"): 0.75,
        ("dogs",   "cats"):   0.40,
        ("wolves", "cats"):   0.35,
    }


# ── reference case ────────────────────────────────────────────────────────────


class TestWcrsReferenceCase:
    def test_winner_is_dogs(self, animal_subgroups, animal_similarities):
        _, winner = wcrs(animal_subgroups, animal_similarities)
        assert winner == "dogs"

    def test_normalized_scores_match_spec(self, animal_subgroups, animal_similarities):
        results, _ = wcrs(animal_subgroups, animal_similarities)
        by_id = {r.id: r for r in results}
        assert by_id["dogs"].normalized_score   == pytest.approx(0.554, abs=0.001)
        assert by_id["wolves"].normalized_score == pytest.approx(0.307, abs=0.001)
        assert by_id["cats"].normalized_score   == pytest.approx(0.139, abs=0.001)

    def test_normalized_scores_sum_to_one(self, animal_subgroups, animal_similarities):
        results, _ = wcrs(animal_subgroups, animal_similarities)
        total = sum(r.normalized_score for r in results)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_avg_similarity_bounds(self, animal_subgroups, animal_similarities):
        results, _ = wcrs(animal_subgroups, animal_similarities)
        for r in results:
            assert 0.0 <= r.avg_similarity <= 1.0

    def test_size_weights_sum_to_one(self, animal_subgroups, animal_similarities):
        results, _ = wcrs(animal_subgroups, animal_similarities)
        total_sw = sum(r.size_weight for r in results)
        assert total_sw == pytest.approx(1.0, abs=1e-5)


# ── edge cases ────────────────────────────────────────────────────────────────


class TestWcrsEdgeCases:
    def test_single_subgroup_winner_and_perfect_score(self):
        results, winner = wcrs([Subgroup(id="only", size=4)], {})
        assert winner == "only"
        assert len(results) == 1
        assert results[0].normalized_score == pytest.approx(1.0)
        assert results[0].avg_similarity == pytest.approx(1.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no subgroups"):
            wcrs([], {})

    def test_all_zero_size_raises(self):
        with pytest.raises(ValueError, match="no subgroups"):
            wcrs([Subgroup(id="a", size=0), Subgroup(id="b", size=0)], {})

    def test_zero_size_subgroup_excluded(self, caplog):
        subgroups = [
            Subgroup(id="good", size=3),
            Subgroup(id="empty", size=0),
        ]
        with caplog.at_level(logging.WARNING, logger="bin.wcrs"):
            results, winner = wcrs(subgroups, {})
        assert winner == "good"
        assert len(results) == 1
        assert any("zero-size" in m for m in caplog.messages)

    def test_missing_similarity_pair_treated_as_zero(self, caplog):
        subgroups = [Subgroup(id="a", size=2), Subgroup(id="b", size=1)]
        with caplog.at_level(logging.WARNING, logger="bin.wcrs"):
            results, winner = wcrs(subgroups, {})  # no similarities provided
        assert winner == "a"  # larger group wins when sim=0
        assert any("missing similarity" in m for m in caplog.messages)

    def test_two_equal_scores_first_wins(self):
        subgroups = [Subgroup(id="x", size=1), Subgroup(id="y", size=1)]
        similarities = {("x", "y"): 0.5}
        results, winner = wcrs(subgroups, similarities)
        assert winner == "x"  # first in list wins ties

    def test_alpha_zero_ignores_size(self):
        subgroups = [Subgroup(id="big", size=100), Subgroup(id="small", size=1)]
        # small is more structurally central
        similarities = {("big", "small"): 0.1}
        results_alpha0, _ = wcrs(subgroups, similarities, alpha=0)
        by_id = {r.id: r for r in results_alpha0}
        # With alpha=0 all size_weights = 1.0; avg_sim decides
        # d̄(big)   = (100·1 + 1·0.1) / 101 ≈ 0.991 → wins by avg_sim too
        # This just checks that size_weight == 1.0 for both
        assert by_id["big"].size_weight   == pytest.approx(1.0)
        assert by_id["small"].size_weight == pytest.approx(1.0)

    def test_alpha_gt_one_amplifies_size(self):
        subgroups = [Subgroup(id="big", size=10), Subgroup(id="small", size=1)]
        similarities = {("big", "small"): 0.9}
        r_alpha1, winner1 = wcrs(subgroups, similarities, alpha=1.0)
        r_alpha3, winner3 = wcrs(subgroups, similarities, alpha=3.0)
        by1 = {r.id: r for r in r_alpha1}
        by3 = {r.id: r for r in r_alpha3}
        # Larger alpha should increase big's normalized_score relative to small's
        assert by3["big"].normalized_score > by1["big"].normalized_score

    def test_all_equal_similarities_size_decides(self):
        subgroups = [Subgroup(id="a", size=5), Subgroup(id="b", size=3), Subgroup(id="c", size=1)]
        similarities = {("a", "b"): 0.5, ("a", "c"): 0.5, ("b", "c"): 0.5}
        _, winner = wcrs(subgroups, similarities)
        assert winner == "a"  # largest group wins when all similarities equal

    def test_symmetric_lookup_both_directions(self, animal_subgroups):
        # Reverse the key direction — result must be identical
        sims_reversed = {
            ("wolves", "dogs"): 0.75,
            ("cats",   "dogs"): 0.40,
            ("cats", "wolves"): 0.35,
        }
        results, winner = wcrs(animal_subgroups, sims_reversed)
        assert winner == "dogs"
        by_id = {r.id: r for r in results}
        assert by_id["dogs"].normalized_score == pytest.approx(0.554, abs=0.001)
