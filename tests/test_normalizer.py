import pandas as pd
import pytest

from bin.normalizer import (
    _build_query_usi,
    _canonical_structure_key,
    _derive_smiles_from_inchi,
    _find_first_column,
    _infer_resolved_workflow,
    _is_valid_field,
    _normalize_library_matches_dataframe,
    _normalize_library_usi,
    _propagate_structure_by_name,
    _to_text,
    _truncate_text,
)
from tests.conftest import ASPIRIN_SMILES, CAFFEINE_SMILES, LEU_PHE_R_SMILES, LEU_PHE_S_SMILES, THEOPHYLLINE_SMILES


# ── _to_text ──────────────────────────────────────────────────────────────────


class TestToText:
    def test_none_returns_empty(self):
        assert _to_text(None) == ""

    @pytest.mark.parametrize("value", ["nan", "NaN", "NAN"])
    def test_nan_string_variants_return_empty(self, value):
        assert _to_text(value) == ""

    def test_strips_surrounding_whitespace(self):
        assert _to_text("  hello  ") == "hello"

    def test_plain_string_unchanged(self):
        assert _to_text("caffeine") == "caffeine"

    def test_integer_becomes_string(self):
        assert _to_text(42) == "42"


# ── _truncate_text ────────────────────────────────────────────────────────────


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert _truncate_text("abc", max_length=10) == "abc"

    def test_text_at_exact_limit_unchanged(self):
        assert _truncate_text("a" * 36) == "a" * 36

    def test_long_text_gets_ellipsis(self):
        result = _truncate_text("a" * 40, max_length=10)
        assert len(result) <= 10
        assert result.endswith("...")


# ── _is_valid_field ───────────────────────────────────────────────────────────


class TestIsValidField:
    @pytest.mark.parametrize("bad", [None, "", "nan", "none", "null", "n/a", "na", "NaN", "  "])
    def test_falsy_values(self, bad):
        assert _is_valid_field(bad) is False

    def test_truthy_value(self):
        assert _is_valid_field("CCMSLIB00001234") is True


# ── _find_first_column ────────────────────────────────────────────────────────


class TestFindFirstColumn:
    def test_returns_first_matching_candidate(self):
        df = pd.DataFrame({"scan": [1], "MQScore": [0.9]})
        assert _find_first_column(df, ["#Scan#", "scan", "Scan"]) == "scan"

    def test_case_insensitive_match(self):
        df = pd.DataFrame({"MQSCORE": [0.9]})
        assert _find_first_column(df, ["MQScore"]) == "MQSCORE"

    def test_priority_order_respected(self):
        # "#Scan#" should be preferred over "Scan" when both exist
        df = pd.DataFrame({"#Scan#": [1], "Scan": [2]})
        assert _find_first_column(df, ["#Scan#", "scan", "Scan"]) == "#Scan#"

    def test_returns_none_when_absent(self):
        df = pd.DataFrame({"foo": [1]})
        assert _find_first_column(df, ["MQScore", "Cosine"]) is None


# ── _canonical_structure_key ──────────────────────────────────────────────────


class TestCanonicalStructureKey:
    def test_inchikey_planar_preferred_over_all(self):
        row = {
            "InChIKey-Planar": "RYYVLZVUVIJVGH",
            "InChIKey": "RYYVLZVUVIJVGH-LFSVMHDDSA-N",
            "Smiles": CAFFEINE_SMILES,
        }
        assert _canonical_structure_key(row) == "RYYVLZVUVIJVGH"

    def test_inchikey_first_block_as_fallback(self):
        row = {"InChIKey": "RYYVLZVUVIJVGH-LFSVMHDDSA-N"}
        assert _canonical_structure_key(row) == "RYYVLZVUVIJVGH"

    def test_smiles_derived_key_is_14_chars(self):
        row = {"Smiles": CAFFEINE_SMILES}
        result = _canonical_structure_key(row)
        assert result != "UNKNOWN_STRUCTURE"
        assert len(result) == 14  # first InChIKey block is always 14 characters

    def test_unknown_structure_when_nothing_provided(self):
        assert _canonical_structure_key({}) == "UNKNOWN_STRUCTURE"

    def test_stereo_stripped_enantiomers_match(self):
        """R and S enantiomers of the same compound must yield the same planar key."""
        key_r = _canonical_structure_key({"Smiles": LEU_PHE_R_SMILES})
        key_s = _canonical_structure_key({"Smiles": LEU_PHE_S_SMILES})
        assert key_r == key_s
        assert key_r != "UNKNOWN_STRUCTURE"


# ── _normalize_library_usi ────────────────────────────────────────────────────


class TestNormalizeLibraryUSI:
    FULL_USI = "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003137982"

    def test_already_normalized_is_unchanged(self):
        assert _normalize_library_usi(self.FULL_USI) == self.FULL_USI

    def test_bare_accession_gets_full_prefix(self):
        assert _normalize_library_usi("CCMSLIB00003137982") == self.FULL_USI

    def test_prefix_without_accession_keyword(self):
        usi = "mzspec:GNPS:GNPS-LIBRARY:CCMSLIB00003137982"
        assert _normalize_library_usi(usi) == self.FULL_USI

    def test_empty_string_returns_empty(self):
        assert _normalize_library_usi("") == ""

    def test_none_returns_empty(self):
        assert _normalize_library_usi(None) == ""


# ── _build_query_usi ──────────────────────────────────────────────────────────


class TestBuildQueryUSI:
    def test_fbmn_contains_task_scan_and_mgf(self):
        usi = _build_query_usi("abc123", "42", "fbmn")
        assert "abc123" in usi
        assert ":scan:42" in usi
        assert "spectra_reformatted.mgf" in usi

    def test_auto_same_as_fbmn(self):
        assert _build_query_usi("t", "1", "auto") == _build_query_usi("t", "1", "fbmn")

    def test_librarysearch_uses_spectrum_file(self):
        usi = _build_query_usi("abc123", "42", "librarysearch", spectrum_file="input.mgf")
        assert "abc123" in usi
        assert ":scan:42" in usi
        assert "input.mgf" in usi

    def test_librarysearch_empty_file_returns_empty(self):
        assert _build_query_usi("abc123", "42", "librarysearch", spectrum_file="") == ""


# ── _infer_resolved_workflow ──────────────────────────────────────────────────


class TestInferResolvedWorkflow:
    def test_explicit_fbmn_returned_as_is(self):
        df = pd.DataFrame({"SpectrumFile": ["spectra_filtered.mgf"]})
        assert _infer_resolved_workflow(df, "fbmn") == "fbmn"

    def test_auto_with_generated_filename_returns_auto(self):
        df = pd.DataFrame({"SpectrumFile": ["spectra_reformatted.mgf"]})
        assert _infer_resolved_workflow(df, "auto") == "auto"

    def test_auto_with_custom_filename_infers_librarysearch(self):
        df = pd.DataFrame({"SpectrumFile": ["my_custom_file.mgf"]})
        assert _infer_resolved_workflow(df, "auto") == "librarysearch"

    def test_no_spectrum_file_col_returns_auto(self):
        df = pd.DataFrame({"MQScore": [0.9]})
        assert _infer_resolved_workflow(df, "auto") == "auto"


# ── _normalize_library_matches_dataframe ─────────────────────────────────────


class TestNormalizeLibraryMatchesDataframe:
    def test_renames_cosine_to_mqscore(self):
        df = pd.DataFrame({"scan": [1], "Cosine": [0.9]})
        result = _normalize_library_matches_dataframe(df)
        assert "MQScore" in result.columns

    def test_none_input_returns_none(self):
        assert _normalize_library_matches_dataframe(None) is None

    def test_existing_canonical_columns_not_overwritten(self):
        df = pd.DataFrame({"#Scan#": [1], "scan": [2], "MQScore": [0.9]})
        result = _normalize_library_matches_dataframe(df)
        # "#Scan#" already present — should not be renamed from "scan"
        assert "#Scan#" in result.columns
        assert "scan" in result.columns


# ── _propagate_structure_by_name ──────────────────────────────────────────────


class TestPropagateStructureByName:
    def test_fills_smiles_for_same_name_missing_row(self):
        df = pd.DataFrame({
            "Compound_Name": ["Caffeine", "Caffeine"],
            "Smiles": [CAFFEINE_SMILES, None],
        })
        result = _propagate_structure_by_name(df)
        assert result.loc[1, "Smiles"] == CAFFEINE_SMILES

    def test_fills_inchikey_from_sibling(self):
        df = pd.DataFrame({
            "Compound_Name": ["Caffeine", "Caffeine"],
            "Smiles": [None, None],
            "InChIKey": ["RYYVLZVUVIJVGH-UHFFFAOYSA-N", None],
        })
        result = _propagate_structure_by_name(df)
        assert result.loc[1, "InChIKey"] == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"

    def test_ambiguous_name_not_propagated(self):
        """Same name but different structures → do not propagate."""
        df = pd.DataFrame({
            "Compound_Name": ["X", "X", "X"],
            "Smiles": [CAFFEINE_SMILES, ASPIRIN_SMILES, None],
        })
        result = _propagate_structure_by_name(df)
        # The third row (missing SMILES) should remain empty
        assert not _is_valid_field(result.loc[2, "Smiles"])

    def test_all_rows_have_structure_no_change(self):
        df = pd.DataFrame({
            "Compound_Name": ["Caffeine", "Caffeine"],
            "Smiles": [CAFFEINE_SMILES, CAFFEINE_SMILES],
        })
        result = _propagate_structure_by_name(df)
        assert list(result["Smiles"]) == [CAFFEINE_SMILES, CAFFEINE_SMILES]

    def test_no_compound_name_column_returns_unchanged(self):
        df = pd.DataFrame({"Smiles": [CAFFEINE_SMILES, None]})
        result = _propagate_structure_by_name(df)
        assert result.equals(df)

    def test_no_structure_columns_returns_unchanged(self):
        df = pd.DataFrame({"Compound_Name": ["A", "A"], "MQScore": [0.9, 0.8]})
        result = _propagate_structure_by_name(df)
        assert list(result.columns) == list(df.columns)

    def test_different_names_not_cross_propagated(self):
        df = pd.DataFrame({
            "Compound_Name": ["Caffeine", "Aspirin"],
            "Smiles": [CAFFEINE_SMILES, None],
        })
        result = _propagate_structure_by_name(df)
        assert not _is_valid_field(result.loc[1, "Smiles"])

    def test_stereoisomers_same_planar_key_propagated(self):
        """Leu-Phe R and S have different SMILES but the same planar InChIKey → propagate."""
        df = pd.DataFrame({
            "Compound_Name": ["Leu-Phe", "Leu-Phe"],
            "Smiles": [LEU_PHE_R_SMILES, None],
        })
        result = _propagate_structure_by_name(df)
        assert _is_valid_field(result.loc[1, "Smiles"])

    def test_multiple_structure_fields_all_propagated(self):
        df = pd.DataFrame({
            "Compound_Name": ["Caffeine", "Caffeine"],
            "Smiles": [CAFFEINE_SMILES, None],
            "InChIKey": ["RYYVLZVUVIJVGH-UHFFFAOYSA-N", None],
        })
        result = _propagate_structure_by_name(df)
        assert _is_valid_field(result.loc[1, "Smiles"])
        assert _is_valid_field(result.loc[1, "InChIKey"])

    def test_case_insensitive_name_match(self):
        """'NORLEUCINE' (with SMILES) should propagate to 'Norleucine' (without)."""
        df = pd.DataFrame({
            "Compound_Name": ["NORLEUCINE", "Norleucine"],
            "Smiles": [CAFFEINE_SMILES, None],
        })
        result = _propagate_structure_by_name(df)
        assert result.loc[1, "Smiles"] == CAFFEINE_SMILES

    def test_case_insensitive_ambiguous_not_propagated(self):
        """Same name different case but different structures → do not propagate."""
        df = pd.DataFrame({
            "Compound_Name": ["X", "x", "X"],
            "Smiles": [CAFFEINE_SMILES, ASPIRIN_SMILES, None],
        })
        result = _propagate_structure_by_name(df)
        assert not _is_valid_field(result.loc[2, "Smiles"])

    def test_partial_structure_row_receives_missing_smiles(self):
        """Row with InChIKey but no SMILES should receive SMILES from sibling with same name."""
        df = pd.DataFrame({
            "Compound_Name": ["Caffeine", "Caffeine"],
            "Smiles": [CAFFEINE_SMILES, None],
            "InChIKey": [None, "RYYVLZVUVIJVGH-UHFFFAOYSA-N"],
        })
        result = _propagate_structure_by_name(df)
        assert result.loc[1, "Smiles"] == CAFFEINE_SMILES

    def test_normalize_pipeline_triggers_propagation(self):
        """_normalize_library_matches_dataframe should call propagation."""
        df = pd.DataFrame({
            "#Scan#": ["1", "1"],
            "MQScore": [0.9, 0.8],
            "Compound_Name": ["Caffeine", "Caffeine"],
            "Smiles": [CAFFEINE_SMILES, None],
        })
        result = _normalize_library_matches_dataframe(df)
        assert _is_valid_field(result.loc[1, "Smiles"])


# ── _derive_smiles_from_inchi ──────────────────────────────────────────────────

_CAFFEINE_INCHI = "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"


class TestDeriveSmilesToInchi:
    def test_fills_smiles_from_valid_inchi(self):
        df = pd.DataFrame({"Smiles": [None], "INCHI": [_CAFFEINE_INCHI]})
        result = _derive_smiles_from_inchi(df)
        assert _is_valid_field(result.loc[0, "Smiles"])

    def test_invalid_inchi_leaves_smiles_empty(self):
        df = pd.DataFrame({"Smiles": [None], "INCHI": ["not-an-inchi"]})
        result = _derive_smiles_from_inchi(df)
        assert not _is_valid_field(result.loc[0, "Smiles"])

    def test_existing_smiles_not_overwritten(self):
        df = pd.DataFrame({"Smiles": [CAFFEINE_SMILES], "INCHI": [_CAFFEINE_INCHI]})
        result = _derive_smiles_from_inchi(df)
        assert result.loc[0, "Smiles"] == CAFFEINE_SMILES

    def test_missing_columns_returns_unchanged(self):
        df = pd.DataFrame({"Smiles": [None]})
        result = _derive_smiles_from_inchi(df)
        assert result.equals(df)

    def test_pipeline_derives_smiles_from_inchi(self):
        """Full pipeline: InChI-only row should end up with a SMILES after normalization."""
        df = pd.DataFrame({
            "#Scan#": ["1"],
            "MQScore": [0.9],
            "Compound_Name": ["Caffeine"],
            "Smiles": [None],
            "INCHI": [_CAFFEINE_INCHI],
        })
        result = _normalize_library_matches_dataframe(df)
        assert _is_valid_field(result.loc[0, "Smiles"])
