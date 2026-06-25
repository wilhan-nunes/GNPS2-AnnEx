# -*- coding: utf-8 -*-
import logging
import re
from functools import lru_cache

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import InchiToInchiKey, MolFromInchi, MolToInchi
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.error")

from .constants import SUPPORTED_WORKFLOWS

logger = logging.getLogger(__name__)


def _clean_compound_name(name: str) -> str:
    """Clean a library compound name using matchms-derived regex patterns."""
    if not isinstance(name, str):
        return name
    name = name.strip()

    # Acyl-lipid candidate annotations are not compound names.
    if re.match(r"^Candidate\s+\S+.*\(delta\s+mass:[^)]*\)", name, re.IGNORECASE):
        return ""

    # Strip parenthetical isomer-count notes injected by some workflows.
    name = re.sub(r"\s*\(known structural isomers:[^)]*\)", "", name, flags=re.IGNORECASE)

    # matchms: _remove_parts_by_regular_expression
    name = re.split(r"[A-Z]{3,6}[0-9]{8,12}-[0-9]{2,5}_[A-Z,0-9]{4,15}_", name)[-1]
    name = re.split(r"[A-Z]{3,6}[0-9]{8,12}-[0-9]{2,3}\!", name)[-1]
    name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2}[0-9]{5,6}.*\|", name)[-1]
    name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2}[0-9]{5,6}", name)[-1]
    name = re.split(r"HMDB:HMDB[0-9]{4,7}-[0-9]{1,7}", name)[-1]
    name = re.split(r"MoNA:[0-9]{5,10}", name)[-1]
    name = re.split(r"ReSpect:[A-Z]{2,3}[0-9]{6}.*\|", name)[-1]
    name = re.split(r"[A-Z]{2,3}[0-9]{6}( )", name)[-1]
    name = re.split(r"^[0-9]{4}_", name)[-1]
    name = re.split(r"_((HCD)|(CID))[0-9]{2}_[0-9]{5,6}$", name)[0]
    name = re.split(r"(?: - )?[0-9]+(?:\.[0-9]+)? ?[eE][Vv](?: Unknown)?$", name)[0]

    # Strip "CollisionEnergy:XXXXXX" suffix used by CMMC and some GNPS2 libraries.
    name = re.sub(r"\s*CollisionEnergy:\d+\s*$", "", name, flags=re.IGNORECASE)

    # matchms: _remove_known_non_compound_parts
    for part in ["Spectral Match to", "from NIST14", "Massbank:"]:
        name = name.replace(part, "")
    name = name.strip("; ")

    # matchms: _remove_misplaced_mass  e.g. "Caffeine 195.0" -> "Caffeine"
    end = name.split(" ")[-1]
    if re.search(r"^[0-9]{2,4}\.[0-9]$", end):
        name = name[: -len(end)].strip()

    name = name.strip()

    # Bare CAS registry numbers (e.g. "56546-36-2") are not compound names.
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", name):
        return ""

    return name


def _to_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _truncate_text(text, max_length=36):
    text = _to_text(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _find_first_column(df, candidates):
    candidate_map = {str(col).lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in candidate_map:
            return candidate_map[col.lower()]
    return None


def _normalize_library_matches_dataframe(df):
    if df is None or len(df) == 0:
        return df

    work_df = df.copy()
    canonical_columns = {
        "#Scan#": ["#Scan#", "scan", "Scan"],
        "MQScore": ["MQScore", "Cosine", "cosine"],
        "Compound_Name": ["Compound_Name", "compound_name", "Compound Name", "CompoundName"],
        "SpectrumFile": ["SpectrumFile", "spectrumfile", "Spectrum File"],
        "SpectrumID": ["SpectrumID", "spectrumid", "Spectrum ID"],
        "library_usi": ["library_usi", "LibraryUSI", "Library_USI", "library usi"],
        "Smiles": ["Smiles", "SMILES", "smiles"],
        "InChIKey": ["InChIKey", "inchikey", "INCHIKEY"],
        "InChIKey-Planar": ["InChIKey-Planar", "inchikey-planar", "INCHIKEY-PLANAR"],
        "INCHI": ["INCHI", "InChI", "inchi"],
        "Library_Class": ["Library_Class", "library_class", "Library Class", "LibraryClass", "libraryclass"],
    }

    for canonical_name, candidates in canonical_columns.items():
        if canonical_name in work_df.columns:
            continue

        matched_column = _find_first_column(work_df, candidates)
        if matched_column is not None:
            work_df = work_df.rename(columns={matched_column: canonical_name})

    work_df = _derive_smiles_from_inchi(work_df)
    work_df = _propagate_structure_by_name(work_df)
    return work_df


def _derive_smiles_from_inchi(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Smiles by converting INCHI via RDKit for rows that have one."""
    if "INCHI" not in df.columns or "Smiles" not in df.columns:
        return df
    _null = {"", "nan", "none", "null", "n/a", "na"}
    missing_smiles = df["Smiles"].astype(str).str.strip().str.lower().isin(_null)
    has_inchi = ~df["INCHI"].astype(str).str.strip().str.lower().isin(_null)
    derived = 0
    for idx in df.index[missing_smiles & has_inchi]:
        try:
            mol = MolFromInchi(str(df.at[idx, "INCHI"]))
            if mol is not None:
                df.at[idx, "Smiles"] = Chem.MolToSmiles(mol)
                derived += 1
        except Exception:
            pass
    if derived:
        logger.debug("_derive_smiles_from_inchi: derived SMILES for %d row(s)", derived)
    return df


_STRUCTURE_FIELDS = ["Smiles", "InChIKey", "InChIKey-Planar", "INCHI"]


def _is_valid_field(value) -> bool:
    """Return True if the value is a non-empty, non-null string."""
    if value is None:
        return False
    s = str(value).strip().lower()
    return s not in ("", "nan", "none", "null", "n/a", "na")


def _smiles_to_planar_inchikey(smiles: str) -> str | None:
    """Convert a SMILES string to a planar (stereo-stripped) InChIKey."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi).split("-")[0]
    except Exception:
        return None


def _inchi_to_planar_inchikey(inchi: str) -> str | None:
    """Convert a full InChI string to a planar (stereo-stripped) InChIKey."""
    try:
        mol = MolFromInchi(inchi)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        inchi_planar = MolToInchi(mol)
        if inchi_planar is None:
            return None
        return InchiToInchiKey(inchi_planar).split("-")[0]
    except Exception:
        return None


def _canonical_structure_key(row):
    inchi_planar = str(row.get("InChIKey-Planar", "")).strip()
    if _is_valid_field(inchi_planar):
        return inchi_planar

    inchikey = str(row.get("InChIKey", "")).strip()
    if _is_valid_field(inchikey):
        return inchikey.split("-")[0]

    smiles = str(row.get("Smiles", "")).strip()
    if _is_valid_field(smiles):
        key = _smiles_to_planar_inchikey(smiles)
        if key:
            return key

    inchi = str(row.get("INCHI", "")).strip()
    if _is_valid_field(inchi):
        key = _inchi_to_planar_inchikey(inchi)
        if key:
            return key

    return "UNKNOWN_STRUCTURE"


def _canonicalize_fragment_smiles(fragment):
    fragment = Chem.Mol(fragment)
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(fragment, canonical=True)


@lru_cache(maxsize=8192)
def _build_positional_isomer_group_key(inchi="", smiles=""):
    inchi = _to_text(inchi)
    smiles = _to_text(smiles)

    mol = None
    if inchi:
        try:
            mol = Chem.MolFromInchi(inchi)
        except Exception:
            mol = None
    if mol is None and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None

    if mol is not None:
        try:
            mol = Chem.Mol(mol)
            Chem.RemoveStereochemistry(mol)
            mol = Chem.RemoveHs(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = ""
            sidechain_fragments = []
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                replaced = Chem.ReplaceCore(mol, scaffold)
                if replaced is not None and replaced.GetNumAtoms() > 0:
                    for fragment in Chem.GetMolFrags(replaced, asMols=True, sanitizeFrags=False):
                        try:
                            Chem.SanitizeMol(fragment)
                        except Exception:
                            pass
                        sidechain_fragments.append(_canonicalize_fragment_smiles(fragment))

            if scaffold_smiles:
                scaffold_formula = rdMolDescriptors.CalcMolFormula(scaffold)
                fragment_key = "|".join(sorted(sidechain_fragments))
                return f"POSISO|{formula}|{scaffold_formula}|{fragment_key}"
        except Exception:
            pass

    return ""


def _propagate_structure_by_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing structural fields for rows whose Compound_Name exactly matches
    another row that already has structure data.

    Only propagates when every structured row in a name group resolves to the
    same canonical planar InChIKey.  If a name maps to multiple different
    structures the group is skipped entirely — an ambiguous synonym should never
    silently receive the wrong structure.

    Fields propagated (when present in the DataFrame):
        Smiles, InChIKey, InChIKey-Planar, INCHI
    """
    if "Compound_Name" not in df.columns:
        return df

    present_fields = [f for f in _STRUCTURE_FIELDS if f in df.columns]
    if not present_fields:
        return df

    work_df = df.copy()

    # Vectorized has_structure: any structure field non-empty/non-null per row.
    _null_sentinel = {"", "nan", "none", "null", "n/a", "na"}
    has_structure = pd.Series(False, index=work_df.index)
    for f in present_fields:
        has_structure |= ~work_df[f].astype(str).str.strip().str.lower().isin(_null_sentinel)

    if not has_structure.any():
        return work_df

    # Pre-compute canonical keys for all structured rows in one apply pass so
    # the groupby loop below never calls iterrows() + _canonical_structure_key
    # per row (iterrows creates a pd.Series per row — O(N) Series allocation).
    structured_df = work_df.loc[has_structure].copy()
    structured_df["_tmp_key"] = structured_df.apply(_canonical_structure_key, axis=1)

    # Pre-compute per-row field-valid counts outside the loop.
    # Doing pandas string chains inside the groupby on tiny per-group slices
    # (avg ~3 rows) would pay per-operation pandas overhead 6 000+ times over.
    # Four full-column passes done here replace 6 000 × 4 tiny-slice passes.
    _field_counts = sum(
        (~structured_df[f].astype(str).str.strip().str.lower().isin(_null_sentinel)).astype(int)
        for f in present_fields
    )

    # Build a per-name source record using case-insensitive matching.
    # Grouping on a lowercased copy keeps the ambiguity guard intact: e.g. "X"
    # with structure A and "x" with structure B land in the same group and are
    # correctly rejected when their canonical keys differ.
    structured_df["_tmp_name_lower"] = (
        structured_df["Compound_Name"].astype(str).str.strip().str.lower()
    )
    name_to_source: dict = {}
    for name_lower, group in structured_df.groupby("_tmp_name_lower"):
        name_lower = str(name_lower).strip()
        if not name_lower or name_lower in {"nan", "none"}:
            continue
        # Require all structured rows in the group to share the same canonical key.
        valid_keys = set(group["_tmp_key"].unique()) - {"UNKNOWN_STRUCTURE"}
        if len(valid_keys) != 1:
            # 0 → all resolved to UNKNOWN (no usable structure); >1 → ambiguous name
            continue
        # Pick the row with the most non-empty structure fields.
        best_idx = _field_counts.loc[group.index].idxmax()
        name_to_source[name_lower] = {f: group.at[best_idx, f] for f in present_fields}

    if not name_to_source:
        return work_df

    # Target: rows missing at least one structure field (not only rows missing all fields).
    # This lets a row that has InChI but no SMILES receive SMILES from a named sibling.
    needs_any_field = pd.Series(False, index=work_df.index)
    for f in present_fields:
        needs_any_field |= work_df[f].astype(str).str.strip().str.lower().isin(_null_sentinel)

    if not needs_any_field.any():
        return work_df

    filled = 0
    for idx in work_df.index[needs_any_field]:
        name = str(work_df.at[idx, "Compound_Name"]).strip().lower()
        source = name_to_source.get(name)
        if source is None:
            continue
        for field, value in source.items():
            # Never overwrite a field that already carries valid data.
            if _is_valid_field(value) and not _is_valid_field(work_df.at[idx, field]):
                work_df.at[idx, field] = value
        filled += 1

    if filled:
        logger.debug("_propagate_structure_by_name: filled structure for %d row(s)", filled)

    return work_df


def _annotate_support_groups(df):
    work_df = df.copy()

    # Deduplicate (INCHI, Smiles) pairs before computing scaffold keys.
    # The lru_cache deduplicates at the RDKit level but the per-row apply
    # still pays Python dispatch + cache-lookup overhead for every cache hit.
    # Building the map from unique pairs and then doing a list-comprehension
    # lookup eliminates that overhead for the ~83% of rows that are cache hits.
    inchi_s = work_df["INCHI"].astype(str) if "INCHI" in work_df.columns else pd.Series("", index=work_df.index)
    smiles_s = work_df["Smiles"].astype(str) if "Smiles" in work_df.columns else pd.Series("", index=work_df.index)
    pair_tuples = list(zip(inchi_s, smiles_s))
    unique_pairs = list(dict.fromkeys(pair_tuples))
    key_map = {
        (inchi, smiles): _build_positional_isomer_group_key(inchi, smiles)
        for inchi, smiles in unique_pairs
    }
    work_df["_positional_group_key"] = [key_map[p] for p in pair_tuples]

    exact_to_positional = {}
    if "_structure_key" in work_df.columns:
        positional_rows = work_df[work_df["_positional_group_key"] != ""]
        if len(positional_rows) > 0:
            exact_to_positional = (
                positional_rows.groupby("_structure_key")["_positional_group_key"]
                .agg(lambda series: series.mode().iloc[0] if len(series.mode()) > 0 else series.iloc[0])
                .to_dict()
            )

    work_df["_support_group_key"] = work_df["_structure_key"].map(exact_to_positional)
    if "_positional_group_key" in work_df.columns:
        work_df["_support_group_key"] = work_df["_support_group_key"].fillna(work_df["_positional_group_key"])
    work_df["_support_group_key"] = work_df["_support_group_key"].fillna("")
    work_df["_support_group_key"] = work_df["_support_group_key"].replace("", pd.NA)
    work_df["_support_group_key"] = work_df["_support_group_key"].fillna(work_df["_structure_key"])

    # Rows with no resolvable structure must not cluster together — each gets a
    # unique sentinel key so they never artificially inflate support counts.
    unknown_mask = work_df["_support_group_key"] == "UNKNOWN_STRUCTURE"
    if unknown_mask.any():
        work_df.loc[unknown_mask, "_support_group_key"] = [
            f"UNKNOWN_STRUCTURE_{i}" for i in work_df.index[unknown_mask]
        ]

    return work_df


def _normalize_library_usi(library_usi):
    library_usi = _to_text(library_usi)
    if not library_usi:
        return ""

    if "mzspec:GNPS:GNPS-LIBRARY:accession:" in library_usi:
        return library_usi

    prefix = "mzspec:GNPS:GNPS-LIBRARY:"
    if library_usi.startswith(prefix):
        suffix = library_usi[len(prefix):]
        if suffix.startswith("accession:"):
            return f"{prefix}{suffix}"
        if suffix.startswith("CCMSLIB"):
            return f"{prefix}accession:{suffix}"

    if library_usi.startswith("CCMSLIB"):
        return f"{prefix}accession:{library_usi}"

    return library_usi


def _infer_resolved_workflow(matches_df, workflow_choice):
    workflow_choice = (workflow_choice or "auto").strip().lower()
    if workflow_choice in SUPPORTED_WORKFLOWS and workflow_choice != "auto":
        return workflow_choice

    spectrum_file_col = _find_first_column(matches_df, ["SpectrumFile"])
    if spectrum_file_col is None:
        return "auto"

    observed_names = {
        _to_text(value).split("/")[-1]
        for value in matches_df[spectrum_file_col].dropna().head(25).tolist()
        if _to_text(value)
    }
    generated_names = {"spectra_filtered.mgf", "spectra_reformatted.mgf"}

    if observed_names and observed_names.isdisjoint(generated_names):
        return "librarysearch"

    return "auto"


def _build_query_usi(task_id, scan, workflow_choice, spectrum_file=""):
    if workflow_choice == "librarysearch":
        spectrum_file = _to_text(spectrum_file).lstrip("/")
        if not spectrum_file:
            return ""

        scan = str(scan)
        return f"mzspec:GNPS2:TASK-{task_id}-inputspectra/{spectrum_file}:scan:{scan}"

    scan = str(scan)
    return f"mzspec:GNPS2:TASK-{task_id}-nf_output/clustering/spectra_reformatted.mgf:scan:{scan}"
