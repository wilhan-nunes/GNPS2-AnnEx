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

    return work_df


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
                fragment_key = "|".join(sorted(sidechain_fragments))
                return f"POSISO|{formula}|{scaffold_smiles}|{fragment_key}"
        except Exception:
            pass

    return ""


def _annotate_support_groups(df):
    work_df = df.copy()
    work_df["_positional_group_key"] = work_df.apply(
        lambda row: _build_positional_isomer_group_key(
            row.get("INCHI", ""),
            row.get("Smiles", ""),
        ),
        axis=1,
    )

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
