import pathlib

import pandas as pd
import pytest

# ── molecule constants used across test modules ───────────────────────────────
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
THEOPHYLLINE_SMILES = "CN1C2=C(C(=O)N(C1=O)C)NC=N2"
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
# Leu-Phe — two stereoisomers that should map to the same planar key
LEU_PHE_R_SMILES = "CC(C)C[C@@H](C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O)N"
LEU_PHE_S_SMILES = "CC(C)C[C@H](C(=O)N[C@H](CC1=CC=CC=C1)C(=O)O)N"

DATA_DIR = pathlib.Path(__file__).parent.parent  # tests/ → project root


# ── shared DataFrames ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_df():
    """Five real library hits for scan 16050 (loaded from small_exert.csv)."""
    return pd.read_csv(DATA_DIR / "small_exert.csv")


@pytest.fixture
def minimal_df():
    """Two scans with only the two required columns (scan + cosine)."""
    return pd.DataFrame({
        "#Scan#": ["1", "1", "2"],
        "MQScore": [0.9, 0.85, 0.7],
    })


@pytest.fixture
def single_scan_df():
    """One scan: three hits that all agree on the same structure (caffeine)."""
    return pd.DataFrame({
        "#Scan#": ["42", "42", "42"],
        "MQScore": [0.95, 0.90, 0.88],
        "Smiles": [CAFFEINE_SMILES] * 3,
        "Compound_Name": ["Caffeine"] * 3,
        "InChIKey-Planar": ["RYYVLZVUVIJVGH"] * 3,
    })


@pytest.fixture
def two_group_scan_df():
    """One scan: 2 caffeine hits + 1 theophylline hit — two distinct support groups."""
    return pd.DataFrame({
        "#Scan#": ["99", "99", "99"],
        "MQScore": [0.92, 0.88, 0.45],
        "Smiles": [CAFFEINE_SMILES, CAFFEINE_SMILES, THEOPHYLLINE_SMILES],
        "Compound_Name": ["Caffeine", "Caffeine", "Theophylline"],
        "InChIKey-Planar": ["RYYVLZVUVIJVGH", "RYYVLZVUVIJVGH", "ZFXYFBGIUFBOJW"],
    })


@pytest.fixture
def mixed_structure_df():
    """Two scans with different levels of structure agreement.

    Scan 10: 3 hits — 2 caffeine + 1 theophylline  (partial agreement)
    Scan 20: 2 hits — 2 leu-phe                    (full agreement)
    """
    return pd.DataFrame({
        "#Scan#": ["10", "10", "10", "20", "20"],
        "MQScore": [0.90, 0.85, 0.30, 0.75, 0.60],
        "Smiles": [
            CAFFEINE_SMILES, CAFFEINE_SMILES, THEOPHYLLINE_SMILES,
            LEU_PHE_R_SMILES, LEU_PHE_R_SMILES,
        ],
        "Compound_Name": [
            "Caffeine", "Caffeine", "Theophylline",
            "Leu-Phe", "Leu-Phe",
        ],
        "InChIKey-Planar": [
            "RYYVLZVUVIJVGH", "RYYVLZVUVIJVGH", "ZFXYFBGIUFBOJW",
            "KFKWRHQBZQICHA", "KFKWRHQBZQICHA",
        ],
    })
