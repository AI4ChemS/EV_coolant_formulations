# src/fambo_filters/structural/ester_filter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "RDKit is required. Install via conda:\n"
        "  conda install -c conda-forge rdkit"
    ) from e


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(frozen=True)
class EsterFilterConfig:
    smiles_col: str = "smiles"
    ester_smarts: str = "[#6][C](=O)[O][#6]"
    output_col: str = "ester_score"


# ---------------------------------------------------------
# Core Logic
# ---------------------------------------------------------

def apply_ester_filter(
    df: pd.DataFrame,
    *,
    config: EsterFilterConfig = EsterFilterConfig(),
) -> pd.DataFrame:
    """
    Adds a column `ester_score`:
        1 → molecule contains ester
        0 → no ester

    Returns:
        Updated dataframe (same shape, +1 column)
    """

    if config.smiles_col not in df.columns:
        raise KeyError(f"Missing required column: '{config.smiles_col}'")

    pattern = Chem.MolFromSmarts(config.ester_smarts)
    if pattern is None:
        raise ValueError(f"Invalid SMARTS pattern: {config.ester_smarts}")

    def has_ester(smiles: Optional[str]) -> int:
        if not isinstance(smiles, str):
            return 0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return int(mol.HasSubstructMatch(pattern))

    out = df.copy()
    out[config.output_col] = out[config.smiles_col].apply(has_ester)

    return out