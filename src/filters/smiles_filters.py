
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from rdkit import Chem
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RDKit is required for SMILES filtering. Install via conda (recommended):\n"
        "  conda install -c conda-forge rdkit\n"
        "or your project’s environment file."
    ) from e


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class SmilesFilterConfig:
    """
    Stage-0 SMILES filtering configuration.
    """
    smiles_col: str = "smiles"

    # Exclude if molecule contains any of these element symbols
    exclude_metals: bool = True

    # Exclude if multi-fragment + ionic (heuristic)
    exclude_salts: bool = True

    # Exclude if any of these functional groups are present (SMARTS)
    exclude_functional_groups: bool = True

    # SMARTS patterns for functional group exclusion
    functional_groups_smarts: Tuple[Tuple[str, str], ...] = (
        ("peroxide", "[OX2][OX2]"),
        ("azide", "[N-]=[N+]=N"),
        ("nitro", "[N+](=O)[O-]"),
        ("nitrile", "[CX2]#N"),
        ("thiol", "[SH]"),
    )


# Metals list (as in your code; you can move to a data file later)
DEFAULT_METALS: Tuple[str, ...] = (
    "Li", "Be", "Na", "Mg", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
)


# -----------------------------
# Helpers
# -----------------------------
def _mol_from_smiles(smiles: str) -> Optional["Chem.Mol"]:
    """
    Parse SMILES into RDKit Mol. Returns None if invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _contains_metal(mol: "Chem.Mol", metals: Sequence[str]) -> bool:
    metal_set = set(metals)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_set:
            return True
    return False


def _is_salt_heuristic(smiles: str, mol: Optional["Chem.Mol"]) -> bool:
    """
    Heuristic salt/ionic mixture detection:
      - SMILES contains '.' (multiple fragments), AND
      - at least one atom has nonzero formal charge
    """
    if "." not in smiles:
        return False
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return True
    return False


def _compile_smarts(smarts_items: Iterable[Tuple[str, str]]) -> Dict[str, "Chem.Mol"]:
    compiled: Dict[str, Chem.Mol] = {}
    for name, smarts in smarts_items:
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            raise ValueError(f"Invalid SMARTS for '{name}': {smarts}")
        compiled[name] = q
    return compiled


# -----------------------------
# Public API
# -----------------------------
def filter_smiles_validity(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    reason_col: str = "reason_code",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep rows with valid RDKit-parsable SMILES.
    Returns (passed_df, excluded_df). Excluded rows get reason_code='invalid_smiles'.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"Missing required column: '{smiles_col}'")

    reasons: List[Optional[str]] = []
    for s in df[smiles_col].astype(str).tolist():
        mol = _mol_from_smiles(s)
        reasons.append(None if mol is not None else "invalid_smiles")

    reason_series = pd.Series(reasons, index=df.index, name=reason_col)
    passed_mask = reason_series.isna()

    passed = df.loc[passed_mask].copy()
    excluded = df.loc[~passed_mask].copy()
    excluded[reason_col] = reason_series.loc[~passed_mask].values

    return passed, excluded


def filter_smiles_exclusions(
    df: pd.DataFrame,
    *,
    config: SmilesFilterConfig = SmilesFilterConfig(),
    metals: Sequence[str] = DEFAULT_METALS,
    reason_col: str = "reason_code",
    use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply chemistry-based exclusions:
      - metals
      - salts (heuristic)
      - functional groups (SMARTS)

    Returns (passed_df, excluded_df). Excluded rows get a reason_code such as:
      - 'contains_metal'
      - 'salt_or_ionic_mixture'
      - 'contains_fg:<name>'
      - 'invalid_smiles'  (if SMILES cannot be parsed)
    """
    smiles_col = config.smiles_col
    if smiles_col not in df.columns:
        raise KeyError(f"Missing required column: '{smiles_col}'")

    fg_queries: Dict[str, Chem.Mol] = {}
    if config.exclude_functional_groups:
        fg_queries = _compile_smarts(config.functional_groups_smarts)

    # Optional tqdm progress
    iterator = df.iterrows()
    if use_tqdm:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(df), desc="Stage-0 SMILES exclusions")
        except Exception:
            pass  # tqdm not available; fall back silently

    reasons: List[Optional[str]] = []
    for _, row in iterator:
        smiles = row[smiles_col]
        smiles = "" if smiles is None else str(smiles)

        mol = _mol_from_smiles(smiles)
        if mol is None:
            reasons.append("invalid_smiles")
            continue

        if config.exclude_metals and _contains_metal(mol, metals):
            reasons.append("contains_metal")
            continue

        if config.exclude_salts and _is_salt_heuristic(smiles, mol):
            reasons.append("salt_or_ionic_mixture")
            continue

        if config.exclude_functional_groups:
            hit_name: Optional[str] = None
            for name, q in fg_queries.items():
                if mol.HasSubstructMatch(q):
                    hit_name = name
                    break
            if hit_name is not None:
                reasons.append(f"contains_fg:{hit_name}")
                continue

        reasons.append(None)

    reason_series = pd.Series(reasons, index=df.index, name=reason_col)
    passed_mask = reason_series.isna()

    passed = df.loc[passed_mask].copy()
    excluded = df.loc[~passed_mask].copy()
    excluded[reason_col] = reason_series.loc[~passed_mask].values

    return passed, excluded


def run_stage0_smiles_filter(
    df: pd.DataFrame,
    *,
    config: SmilesFilterConfig = SmilesFilterConfig(),
    metals: Sequence[str] = DEFAULT_METALS,
    reason_col: str = "reason_code",
    use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: validity + exclusions in one step.
    Returns combined excluded rows with a reason_code.
    """
    # 1) validity
    passed1, excluded1 = filter_smiles_validity(df, smiles_col=config.smiles_col, reason_col=reason_col)

    # 2) exclusions (metals/salts/FG)
    passed2, excluded2 = filter_smiles_exclusions(
        passed1,
        config=config,
        metals=metals,
        reason_col=reason_col,
        use_tqdm=use_tqdm,
    )

    # combine excluded
    excluded_all = pd.concat([excluded1, excluded2], axis=0, ignore_index=False)
    return passed2, excluded_all