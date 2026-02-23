
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VikorConfig:
    miu_term_col: str = "Miu_term"
    thermo_term_col: str = "RHO_CP_term"
    nonlinear_count_col: str = "non_linear_count"
    ester_binary_col: str = "ester"

    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.30, 0.20)
    v: float = 0.5

    vikor_col: str = "vikor_score"
    keep_intermediates: bool = True


def _minmax01(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    denom = (x.max() - x.min())
    if denom == 0 or np.isclose(denom, 0):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / denom


def compute_vikor_scores(
    df: pd.DataFrame,
    *,
    config: VikorConfig = VikorConfig(),
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Computes VIKOR score and optionally saves dataframe.

    Parameters
    ----------
    df : input dataframe
    config : VikorConfig
    save_path : if provided, dataframe will be saved to this path
                (supports .csv and .parquet)

    Returns
    -------
    dataframe with vikor_score column
    """

    required = [
        config.miu_term_col,
        config.thermo_term_col,
        config.nonlinear_count_col,
        config.ester_binary_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()

    # --- Normalize criteria ---
    miu_score = _minmax01(out[config.miu_term_col])
    miu_score = 1.0 - miu_score  # minimize MIU

    thermo_score = _minmax01(out[config.thermo_term_col])
    nonlinear_score = _minmax01(out[config.nonlinear_count_col])
    ester_score = out[config.ester_binary_col].astype(int)

    scores = np.vstack(
        [
            miu_score.to_numpy(),
            thermo_score.to_numpy(),
            nonlinear_score.to_numpy(),
            ester_score.to_numpy(),
        ]
    ).T

    weights = np.asarray(config.weights, dtype=float)
    weights = weights / weights.sum()

    # --- Ideal best & worst ---
    f_star = np.max(scores, axis=0)
    f_minus = np.min(scores, axis=0)

    denom = f_star - f_minus
    denom = np.where(np.isclose(denom, 0), 1.0, denom)

    dist = weights * (f_star - scores) / denom
    S = np.sum(dist, axis=1)
    R = np.max(dist, axis=1)

    # --- Q index ---
    v = config.v
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    S_term = np.zeros_like(S) if np.isclose(S_minus - S_star, 0) else (S - S_star) / (S_minus - S_star)
    R_term = np.zeros_like(R) if np.isclose(R_minus - R_star, 0) else (R - R_star) / (R_minus - R_star)

    Q = v * S_term + (1 - v) * R_term

    out[config.vikor_col] = Q

    if config.keep_intermediates:
        out["miu_score"] = miu_score
        out["thermo_score"] = thermo_score
        out["nonlinear_score"] = nonlinear_score
        out["ester_score"] = ester_score
        out["S"] = S
        out["R"] = R

    # --- Optional saving ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".parquet":
            out.to_parquet(save_path, index=True)
        else:
            out.to_csv(save_path, index=False)

        print(f"[done] VIKOR results saved to: {save_path}")

    return out