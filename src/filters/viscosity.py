
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

from mixprop.visc_pred_wrapper import visc_pred_single


@dataclass(frozen=True)
class ViscosityPredConfig:
    smiles_col: str = "SMILES"
    miu_col: str = "MIU"

    # MixProp args
    smi2: str = "c1ccccc1"
    molfrac1: float = 1.0
    T: float = 298.0
    n_models: int = 25
    threshold: float = 0.022
    num_workers: int = 0
    check_phase: bool = False
    checkpoint_dir: str = "checkpoints/excess_props/nist_dippr_model/nist_dippr_model"

    # output columns
    pred_col: str = "viscosity_pred"
    rel_col: str = "viscosity_reliability"


def add_viscosity_predictions(
    df: pd.DataFrame,
    *,
    config: ViscosityPredConfig = ViscosityPredConfig(),
    input_path_for_args: str = "",  
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Adds two columns to df:
      - viscosity_pred
      - viscosity_reliability

    The prediction is computed using MixProp `visc_pred_single(args)`.

    Returns:
      df_out: dataframe with added columns
    """
    if config.smiles_col not in df.columns:
        raise KeyError(f"Missing required column: '{config.smiles_col}'")
    if config.miu_col not in df.columns:
        raise KeyError(f"Missing required column: '{config.miu_col}'")

    preds = []
    rels = []

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Predicting viscosity", unit="row")

    for _, row in iterator:
        smi1 = row[config.smiles_col]

        args = {
            "smi1": smi1,
            "smi2": config.smi2,
            "molfrac1": config.molfrac1,
            "T": config.T,
            "n_models": config.n_models,
            "threshold": config.threshold,
            "num_workers": config.num_workers,
            "check_phase": config.check_phase,
            "checkpoint_dir": config.checkpoint_dir,
            "input_path": input_path_for_args, 
        }

        out = visc_pred_single(args)
        preds.append(out[0])
        rels.append(out[1])

    df_out = df.copy()
    df_out[config.pred_col] = preds
    df_out[config.rel_col] = rels

    df_out = df_out[[config.smiles_col, config.miu_col, config.pred_col, config.rel_col]]

    return df_out


def add_viscosity_predictions_from_csv(
    input_csv: str | Path,
    *,
    config: ViscosityPredConfig = ViscosityPredConfig(),
    output_csv: Optional[str | Path] = None,
    input_path_for_args: str = "",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper: load csv -> add predictions -> optionally save.
    """
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    df_out = add_viscosity_predictions(
        df,
        config=config,
        input_path_for_args=input_path_for_args,
        show_progress=show_progress,
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False)

    return df_out


def main():
    import argparse

    p = argparse.ArgumentParser(description="Add MixProp viscosity predictions to a MIU CSV.")
    p.add_argument("--input", required=True, help="Path to input CSV with columns SMILES, MIU")
    p.add_argument("--output", required=True, help="Path to output CSV")
    p.add_argument("--checkpoint_dir", default="checkpoints/excess_props/nist_dippr_model/nist_dippr_model")
    p.add_argument("--smi2", default="c1ccccc1")
    p.add_argument("--T", type=float, default=298.0)
    p.add_argument("--molfrac1", type=float, default=1.0)
    p.add_argument("--n_models", type=int, default=25)
    p.add_argument("--threshold", type=float, default=0.022)
    p.add_argument("--num_workers", type=int, default=0)

    args = p.parse_args()

    cfg = ViscosityPredConfig(
        checkpoint_dir=args.checkpoint_dir,
        smi2=args.smi2,
        T=args.T,
        molfrac1=args.molfrac1,
        n_models=args.n_models,
        threshold=args.threshold,
        num_workers=args.num_workers,
    )

    df_out = add_viscosity_predictions_from_csv(
        args.input,
        config=cfg,
        output_csv=args.output,
        input_path_for_args="",   # optional
        show_progress=True,
    )

    print(f"[done] wrote {args.output} | rows={len(df_out):,}")


if __name__ == "__main__":
    main()