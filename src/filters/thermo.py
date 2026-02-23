from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# your stack
import torch
torch.set_float32_matmul_precision("high")  # or "medium"

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from chemprop import data, featurizers, models, nn  # adjust if your imports differ


@dataclass(frozen=True)
class ThermoModelConfig:
    prop_name: str                       # e.g., "density", "bp", "mp", "cp"
    smiles_col: str                      # e.g., "smiles"
    target_col: str                      # single target column name in CSV
    test_fraction: float = 0.1
    val_fraction: float = 0.1            # applied on train split
    ensemble_size: int = 5
    epochs: int = 50
    batch_size: int = 64
    seed: int = 42
    accelerator: str = "auto"
    devices: int = 1


def _set_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


def _make_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    *,
    smiles_col: str,
    target_col: str,
):
    # build datapoints
    train_smis = df_train[smiles_col].astype(str).tolist()
    train_y = df_train[target_col].astype(float).to_numpy()

    val_smis = df_val[smiles_col].astype(str).tolist()
    val_y = df_val[target_col].astype(float).to_numpy()

    train_data = [data.MoleculeDatapoint.from_smi(smi, [float(y)]) for smi, y in zip(train_smis, train_y)]
    val_data   = [data.MoleculeDatapoint.from_smi(smi, [float(y)]) for smi, y in zip(val_smis, val_y)]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data, featurizer=featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data, featurizer=featurizer)
    val_dset.normalize_targets(scaler)

    return train_dset, val_dset, scaler


def _build_model(scaler) -> "models.MPNN":
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)
    batch_norm = True
    metric_list = [nn.metrics.MAE(), nn.metrics.RMSE()]  # MAE drives early stopping
    return models.MPNN(mp, agg, ffn, batch_norm, metric_list)


def train_property_ensemble(
    csv_path: str | Path,
    out_dir: str | Path,
    config: ThermoModelConfig,
) -> Dict[str, Path]:
    """
    Train an ensemble for one property. Writes:
      out_dir/<prop_name>/
        - manifest.json
        - model_0.ckpt ... model_{K-1}.ckpt
    Returns dict of checkpoint paths.
    """
    _set_seed(config.seed)

    csv_path = Path(csv_path)
    out_dir = Path(out_dir) / config.prop_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic cleaning: remove rows missing smiles/target
    df = df.dropna(subset=[config.smiles_col, config.target_col]).reset_index(drop=True)

    # Split: train vs test (test kept for your own evaluation)
    train_df, test_df = train_test_split(df, test_size=config.test_fraction, random_state=config.seed)

    # Split train into train/val
    train_df, val_df = train_test_split(train_df, test_size=config.val_fraction, random_state=config.seed)

    # Save split snapshots for reproducibility
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    manifest = {
        "created_at_unix": time.time(),
        "csv_path": str(csv_path),
        "n_total": int(df.shape[0]),
        "n_train": int(train_df.shape[0]),
        "n_val": int(val_df.shape[0]),
        "n_test": int(test_df.shape[0]),
        "config": asdict(config),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ckpts: Dict[str, Path] = {}

    # Build loaders once per model because scaling depends on train subset
    for k in range(config.ensemble_size):
        # Different seed per member (reproducible but diverse)
        _set_seed(config.seed + k)

        train_dset, val_dset, scaler = _make_datasets(
            train_df, val_df,
            smiles_col=config.smiles_col,
            target_col=config.target_col,
        )

        train_loader = data.build_dataloader(train_dset, batch_size=config.batch_size, shuffle=True)
        val_loader = data.build_dataloader(val_dset, batch_size=config.batch_size, shuffle=False)

        mpnn = _build_model(scaler)

        ckpt_cb = ModelCheckpoint(
            dirpath=str(out_dir),
            filename=f"model_{k}",
            monitor="val_loss",
            mode="min",
            save_last=False,
            save_top_k=1,
        )

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator=config.accelerator,
            devices=config.devices,
            max_epochs=config.epochs,
            callbacks=[ckpt_cb],
        )
        trainer.fit(mpnn, train_loader, val_loader)

        best_path = Path(ckpt_cb.best_model_path)
        ckpts[f"model_{k}"] = best_path

    return ckpts


@torch.inference_mode()
def predict_property_ensemble(
    df: pd.DataFrame,
    checkpoint_paths: List[str | Path],
    *,
    smiles_col: str = "smiles",
    batch_size: int = 256,
    accelerator: str = "auto",
    devices: int = 1,
    pred_col: str = "pred",
    std_col: str = "std",
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Predict mean/std using an ensemble of saved checkpoints.
    """
    smis = df[smiles_col].astype(str).tolist()
    test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(test_data, featurizer=featurizer)
    loader = data.build_dataloader(dset, batch_size=batch_size, shuffle=False)

    trainer = pl.Trainer(logger=False, enable_progress_bar=True, accelerator=accelerator, devices=devices)

    preds_all = []
    for p in checkpoint_paths:
        mpnn = models.MPNN.load_from_checkpoint(str(p))
        preds = trainer.predict(mpnn, loader)
        preds = np.concatenate(preds, axis=0).squeeze()
        preds_all.append(preds)

    P = np.vstack(preds_all)  # (K, N)
    out = df.copy()
    out[pred_col] = P.mean(axis=0)
    out[std_col] = P.std(axis=0, ddof=0)
    out.to_csv(output_dir / "predictions.csv", index=False) if output_dir is not None else None
    return out


def apply_uncertainty_filter(
    df: pd.DataFrame,
    *,
    std_col: str = "std",
    threshold: float,
    status_col: str = "uncertainty_status",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into (certain, uncertain) based on ensemble std threshold.
    """
    out = df.copy()
    out[status_col] = np.where(out[std_col] > threshold, "Uncertain", "Certain")
    certain = out[out[status_col] == "Certain"].copy()
    uncertain = out[out[status_col] == "Uncertain"].copy()
    return certain, uncertain