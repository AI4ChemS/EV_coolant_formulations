# src/fambo_filters/excess/mixprop_visc_wrapper.py

import numpy as np
import os
import pandas as pd
from pathlib import Path
import zenodo_get as zget
from zipfile import ZipFile

from mixprop.train import predict
from mixprop.data import MoleculeDataset, MoleculeDataLoader, MoleculeDatapoint
from mixprop.utils import load_args, load_checkpoint, load_scalers

from rdkit import Chem
from tqdm import tqdm
import torch


def download_models():
    """
    Downloads and extracts pretrained models from Zenodo record 8042966.
    """
    zget.zenodo_get(["-r", "8042966"])

    with ZipFile("pretrained_models.zip", "r") as zObject:
        zObject.extractall(path=".")


class mixprop_model:
    def __init__(self, checkpoint_dir):
        self.checkpoints = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(".pt"):
                    fname = os.path.join(root, fname)
                    scalers = load_scalers(fname)
                    self.scaler, self.features_scaler = scalers[0], scalers[1]
                    self.train_args = load_args(fname)
                    device = torch.device("cuda")
                    model = load_checkpoint(fname, device=device)
                    self.checkpoints.append(model)

    def __call__(self, args):
        if args["n_models"] is None:
            args["n_models"] = len(self.checkpoints)

        assert args["n_models"] <= len(self.checkpoints), (
            "Too many models requested. {} models requested.".format(args["n_models"])
        )
        assert args["n_models"] > 1, "Multiple models are needed for reliability analysis."

        model_input = MoleculeDatapoint(
            smiles=[args["smi1"], args["smi2"]],
            features=[args["molfrac1"], args["T"]],
        )
        model_input = MoleculeDataset([model_input])
        model_input_loader = MoleculeDataLoader(dataset=model_input, num_workers=args["num_workers"])

        all_model_preds = []
        for model in self.checkpoints[: args["n_models"]]:
            model_preds = predict(
                model=model,
                data_loader=model_input_loader,
                scaler=self.scaler,
                disable_progress_bar=True,
            )
            all_model_preds.append(model_preds)

        avg_prediction = np.mean(all_model_preds)
        reliability = np.var(all_model_preds) < args["threshold"]
        return avg_prediction, reliability


def load_model(args):
    if "checkpoint_dir" in args.keys():
        return mixprop_model(args["checkpoint_dir"])
    else:
        path = str(Path(__file__).absolute())
        path = "/".join(path.split("/")[:-1])
        checkpoint_dir = os.path.join(path, "pretrained_models/nist_dippr_model/nist_dippr_model")

        print("Loading models from {}".format(checkpoint_dir))

        return mixprop_model(checkpoint_dir)


def onepoint_assertions(args):
    # Requirements:
    assert type(args["smi1"]) == str, "Molecules need to be input as SMILES strings."
    assert type(args["smi2"]) == str, "Molecules need to be input as SMILES strings."
    assert (
        (type(args["T"]) == float)
        | (type(args["T"]) == int)
        | (type(args["T"]) == np.int64)
        | (type(args["T"]) == float)
    ), "Temperature needs to be input as a number."
    assert (
        (type(args["molfrac1"]) == float)
        | (type(args["molfrac1"]) == int)
        | (type(args["T"]) == np.int64)
        | (type(args["T"]) == float)
    ), "Mole fraction needs to be input as a number."

    assert (args["molfrac1"] >= 0.0) & (args["molfrac1"] <= 1.0), "Mole fraction needs to be between 0 and 1."

    mol1 = Chem.MolFromSmiles(args["smi1"])
    mol2 = Chem.MolFromSmiles(args["smi2"])
    assert type(mol1) == Chem.rdchem.Mol, "Invalid SMILES entry, please check that your SMILES are valid."
    assert type(mol2) == Chem.rdchem.Mol, "Invalid SMILES entry, please check that your SMILES are valid."

    # Warnings:
    if ("." in args["smi1"]) | ("." in args["smi2"]):
        print("\nWARNING: Multiple molecules are contained within a single SMILES. Predictions may be unreliable.")

    if (args["T"] < 293) | (args["T"] > 323):
        print("\nWARNING: Temperature is outside of recommended range (293 < T < 323). Predictions may be unreliable.")


def visc_pred_onepoint(model, args):
    # Check input validity:
    onepoint_assertions(args)

    prediction_log, reliability = model(args)
    prediction = 10 ** prediction_log  # Prediction must be converted cP units (without the log)

    return prediction, reliability


def visc_pred_single(args):
    model = load_model(args)
    prediction, reliability = visc_pred_onepoint(model, args)

    return prediction, reliability


def visc_pred_T_curve(args):
    model = load_model(args)

    T_low = 293
    T_high = 323
    interval = 5
    T_vals = np.arange(T_low, T_high + interval, interval)

    preds, rels = [], []
    for T in T_vals:
        T = float(T)
        args["T"] = T
        prediction, reliability = visc_pred_onepoint(model, args)
        preds.append(prediction)
        rels.append(reliability)

    return preds, T_vals, rels


def visc_pred_molfrac1_curve(args):
    model = load_model(args)

    frac_low = 0.0
    frac_high = 1.0
    interval = 0.1
    frac_vals = np.arange(frac_low, frac_high + interval, interval)

    preds, rels = [], []
    for frac in frac_vals:
        frac = float(frac)
        args["molfrac1"] = frac
        prediction, reliability = visc_pred_onepoint(model, args)
        preds.append(prediction)
        rels.append(reliability)

    return preds, frac_vals, rels


def visc_pred_read_csv(args):  # Need additional path argument
    model = load_model(args)

    data = pd.read_csv(args["input_path"])
    cols = data.columns

    preds, rels = [], []
    data["Viscoisty Predictions"] = np.nan
    data["Reliability"] = np.nan

    for i in tqdm(range(len(data)), desc="Processing Data", unit="row"):
        args["smi1"] = data[cols[0]].iloc[i]
        args["smi2"] = data[cols[1]].iloc[i]
        args["molfrac1"] = data[cols[2]].iloc[i]
        args["T"] = data[cols[3]].iloc[i]
        prediction, reliability = visc_pred_onepoint(model, args)
        preds.append(prediction)
        rels.append(reliability)

        data.loc[:i, "Viscoisty Predictions"] = preds
        data.loc[:i, "Reliability"] = rels

        if (i + 1) % 1000 == 0:
            data.to_csv(args["input_path"], index=False)

    return preds, rels, data


def visc_nonlinearity_pred_check(args):
    """
    Predicts the viscosity curve of a binary mixture and checks if any mixture fraction
    has a viscosity at least 0.1 lower than both pure components.

    Returns:
        data (pd.DataFrame): input dataframe with columns appended:
            - model_preds_{frac}
            - mixing_rule_preds_{frac}
            - non_linearity
            - excess_prop
            - excess_prop_value
            - synergysm_to_mixing_rule
    """

    model = load_model(args)

    data = pd.read_csv(args["input_path"])
    cols = data.columns

    for i in tqdm(range(len(data)), desc="Processing Data", unit="row"):
        args["smi1"] = data[cols[0]].iloc[i]
        args["smi2"] = data[cols[1]].iloc[i]

        # Define mole fraction range 
        frac_vals = [0.25, 0.5, 0.75]

        # Predict viscosity at pure fluid fractions (molfrac1 = 0 and molfrac1 = 1)
        args["molfrac1"] = 0.0
        pure_0_visc, _ = visc_pred_onepoint(model, args)

        args["molfrac1"] = 1.0
        pure_1_visc, _ = visc_pred_onepoint(model, args)

        reduced_visc_fracs = []
        syn_value_lst = []
        excess_prop_fracs, syn_prop_fracs = [], []

        for frac in frac_vals:
            frac = float(frac)
            args["molfrac1"] = frac

            prediction, reliability = visc_pred_onepoint(model, args)

            mixing_rule_prediction = pure_0_visc * (1 - frac) + pure_1_visc * frac
            non_lin_metric = min(pure_0_visc, pure_1_visc) - 0.1 * min(pure_0_visc, pure_1_visc)

            excess_miu_metric = ((pure_0_visc * (1 - frac) + pure_1_visc * frac) - prediction) / np.abs(
                pure_1_visc - pure_0_visc
            )
            excess_miu_mixing_rule = 100 * (
                (pure_0_visc * (1 - frac) + pure_1_visc * frac) - prediction
            ) / (pure_0_visc * (1 - frac) + pure_1_visc * frac)

            data.loc[i, f"model_preds_{frac}"] = prediction
            data.loc[i, f"mixing_rule_preds_{frac}"] = mixing_rule_prediction

            if prediction <= non_lin_metric and reliability is True:
                reduced_visc_fracs.append(frac)
                syn_value_lst.append(np.abs(prediction - min(pure_0_visc, pure_1_visc)))

            elif excess_miu_metric >= 0.4 and reliability is True:
                excess_prop_fracs.append(frac)

            if excess_miu_mixing_rule >= 19 or (pure_0_visc * (1 - frac) + pure_1_visc * frac) - prediction > 0.7:
                syn_prop_fracs.append(frac)

        is_nonlinear = bool(reduced_visc_fracs)
        is_excess_prop = bool(excess_prop_fracs)
        is_syn_prop = bool(syn_prop_fracs)

        if is_nonlinear:
            syn_value = min(syn_value_lst)
            syn_index = frac_vals[syn_value_lst.index(syn_value)]
        else:
            syn_value = None
            syn_index = None

        data.loc[i, "non_linearity"] = is_nonlinear
        data.loc[i, "excess_prop"] = is_excess_prop
        data.loc[i, "excess_prop_value"] = syn_value
        data.loc[i, "synergysm_to_mixing_rule"] = is_syn_prop

    return data


def main():
    """
    Example CLI usage:
      python -m fambo_filters.excess.mixprop_visc_wrapper --input examples/example_input.csv --checkpoint_dir pretrained_models/...
    """
    import argparse

    p = argparse.ArgumentParser(description="Run MixProp viscosity-based excess-property screening on a CSV.")
    p.add_argument("--input", required=True, help="CSV with columns: smi1, smi2, molfrac1, T (first 4 columns used).")
    p.add_argument("--checkpoint_dir", default="pretrained_models/nist_dippr_model/nist_dippr_model")
    p.add_argument("--n_models", type=int, default=25)
    p.add_argument("--threshold", type=float, default=0.022)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--T", type=float, default=298.0)

    args_cli = p.parse_args()

    args = {
        "smi1": "O",
        "smi2": "c1ccccc1",
        "molfrac1": 0.25,
        "T": args_cli.T,
        "n_models": args_cli.n_models,
        "threshold": args_cli.threshold,
        "num_workers": args_cli.num_workers,
        "check_phase": False,
        "checkpoint_dir": args_cli.checkpoint_dir,
        "input_path": args_cli.input,
    }

    out_df = visc_nonlinearity_pred_check(args)
    out_path = Path(args_cli.input).with_suffix("").as_posix() + "_mixprop_out.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[done] wrote: {out_path}")


if __name__ == "__main__":
    main()  