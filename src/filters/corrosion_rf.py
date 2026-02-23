import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem

import mordred
import mordred.descriptors

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Feature utilities
# -----------------------------
def replace_string_with_nan(value):
    s = str(value)
    if "a" in s:
        return np.nan
    elif "b" in s:
        return np.nan
    return value


def smiles_to_mols(smiles_list):
    return [rdkit.Chem.MolFromSmiles(s) for s in smiles_list]


def mordred_features(smiles_list, ignore_3D=True):
    calc = mordred.Calculator(mordred.descriptors, ignore_3D=ignore_3D)
    mols = smiles_to_mols(smiles_list)
    feats = calc.pandas(mols)
    return feats


def clean_train_features(features_df: pd.DataFrame):
    features_df = features_df.applymap(replace_string_with_nan)

    zero_columns = features_df.columns[features_df.eq(0).all()]
    features_df = features_df.drop(columns=zero_columns)

    features_cleaned = features_df.dropna(axis=1)

    kept_columns = list(features_cleaned.columns)
    return features_cleaned, kept_columns


def align_test_features(test_features_df: pd.DataFrame, kept_columns):
    test_features_df = test_features_df.applymap(replace_string_with_nan)

    missing = [c for c in kept_columns if c not in test_features_df.columns]
    for c in missing:
        test_features_df[c] = np.nan

    test_aligned = test_features_df[kept_columns]
    return test_aligned


# -----------------------------
# Train
# -----------------------------
def train_rf_from_csv(
    train_csv_path: str,
    smiles_col: str = "smiles",
    label_col: str = "corrosion_status",
    n_estimators: int = 100,
    random_state: int = 42,
    ignore_3D: bool = True,
):
    train_df = pd.read_csv(train_csv_path)

    feats = mordred_features(train_df[smiles_col].astype(str).tolist(), ignore_3D=ignore_3D)
    feats_clean, kept_columns = clean_train_features(feats)

    X = feats_clean.values
    y = train_df[label_col].values

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)

    return rf, kept_columns


# -----------------------------
# Predict + Save
# -----------------------------
def predict_rf_and_save(
    rf_model,
    kept_columns,
    test_csv_path: str,
    output_csv_path: str,
    smiles_col: str = "smiles",
    pred_col_name: str = "corr_pred",
    ignore_3D: bool = True,
):
    """
    Predict on test CSV, append predictions as new column,
    and save to user-specified output path.
    """
    test_df = pd.read_csv(test_csv_path)

    test_feats = mordred_features(test_df[smiles_col].astype(str).tolist(), ignore_3D=ignore_3D)
    test_aligned = align_test_features(test_feats, kept_columns)

    X_test = test_aligned.values
    y_pred = rf_model.predict(X_test)

    # Add predictions as new column
    test_df[pred_col_name] = y_pred

    # Save to specified path
    test_df.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to: {output_csv_path}")

    return test_df, y_pred


# -----------------------------
# Optional evaluation
# -----------------------------
def evaluate_predictions(y_true, y_pred, verbose=True):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    if verbose:
        print("Accuracy:", acc)
        print("Confusion matrix:\n", cm)
        print("Classification report:\n", report)

    return {"accuracy": acc, "confusion_matrix": cm, "report": report}