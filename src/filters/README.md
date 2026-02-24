# `src/filters` — Screening & Ranking Modules

This folder contains the **core filtering stages** used in the *Self‑Driving Discovery of Electric Vehicle Coolants* pipeline.  
Each module is designed to be **importable** from notebooks and scripts, and most modules can also be run as a **standalone script** (via a `main()` entry point if present).

> **Typical flow:** `acquisition → smiles_filters → thermo → viscosity → excess_prop → (corrosion/safety) → ester → Vikor`

---

## Folder Contents

```
src/filters/
  acquisition.py
  smiles_filters.py
  thermo.py
  viscosity.py
  excess_prop.py
  ester.py
  corrosion_rf.py
  corrosion_llm.py
  safety.py
  Vikor.py
  pretrained_models/
  __init__.py
```

### Notes on data conventions
- Most stages expect a **pandas DataFrame** with a SMILES column (commonly `smiles` or `SMILES`).
- Intermediate outputs are typically saved as **CSV** or **Parquet** for reproducibility and to support checkpointing.
- Where possible, stages add **new columns** rather than mutating existing ones destructively.

---

## Module Guide

## 1) `acquisition.py` — PubChem data acquisition (Stage 1)
**Purpose:** Download molecular identifiers and basic descriptors from PubChem using the PUG REST API.

**Typical outputs:**
- A “material bank” table containing at least:
  - `smiles` (IsomericSMILES)
  - `name` (IUPACName)
  - `formula` (MolecularFormula)
  - `molecular_weight` (MolecularWeight, if requested)
- Optional **checkpoints** written periodically to avoid losing progress.

**Key technical points**
- Batch queries (CID chunks) are used to reduce API overhead.
- Includes retry/backoff logic and polite sleeps to reduce rate-limit issues.
- Consider running acquisition in ranges (e.g., 1M CID blocks) and merging outputs offline.

---

## 2) `smiles_filters.py` — SMILES validity + exclusions (Stage 0)
**Purpose:** Remove invalid SMILES and optionally exclude specific chemistry classes.

**Typical behavior**
- Validates SMILES with RDKit (`Chem.MolFromSmiles`).
- Optionally excludes:
  - salts / ionic fragments (dot-disconnected structures, charged fragments)
  - metals
  - functional groups via SMARTS (e.g., peroxide, azide, nitro, nitrile, thiol)

**Outputs**
- Either:
  - a filtered DataFrame (pass set)
  - and a rejected DataFrame (fail set), for auditability

**Tip:** For large datasets, avoid per-row `iterrows()` where possible and prefer vectorized or `Series.apply` operations.

---

## 3) `thermo.py` — Thermophysical property models (ML)
**Purpose:** Train and/or run predictors for thermophysical properties (e.g., **density**, **heat capacity**, **boiling point**, **melting point**).

**Typical inputs**
- A CSV file with:
  - SMILES column
  - one or more target columns

**Typical outputs**
- Checkpoints saved to a user-provided directory.
- Prediction tables augmented with:
  - `pred` (mean)
  - `std` (ensemble uncertainty)
  - optional flags like `uncertainty_status`

**Key technical points**
- Models are trained using a message passing neural network (MPNN) stack (Chemprop-style).
- Ensemble prediction is used to estimate epistemic uncertainty via prediction variance/STD.
- GPU usage depends on your PyTorch/CUDA setup.

---

## 4) `viscosity.py` — Dynamic viscosity prediction
**Purpose:** Add **dynamic viscosity predictions** and a **reliability** indicator (from an ensemble) to a DataFrame.

**Typical behavior**
- Reads a CSV (e.g., containing measured `MIU`) and appends:
  - `viscosity_pred`
  - `viscosity_reliability`

**Key technical points**
- Uses a pretrained ensemble (MixProp).
- Reliability is typically computed from ensemble variance relative to a threshold.

**Performance note:** Row-wise prediction can be slow for very large files; batching may be required for scaling.

---

## 5) `excess_prop.py` — Excess property / non-linearity screening
**Purpose:** Determine whether **binary mixtures** show beneficial non-ideal behavior (e.g., strong viscosity reduction vs. mixing rule).

**Typical inputs**
- CSV specifying mixture pairs and conditions (columns commonly: `smi1`, `smi2`, `molfrac1`, `T`).
- Access to pretrained MixProp checkpoints (or a downloaded model bundle).

**Typical outputs**
- Augmented DataFrame with per-fraction model predictions and derived flags, e.g.:
  - `non_linearity` (bool)
  - `excess_prop` (bool)
  - `excess_prop_value` (numeric)
  - `synergysm_to_mixing_rule` (bool)

**Key technical points**
- Evaluates predicted mixture curves at selected mole fractions (e.g., 0.25/0.5/0.75).
- Compares predictions against a linear mixing rule baseline.

---

## 6) `ester.py` — Ester substructure filter
**Purpose:** Detect whether a molecule contains an **ester** group via SMARTS.

**Typical outputs**
- Adds `ester_score` (0/1) or `ester` (bool/int) to the DataFrame.

**SMARTS used (example):**
- Ester: `[#6][C](=O)[O][#6]`

**Tip:** Keep substructure filters **separate and explicit**—they are easy to validate and audit.

---

## 7) `corrosion_rf.py` / `corrosion_llm.py` — Corrosion screening
**Purpose:** Filter candidates predicted to be corrosive (or conversely, retain candidates predicted non-corrosive).

Two alternative approaches may exist:
- **RF model** (`corrosion_rf.py`): classical ML model for corrosion classification/regression.
- **LLM-assisted** (`corrosion_llm.py`): rule-based or language-model–assisted screening (depends on your implementation).

**Recommendation:** Log inputs/outputs carefully—corrosion filtering is often a high-impact decision stage.

---

## 8) `safety.py` — Safety screening (GHS-based)
**Purpose:** Screen out candidates based on safety constraints (e.g., GHS hazard codes).

**Typical inputs**
- A DataFrame with identifiers that can be used to map to hazard statements (often via PubChem or external safety data).

**Typical outputs**
- Candidate pass/fail flags and optionally a table of excluded hazard reasons.

---

## 9) `Vikor.py` — Multi-criteria decision analysis (VIKOR)
**Purpose:** Combine multiple criteria into a single ranking score.

**Typical criteria used**
- `Miu_term` (minimize → inverted normalization)
- `RHO_CP_term` (maximize)
- `non_linear_count` (maximize)
- `ester` (binary constraint)

**Outputs**
- `vikor_score` (lower is better)
- optionally intermediate values (`S`, `R`, normalized criterion scores)

**Key technical points**
- Uses min-max normalization to [0, 1] with safe handling for constant columns.
- Weights and strategy parameter `v` are configurable.

---

## `pretrained_models/`
Holds pretrained weights used by MixProp and/or other predictors.  
If you are distributing this repo publicly, consider whether model files are:
- small enough to store in GitHub, or
- should be downloaded via a script (Zenodo / release assets), or
- stored using Git LFS.

---

## Quick-start Examples

### Import a stage from a notebook
```python
from filters.ester import apply_ester_filter  # example; adjust import to your package layout
df = apply_ester_filter(df)
```

### Run VIKOR scoring
```python
from filters.Vikor import compute_vikor_scores
df_scored = compute_vikor_scores(df, save_path="results_with_vikor.csv")
top = df_scored.sort_values("vikor_score").head(50)
```

---

## Common pitfalls

- **RDKit install:** use conda (`conda install -c conda-forge rdkit`).
- **Lightning mismatch:** if using Chemprop/MPNN training, ensure you import `lightning.pytorch` (not legacy `pytorch_lightning`) when required by your version.
- **Column naming:** be consistent (`smiles` vs `SMILES`). If needed, standardize once at the start of your pipeline.
- **Large file performance:** prefer Parquet for intermediate storage and avoid per-row Python loops when possible.

---

## Where to look next
- `notebooks/` contains step-by-step execution examples and experiments.
- `data/` contains sample inputs/outputs and intermediate artifacts used in the pipeline.
