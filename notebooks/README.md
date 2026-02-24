# `notebooks/` --- Reproducible Pipeline Execution

This directory contains the step-by-step Jupyter notebooks used to
execute and reproduce the full screening workflow described in:

> **Self-Driving Discovery of Electric Vehicle Coolants**

Each notebook corresponds to a specific stage of the discovery funnel.\
The numbering reflects the intended execution order.

------------------------------------------------------------------------

## Execution Order

### `01_acquire_pubchem.ipynb`

Automated acquisition of molecular data from PubChem.\
Generates the initial material bank used in downstream screening.

### `02_smiles_based_filters.ipynb`

Validation and structural filtering of candidate molecules using RDKit.

### `03_thermophysical_props.ipynb`

Training and/or inference of machine-learning models for thermophysical
property prediction (density, heat capacity, boiling point, melting
point).

### `04_corrosion.ipynb`

Corrosion risk prediction using machine-learning or rule-based models.

### `05_safety.ipynb`

Safety screening based on hazard and regulatory criteria.

### `06_excess_prop.ipynb`

Evaluation of binary mixture non-linearity and excess-property behavior.

### `07_ester.ipynb`

Substructure screening for ester-containing molecules.

### `08_vikor.ipynb`

Multi-criteria ranking using the VIKOR decision-making framework.

### `09_filtering_process.ipynb`

End-to-end demonstration of the complete filtering pipeline.

------------------------------------------------------------------------

## Checkpoints

The `checkpoints/` directory stores intermediate model weights and
artifacts generated during training or evaluation.\
Large checkpoint files may require Git LFS or external storage if
distributed publicly.

------------------------------------------------------------------------

## Reproducibility Notes

-   Notebooks are intended to be executed sequentially.
-   Intermediate outputs are saved to disk to allow resuming without
    recomputation.
-   For large-scale screening, converting notebooks into scripts is
    recommended for improved scalability and automation.

------------------------------------------------------------------------

## Recommended Usage

For quick exploration: 1. Run notebooks sequentially from `01` to `09`.
2. Inspect intermediate outputs after each stage. 3. Use
`09_filtering_process.ipynb` for a consolidated pipeline demonstration.

For production-scale screening: - Prefer calling the corresponding
modules in `src/filters/` directly. - Use batch processing and
checkpointing to manage large datasets.
