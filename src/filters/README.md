# `src/filters` --- Screening and Ranking Modules

This directory contains the core computational modules that implement
the multi-stage screening pipeline described in:

> **Self-Driving Discovery of Electric Vehicle Coolants**

The modules are designed to be modular, reproducible, and easily
callable from notebooks or scripts. Each stage performs a well-defined
transformation of candidate molecules, progressively narrowing the
search space toward high-performance coolant candidates.

------------------------------------------------------------------------

## Overview of Modules

### `acquisition.py`

PubChem data acquisition using the REST API with batching and
checkpointing.

### `smiles_filters.py`

SMILES validation and chemical exclusion rules using RDKit.

### `thermo.py`

Machine-learning models for thermophysical property prediction (density,
heat capacity, boiling point, melting point), including ensemble-based
uncertainty estimation.

### `viscosity.py`

Dynamic viscosity prediction using pretrained ensemble models (MixProp),
with reliability indicators.

### `excess_prop.py`

Evaluation of mixture non-linearity and excess-property behavior
relative to linear mixing rules.

### `ester.py`

SMARTS-based ester substructure detection (`ester_score`).

### `corrosion_rf.py` / `corrosion_llm.py`

Corrosion screening via machine-learning or rule-based approaches.

### `safety.py`

Safety and hazard-based candidate filtering.

### `Vikor.py`

Multi-criteria decision analysis (VIKOR) for final candidate ranking.

------------------------------------------------------------------------

## Design Principles

-   **Modular architecture** --- each filter can operate independently.
-   **Reproducible workflow** --- intermediate outputs can be
    checkpointed.
-   **Extensible structure** --- new models or filters can be integrated
    easily.
-   **Consistent data handling** --- DataFrame-based inputs with
    appended evaluation columns.

------------------------------------------------------------------------

## Typical Execution Order

1.  Acquisition\
2.  SMILES filtering\
3.  Thermophysical prediction\
4.  Viscosity prediction\
5.  Excess-property screening\
6.  Corrosion and safety screening\
7.  VIKOR ranking

This structured funnel progressively refines the chemical search space
toward optimal electric vehicle coolant candidates.
