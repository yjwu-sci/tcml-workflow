# Changelog

## v1.0.0

First public release.

- Composition (34-dim) and ion-site TDA (107-dim) descriptors
- Nested-CV training of XGBoost, SVR and GPR on the Phonix κ_L database, with random folds and with folds grouped by chemical formula
- Prediction of κ_L for user-supplied structures
- Materials Project screening, and its ranked output for 12,503 non-metallic structures (database version 2026.04.13)

Trained models are not shipped: unpickling executes code and the files are tied to specific scikit-learn and xgboost versions. `scripts/03_train.py` regenerates them. Phonix and Materials Project data are fetched from their sources by the scripts.
