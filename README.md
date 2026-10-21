# tcml-workflow

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Structure-sensitive prediction of the lattice thermal conductivity (κ_L) of inorganic crystals from composition descriptors and persistent-homology descriptors of the ion networks.

Companion code for Y.-J. Wu, M.-Y. Hung, Y. Xu and H.-K. Tian, _Topological Data Analysis of Crystal Structures for Polymorph-Sensitive Machine-Learning Prediction of Lattice Thermal Conductivity_ (manuscript submitted).

## What it does

Given a crystal structure, this workflow:
1. Encodes the chemical formula as 34 composition descriptors (17 periodic-table groups × [atom fraction, weighted-average atomic number]).
2. Splits the ions into cation-like and anion-like sets by Pauling electronegativity and builds three point clouds (all ions, cations, anions) from a 2 × 2 × 2 supercell, scaled by the median nearest-neighbour distance.
3. Computes the persistent homology of each point cloud (alpha complex, GUDHI): H0 connected components, H1 loops and H2 voids.
4. Summarises each persistence diagram by 11 statistics (counts and summed lifetime per atom, lifetime mean, maximum and 95th percentile, mean birth and death, lifetime entropy, persistence-image intensity and centroid), and adds 4 cation–anion contrasts and 4 packing-density features: 107 TDA descriptors in total.
5. Trains XGBoost, SVR or GPR on log₁₀ κ_L at 300 K from the Phonix first-principles database with nested cross-validation, and predicts κ_L for new structures or for a whole Materials Project query.

A composition-only model gives every polymorph of a formula the same input; the TDA descriptors give each polymorph its own.

## Install

```bash
git clone https://github.com/yjwu-sci/tcml-workflow.git
cd tcml-workflow
conda create -n tcml python=3.10 -y && conda activate tcml
pip install -e ".[tda,mp,dev]"
```

`[tda]` installs GUDHI and persim, needed for the TDA descriptors. `[mp]` installs `mp-api` for Materials Project screening, which needs an API key:

```bash
export MP_API_KEY="your_key"
```

## Quick start

Compute both descriptor blocks for a structure:

```python
from ase.io import read
from tcml_workflow.descriptors import compute_periodic_descriptors, compute_tda_descriptors

atoms = read("path/to/your.cif")

composition = compute_periodic_descriptors(atoms.get_chemical_formula())
# -> numpy.ndarray, shape (34,)

tda = compute_tda_descriptors([("my-structure", atoms)])
# -> pandas.DataFrame, one row per structure: the 107 TDA descriptors
#    plus bookkeeping columns (cif_file, n_all, n_cat, n_an, anion_elements, ...)
```

`compute_tda_descriptors` takes an iterable of `(material_id, ase.Atoms)` pairs, so the same call batches over a dataset.

To predict κ_L, train a model once (no trained model is shipped: pickled estimators execute code when loaded and are tied to specific library versions):

```bash
python scripts/01_download_phonix.py --out_dir data/phonix          # Phonix summary, 149 MB
python scripts/02_compute_descriptors.py --phonix data/phonix/summary.parquet \
       --out_dir data/processed --tda
python scripts/03_train.py --algorithm XGBoost --train data/processed/train.parquet --out_dir results
```

Then predict your own structures:

```bash
python scripts/05_predict_candidates.py --model results/xgboost_periodic_tda_final_model.pkl \
       --candidates data/candidates_example.csv --out_dir results/candidate_predictions
```

The candidates file has one structure per row (`label`, `formula`, `cif_path`, and optionally a reference value `dft_kappa_total`); the output `predictions.csv` gives κ_L at 300 K in W m⁻¹ K⁻¹. `notebooks/00_quickstart.ipynb` walks through download, descriptors, training and evaluation on a 200-material subset.

`scripts/03_train.py` also trains SVR and GPR (`--algorithm SVR` or `GPR`; use `--subsample 1500` for GPR) and the composition-only model (`--no_tda`). `scripts/04_evaluate.py --results_dir results` collects the metrics of all trained models into one table.

## Screening Materials Project

```bash
python scripts/03_train.py --algorithm XGBoost --train data/processed/train.parquet --out_dir results --no_tda
python scripts/21_mp_screening_tda_nonmetals.py      # needs MP_API_KEY and both XGBoost models
```

The script queries the non-metallic, experimentally reported entries with 2–10 elements and at most 30 sites, converts each structure to its conventional standard cell (the cell setting of the Phonix training structures), and writes a table ranked by the predicted κ_L of the composition + TDA model, with the composition-only prediction alongside. The pool is saved with the database version it was queried from, and an interrupted run resumes from its descriptor cache (`--predict-only` reuses the cache, `--requery` queries again).

`results/mp_screening_tda_full_nonmetals/` holds the output for Materials Project database version 2026.04.13: 12,503 structures, with `in_training_pool` marking the entries that are also in the Phonix training set (see [`results/README.md`](results/README.md)).

## Evaluation grouped by chemical formula

`scripts/03_train.py` uses random folds. For folds in which no chemical formula, and hence no pair of polymorphs, is on both sides of a split:

```bash
python scripts/27_database_structure_variables.py   # atoms per primitive cell, volume per atom, density
python scripts/35_grouped_nested_cv.py              # every descriptor set and algorithm; several hours
```

`scripts/35` compares three descriptor sets (composition only; composition + database structure, i.e. the three variables above and the space group; composition + TDA) and writes, for every run, the metrics and the held-out prediction of every material. For the XGBoost runs it also reports the share of same-composition pairs whose two polymorphs are predicted in the Phonix order.

Both protocols use 5 outer × 10 inner folds, `random_state = 0`, and the hyperparameter search spaces in `tcml_workflow/models.py`.

## Scope

- The models predict the lattice contribution κ_L at 300 K. Phonix contains no electronic contribution, so screening is restricted to non-metallic structures.
- The TDA descriptors are computed on a finite supercell and depend on the cell setting: training and prediction structures must use the same convention (here the conventional standard cell).
- The predictions are meant for ranking candidates before first-principles calculation.

## Data

Training data: [Phonix](https://phonix-db.org), downloaded by `scripts/01` from Hugging Face (`phonix-db/phonix-summary`); the shipped screening output used release v2026.03.28. Screening structures: [Materials Project](https://materialsproject.org). Both are used under CC BY 4.0.

## Tests

`pytest tests/` covers the data loader, the composition descriptors and the model registry; GitHub Actions runs it on Python 3.10 and 3.11.

## Citation

```bibtex
@unpublished{wu2026tcml,
  title  = {Topological Data Analysis of Crystal Structures for Polymorph-Sensitive
            Machine-Learning Prediction of Lattice Thermal Conductivity},
  author = {Wu, Yen-Ju and Hung, Ming-Yuan and Xu, Yibin and Tian, Hong-Kang},
  year   = {2026},
  note   = {manuscript submitted}
}
```

## License

MIT — see [LICENSE](LICENSE).
