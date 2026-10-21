# `results/`

Outputs of the scripts are written here and are not tracked, except the Materials Project screening output below.

## Screening output

`mp_screening_tda_full_nonmetals/`, written by `../scripts/21_mp_screening_tda_nonmetals.py` with XGBoost models trained by `../scripts/03_train.py`.

| File | Contents |
|---|---|
| `pool_ids.csv` | The screening pool as queried: `mp_id` and `formula` of the 12,503 non-metallic, experimentally reported entries with 2–10 elements and at most 30 sites |
| `pool_meta.json` | Materials Project database version (2026.04.13), query filters and pool size |
| `tda_rerank_full.csv` | All 12,503 entries, sorted by the κ_L predicted by the composition + TDA model. Columns: `mp_id`, `formula`, `nsites`, `kappa_pred_tda_W_mK_300K`, `log_k_pred_tda`, `rank_tda`, `kappa_pred_periodic_W_mK_300K`, `log_k_pred_periodic`, `rank_periodic`, `rank_change`, `in_training_pool` |
| `tda_rerank_top100.csv` | The first 100 rows of `tda_rerank_full.csv` |
| `meta.json` | Database version, cell convention, filters, counts and summary statistics of the run |

`periodic` denotes the composition-only model and `tda` the composition + TDA model; κ_L is in W m⁻¹ K⁻¹ at 300 K. `in_training_pool` marks entries that are also in the Phonix training set, whose predictions are therefore not out-of-sample.

## Not tracked

Trained models (`*.pkl`), metrics files, held-out predictions, the descriptor cache of the screening run and the run logs. Regenerate them with `scripts/03_train.py`, `scripts/35_grouped_nested_cv.py` and `scripts/21_mp_screening_tda_nonmetals.py`.
