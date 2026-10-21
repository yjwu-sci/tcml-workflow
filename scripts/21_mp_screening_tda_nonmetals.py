"""Non-metallic Materials Project screening with the TDA pipeline.

Queries the non-metallic, experimentally reported pool (is_metal = False, theoretical = False,
2 <= nelements <= 10, nsites <= 30), records the Materials Project database version, fetches the
structures, computes periodic + TDA descriptors, predicts kappa_L with the final XGBoost models
and writes the ranked table. Materials Project adds and deprecates entries between database
releases, so the pool is saved as pool_ids.csv together with the version it was queried from,
and a later run reuses that list instead of querying again. Structures are fetched in throttled
chunks and the descriptor cache is appended as it goes, so an interrupted run resumes where it
stopped.

Each Materials Project structure is converted to its conventional standard cell
(pymatgen SpacegroupAnalyzer, symprec 0.01) before the descriptors are computed. The training
descriptors are computed on the cells stored in Phonix, which are conventional cells; the TDA
descriptors are evaluated on a finite 2 x 2 x 2 expansion and depend on the cell setting, so the
screening must use the same convention (MP returns primitive cells for centred lattices).

Run from the tcml-workflow root with MP_API_KEY in the environment:
    set -a; source /path/to/.env; set +a
    python scripts/21_mp_screening_tda_nonmetals.py                 # query (once) + fetch + predict
    python scripts/21_mp_screening_tda_nonmetals.py --query-only    # save the pool list and stop
    python scripts/21_mp_screening_tda_nonmetals.py --predict-only  # reuse the descriptor cache

Outputs in results/mp_screening_tda_full_nonmetals/ (pool_ids.csv, pool_meta.json, the two
ranked CSVs and meta.json are tracked):
    pool_ids.csv  pool_meta.json  features_cache_conventional.csv  unresolved_ids.txt
    tda_rerank_full.csv  tda_rerank_top100.csv  meta.json  run.log
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
OUT = REPO / "results/mp_screening_tda_full_nonmetals"
POOL = OUT / "pool_ids.csv"
POOL_META = OUT / "pool_meta.json"
CACHE = OUT / "features_cache_conventional.csv"   # earlier caches used the raw MP cell; do not mix
CELL_CONVENTION = "pymatgen SpacegroupAnalyzer(symprec=0.01).get_conventional_standard_structure()"
UNRES = OUT / "unresolved_ids.txt"
FILTERS = dict(is_metal=False, theoretical=False, num_elements=(2, 10), num_sites=(1, 30))
PER = ("coef_", "prop_")
TDAP = ("rho_", "ALL_H", "CAT_H", "AN_H", "X_H")
MODELS = {"tda": "results/xgboost_periodic_tda_final_model.pkl",
          "periodic": "results/xgboost_periodic_final_model.pkl"}


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def query_pool():
    from mp_api.client import MPRester
    with MPRester(os.environ["MP_API_KEY"], mute_progress_bars=True) as mpr:
        dbv = str(mpr.get_database_version())
        docs = mpr.materials.summary.search(**FILTERS, fields=["material_id", "formula_pretty"])
    pool = pd.DataFrame({"mp_id": [str(d.material_id) for d in docs],
                         "formula": [d.formula_pretty for d in docs]}).drop_duplicates("mp_id")
    pool = pool.sort_values("mp_id", key=lambda s: s.str.split("-").str[1].astype(int)).reset_index(drop=True)
    pool.to_csv(POOL, index=False)
    json.dump(dict(date=datetime.now().isoformat(timespec="seconds"), mp_database_version=dbv,
                   filters={k: list(v) if isinstance(v, tuple) else v for k, v in FILTERS.items()},
                   n_pool=len(pool)), open(POOL_META, "w"), indent=2)
    log(f"queried Materials Project {dbv}: {len(pool)} entries -> {POOL.relative_to(REPO)}")
    return pool, dbv


def load_pool():
    if not (POOL.exists() and POOL_META.exists()):
        return query_pool()
    dbv = json.load(open(POOL_META))["mp_database_version"]
    pool = pd.read_csv(POOL)
    log(f"pool list {POOL.relative_to(REPO)}: {len(pool)} entries from Materials Project {dbv}")
    return pool, dbv


def conventional(s):
    """Conventional standard cell, the convention of the Phonix structures used in training."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        return SpacegroupAnalyzer(s, symprec=0.01).get_conventional_standard_structure(), True
    except Exception:
        return s, False


def fetch(pool, pool_dbv, chunk=200, pause=3.0, retries=5):
    from mp_api.client import MPRester
    from tcml_workflow.descriptors import compute_tda_descriptors, compute_periodic_dataframe
    done = set(pd.read_csv(CACHE, usecols=["mp_id"]).mp_id) if CACHE.exists() else set()
    unres = set(UNRES.read_text().split()) if UNRES.exists() else set()
    todo = [m for m in pool.mp_id if m not in done and m not in unres]
    form = pool.set_index("mp_id").formula.to_dict()
    log(f"pool {len(pool)} | cached {len(done & set(pool.mp_id))} | unresolved {len(unres)} | to fetch {len(todo)}")
    with MPRester(os.environ["MP_API_KEY"], mute_progress_bars=True) as mpr:
        dbv = str(mpr.get_database_version())
        if dbv != pool_dbv:
            raise SystemExit(f"Materials Project is now {dbv} but the pool list is from {pool_dbv}; "
                             "rerun with --requery to screen one database version")
        for s in range(0, len(todo), chunk):
            ids = todo[s:s + chunk]
            docs = None
            for attempt in range(1, retries + 1):
                try:
                    docs = mpr.materials.search(material_ids=ids, fields=["material_id", "structure"])
                    break
                except Exception as e:
                    wait = 20 * attempt
                    log(f"  chunk {s}: {type(e).__name__}: {str(e)[:120]} -> retry in {wait}s ({attempt}/{retries})")
                    time.sleep(wait)
            if docs is None:
                log(f"  chunk {s}: gave up after {retries} attempts; rerun to resume")
                continue
            st = {str(d.material_id): d.structure for d in docs}
            missing = [m for m in ids if m not in st]
            if missing:
                with open(UNRES, "a") as f:
                    f.write("\n".join(missing) + "\n")
            rows, failed = [], []
            for m in ids:
                if m in st:
                    s_conv, ok = conventional(st[m])
                    rows.append((m, s_conv))
                    if not ok:
                        failed.append(m)
            if failed:
                log(f"  chunk {s}: symmetry analysis failed for {len(failed)} entries, raw MP cell used: {failed}")
            if rows:
                labels = [f"c{i}" for i in range(1, len(rows) + 1)]
                tda = compute_tda_descriptors([(lab, s_.to_ase_atoms()) for lab, (_, s_) in zip(labels, rows)])
                tda = tda.reset_index(drop=True)
                tda.insert(0, "mp_id", [m for m, _ in rows])
                per = compute_periodic_dataframe([form[m] for m in tda.mp_id], index=tda.mp_id.tolist(),
                                                 n_groups=17).reset_index().rename(columns={"index": "mp_id"})
                out = per.merge(tda, on="mp_id", how="left")
                out.insert(1, "nsites", [len(st[m]) for m in out.mp_id])
                out.to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)
            log(f"  {min(s + chunk, len(todo))}/{len(todo)} fetched ({len(missing)} unresolved in chunk)")
            time.sleep(pause)


def predict(pool, dbv):
    from joblib import load
    tr = pd.read_parquet("data/processed/train.parquet")

    def cols_of(tda):
        c = ["Temperature"] if "Temperature" in tr.columns else []
        return c + [x for x in tr.columns if x.startswith(PER) or (tda and x.startswith(TDAP))]

    def prep(X):
        X = X.copy()
        cm = np.nanmax(np.abs(X), axis=0)
        for j in np.where(cm > 1e6)[0]:
            X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
        return np.clip(np.nan_to_num(X), -1e9, 1e9)

    mp = pd.read_csv(CACHE).drop_duplicates("mp_id")
    mp = mp[mp.mp_id.isin(set(pool.mp_id))].reset_index(drop=True)
    out = mp[["mp_id", "nsites"]].merge(pool, on="mp_id", how="left")
    out["in_training_pool"] = out.mp_id.isin(set(tr.mp_id))
    for key, path in MODELS.items():
        cols = cols_of(key == "tda")
        miss = [c for c in cols if c not in mp.columns]
        if miss:
            raise SystemExit(f"{key}: missing feature columns {miss[:5]}")
        pred = load(path).predict(prep(mp[cols].to_numpy(float)))
        out[f"log_k_pred_{key}"] = pred
        out[f"kappa_pred_{key}_W_mK_300K"] = 10 ** pred
    out["rank_tda"] = out.kappa_pred_tda_W_mK_300K.rank(method="min").astype(int)
    out["rank_periodic"] = out.kappa_pred_periodic_W_mK_300K.rank(method="min").astype(int)
    out["rank_change"] = out.rank_periodic - out.rank_tda
    out = out.sort_values("kappa_pred_tda_W_mK_300K").reset_index(drop=True)
    order = ["mp_id", "formula", "nsites", "kappa_pred_tda_W_mK_300K", "log_k_pred_tda", "rank_tda",
             "kappa_pred_periodic_W_mK_300K", "log_k_pred_periodic", "rank_periodic", "rank_change",
             "in_training_pool"]
    out[order].to_csv(OUT / "tda_rerank_full.csv", index=False)
    out[order].head(100).to_csv(OUT / "tda_rerank_top100.csv", index=False)

    k = out.kappa_pred_tda_W_mK_300K
    stats = dict(n=len(out), min=float(k.min()), median=float(k.median()),
                 n_below1=int((k < 1).sum()), n_le5=int((k <= 5).sum()))
    top20 = out.head(20).mp_id.tolist()
    n_unres = len(set(UNRES.read_text().split()) & set(pool.mp_id)) if UNRES.exists() else 0
    meta = dict(date=datetime.now().isoformat(timespec="seconds"), mp_database_version=dbv,
                cell_convention=CELL_CONVENTION, formula_parser="pymatgen Composition",
                n_in_training_pool=int(out.in_training_pool.sum()),
                n_in_training_pool_top20=int(out.head(20).in_training_pool.sum()),
                filters={k: list(v) if isinstance(v, tuple) else v for k, v in FILTERS.items()},
                n_pool=len(pool), n_screened=len(out), n_unresolved=n_unres,
                models=MODELS, stats=stats, top20=top20)
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    log(f"screened {stats['n']} of {len(pool)} | min {stats['min']:.3f} | median {stats['median']:.3f} | "
        f"<1 {stats['n_below1']} | <=5 {stats['n_le5']}")
    log(f"wrote {OUT.relative_to(REPO)}/tda_rerank_full.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-only", action="store_true", help="save the pool list and stop")
    ap.add_argument("--requery", action="store_true", help="query the pool again even if pool_ids.csv exists")
    ap.add_argument("--predict-only", action="store_true", help="reuse the descriptor cache")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool, dbv = query_pool() if args.requery else load_pool()
    if args.query_only:
        raise SystemExit(0)
    if not args.predict_only:
        fetch(pool, dbv)
    predict(pool, dbv)
