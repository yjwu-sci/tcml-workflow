"""Nested cross-validation grouped by chemical formula, for every descriptor set and algorithm.

Same protocol as scripts/03_train.py (tcml_workflow.pipeline.NestedCVPipeline, 5 outer x 10 inner folds,
RandomizedSearchCV 1,000 iterations for XGBoost and 200 for SVR/GPR, random_state 0, same feature
sanitisation, GPR on the same stratified 1,500-entry subsample), except that the outer AND inner folds are
GroupKFold splits by chemical formula (shuffled, random_state 0): no formula, and hence no polymorph pair,
is on both sides of any split.

Runs (resumable: a run whose metrics file exists is skipped):
  xgboost_periodic, xgboost_periodic_slack_sg, xgboost_periodic_tda,
  svr_periodic, svr_periodic_tda, gpr_periodic_n1500, gpr_periodic_tda_n1500
periodic = 34 composition descriptors; tda = + 107 TDA descriptors; slack_sg = + atoms in the primitive cell,
volume per atom, density (results/database_structure/database_structure_variables.csv) and space-group number.

Outputs in OUT_DIR (default results/grouped_nested): <tag>_metrics.json, <tag>_oof.csv (outer-fold predictions),
<tag>_final_model.pkl (not used downstream), and summary.json with the metrics of every run and, for the
XGBoost runs, the share of same-composition pairs whose two polymorphs are predicted in the Phonix order.
  N_ITER=<k>  overrides the search budget (smoke test only)
Run from the repository root with the `tcml` conda env; cores via LOKY_MAX_CPU_COUNT.
"""
from __future__ import annotations
import itertools, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
import sklearn, xgboost
from sklearn.model_selection import GroupKFold, train_test_split

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from tcml_workflow import ModelConfig, NestedCVPipeline

OUT = Path(os.environ.get("OUT_DIR", REPO / "results/grouped_nested")).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
N_ITER = os.environ.get("N_ITER")
t0 = time.time()
PER = ("coef_", "prop_")
TDAP = ("rho_", "ALL_H", "CAT_H", "AN_H", "X_H")
DB = ["n_atoms_prim", "vol_per_atom", "density", "spg_number"]


def log(msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')} | {time.time() - t0:7.0f} s] {msg}"
    print(line, flush=True)
    with open(OUT / "progress.txt", "a") as f:
        f.write(line + "\n")


tr = pd.read_parquet("data/processed/train.parquet")
tr = tr.merge(pd.read_csv("results/database_structure/database_structure_variables.csv")[["mp_id"] + DB[:3]],
              on="mp_id", how="left", validate="one_to_one")
comp = [c for c in tr.columns if c.startswith(PER)]
tda = [c for c in tr.columns if c.startswith(TDAP)]
assert len(comp) == 34 and len(tda) == 107 and tr[DB].notna().all().all()
FEATS = {"periodic": comp, "periodic_slack_sg": comp + DB, "periodic_tda": comp + tda}
RUNS = [("XGBoost", "periodic", None), ("XGBoost", "periodic_slack_sg", None), ("XGBoost", "periodic_tda", None),
        ("SVR", "periodic", None), ("SVR", "periodic_tda", None),
        ("GPR", "periodic", 1500), ("GPR", "periodic_tda", 1500)]


def sanitise(X):   # identical to scripts/03_train.py
    X = X.copy()
    col_max = np.nanmax(np.abs(X), axis=0)
    for j in np.where(col_max > 1e6)[0]:
        X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
    return np.clip(X, -1e9, 1e9)


def accuracy(d, ratio):   # same-composition pairs whose kappa differs by more than `ratio`
    n = c = 0
    for _, g in d.groupby("formula"):
        if len(g) < 2:
            continue
        v = g[["k", "p"]].to_numpy()
        for i, j in itertools.combinations(range(len(g)), 2):
            ki, kj, pi, pj = v[i, 0], v[j, 0], v[i, 1], v[j, 1]
            if max(ki, kj) / min(ki, kj) < ratio:
                continue
            n += 1
            if pi != pj and (ki < kj) == (pi < pj):
                c += 1
    return (c / n * 100 if n else float("nan")), n


for algo, feat, sub_n in RUNS:
    tag = f"{algo.lower()}_{feat}" + (f"_n{sub_n}" if sub_n else "")
    if (OUT / f"{tag}_metrics.json").exists():
        log(f"skip (done) {tag}")
        continue
    sub = tr[FEATS[feat] + ["log_k", "mp_id", "formula"]].dropna()
    assert len(sub) == len(tr)
    if sub_n is not None and sub_n < len(sub):   # identical to scripts/03_train.py --subsample
        strata = pd.qcut(sub["log_k"], q=5, labels=False, duplicates="drop")
        sub, _ = train_test_split(sub, train_size=sub_n, random_state=0, stratify=strata)
        sub = sub.reset_index(drop=True)
    X = sanitise(sub[FEATS[feat]].to_numpy(float))
    y = sub["log_k"].to_numpy(float)
    groups = sub["formula"].to_numpy()
    cfg = ModelConfig(algorithm=algo, random_state=0, **({"n_iter_search": int(N_ITER)} if N_ITER else {}))
    # leakage check: no formula on both sides of any outer split, nor of any inner split of the first outer fold
    for a, b in GroupKFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.random_state).split(X, y, groups):
        assert not set(groups[a]) & set(groups[b])
    a0, _ = next(GroupKFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.random_state).split(X, y, groups))
    for ia, ib in GroupKFold(n_splits=cfg.inner_splits, shuffle=True, random_state=cfg.random_state).split(X[a0], y[a0], groups[a0]):
        assert not set(groups[a0][ia]) & set(groups[a0][ib])
    log(f"start {tag}: n={len(y)}, features={X.shape[1]}, search={cfg.n_iter_search}, formulas={len(set(groups))}")
    pipe = NestedCVPipeline(cfg)
    m = pipe.fit(X, y, groups=groups)
    assert np.isfinite(pipe.oof_pred).all()
    pd.DataFrame({"mp_id": sub["mp_id"], "formula": sub["formula"], "log_k": y, "pred": pipe.oof_pred}).to_csv(
        OUT / f"{tag}_oof.csv", index=False)
    pipe.save(OUT, tag)   # writes <tag>_metrics.json last-but-one; resume key
    log(f"done {tag}: R2 {m.r2_mean:.4f} +- {m.r2_std:.4f}, R {m.r_mean:.4f}, RMSE {m.rmse_mean:.4f}")

# ---------------- summary ----------------
summary = dict(metrics={}, polymorph_ordering={})
for algo, feat, sub_n in RUNS:
    tag = f"{algo.lower()}_{feat}" + (f"_n{sub_n}" if sub_n else "")
    mm = json.load(open(OUT / f"{tag}_metrics.json"))
    summary["metrics"][tag] = {k: mm[k] for k in ("r_mean", "r_std", "r2_mean", "r2_std", "rmse_mean", "rmse_std")}
    if algo == "XGBoost":
        d = pd.read_csv(OUT / f"{tag}_oof.csv").rename(columns={"pred": "p"})
        d["k"] = 10 ** d["log_k"]
        summary["polymorph_ordering"][tag] = {lab: dict(zip(("accuracy", "n_pairs"), accuracy(d, r)))
                                  for lab, r in (("all", 1.0), ("gt20pct", 1.2), ("gt2x", 2.0))}
summary["provenance"] = dict(script=Path(__file__).name, finished_at=datetime.now(timezone.utc).isoformat(),
                             cpu_count=os.cpu_count(),
                             loky_max_cpu_count=os.environ.get("LOKY_MAX_CPU_COUNT"), n_iter_override=N_ITER,
                             platform=platform.platform(), python=sys.version.split()[0], numpy=np.__version__,
                             xgboost=xgboost.__version__, sklearn=sklearn.__version__, runtime_s=round(time.time() - t0, 1))
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
log("ALL DONE\n" + json.dumps({k: summary[k] for k in ("metrics", "polymorph_ordering")}, indent=1))
