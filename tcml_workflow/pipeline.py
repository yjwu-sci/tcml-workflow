"""Nested cross-validation training pipeline.

Outer CV (5-fold) reports generalization metrics. Inner CV (10-fold) runs
RandomizedSearchCV per outer fold. The best outer fold's hyperparameters
then retrain on the full data to produce the final deployable model.

XGBoost, SVR and GPR share this single implementation and differ only in
the ModelConfig.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import dump
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tcml_workflow.models import ModelConfig, get_model_and_param_space


@dataclass
class NestedCVMetrics:
    """Per-fold and aggregate metrics returned by NestedCVPipeline.fit."""

    fold_r: list[float] = field(default_factory=list)
    fold_r2: list[float] = field(default_factory=list)
    fold_rmse: list[float] = field(default_factory=list)
    fold_mse: list[float] = field(default_factory=list)
    best_params_per_fold: list[dict] = field(default_factory=list)
    best_fold_index: int = -1

    @property
    def r_mean(self) -> float: return float(np.mean(self.fold_r)) if self.fold_r else float("nan")
    @property
    def r_std(self) -> float: return float(np.std(self.fold_r)) if self.fold_r else float("nan")
    @property
    def r2_mean(self) -> float: return float(np.mean(self.fold_r2)) if self.fold_r2 else float("nan")
    @property
    def r2_std(self) -> float: return float(np.std(self.fold_r2)) if self.fold_r2 else float("nan")
    @property
    def rmse_mean(self) -> float: return float(np.mean(self.fold_rmse)) if self.fold_rmse else float("nan")
    @property
    def rmse_std(self) -> float: return float(np.std(self.fold_rmse)) if self.fold_rmse else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "r_mean": self.r_mean, "r_std": self.r_std,
            "r2_mean": self.r2_mean, "r2_std": self.r2_std,
            "rmse_mean": self.rmse_mean, "rmse_std": self.rmse_std,
        })
        return d


class NestedCVPipeline:
    """Run Nested-CV on a given ModelConfig.

    SVR and GPR are auto-wrapped in a StandardScaler→estimator Pipeline so
    the kernel sees standardized features; kernel methods are sensitive to
    the scale of each feature.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.estimator: Optional[object] = None  # final model retrained on all data
        self.metrics: Optional[NestedCVMetrics] = None
        self.search_param_prefix: str = ""

    def _build_search_estimator(self) -> tuple[object, dict]:
        """Wire StandardScaler if needed and return (estimator, param_grid)."""
        base_est, param_space, needs_scaling = get_model_and_param_space(self.config)

        if needs_scaling:
            est = Pipeline([("scaler", StandardScaler()), ("model", base_est)])
            # Re-prefix param keys to address the pipeline step
            prefixed = {f"model__{k}": v for k, v in param_space.items()}
            return est, prefixed
        return base_est, param_space

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_id: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> NestedCVMetrics:
        """Run Nested-CV and retain the best-fold-retrained final model.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix
        y : (n_samples,) target (log_k)
        sample_id : optional identifier for printing (e.g. mp_id or formula)
        groups : optional group labels (e.g. the chemical formula). When given, both the outer and the
            inner folds are GroupKFold splits (shuffled, same random_state), so that no group occurs on
            both sides of any split. Without groups the folds are the shuffled KFold splits used before.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        cfg = self.config
        if groups is None:
            outer_cv = KFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.random_state)
        else:
            groups = np.asarray(groups)
            outer_cv = GroupKFold(n_splits=cfg.outer_splits, shuffle=True, random_state=cfg.random_state)

        metrics = NestedCVMetrics()
        best_estimators = []
        best_param_grids = []
        self.oof_pred = np.full(len(y), np.nan)

        scheme = "KFold" if groups is None else "GroupKFold"
        print(f"=== Nested CV: {cfg.algorithm} | outer={cfg.outer_splits} | inner={cfg.inner_splits} | {scheme} ===")

        for fold_id, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if groups is None:
                inner_cv = KFold(n_splits=cfg.inner_splits, shuffle=True, random_state=cfg.random_state)
                fit_kw = {}
            else:
                inner_cv = GroupKFold(n_splits=cfg.inner_splits, shuffle=True, random_state=cfg.random_state)
                fit_kw = {"groups": groups[tr_idx]}
            est, param_grid = self._build_search_estimator()

            search = RandomizedSearchCV(
                estimator=est,
                param_distributions=param_grid,
                n_iter=cfg.n_iter_search,
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=cfg.random_state,
                refit=True,
            )
            search.fit(X_tr, y_tr, **fit_kw)

            y_pred = search.predict(X_te)
            self.oof_pred[te_idx] = y_pred
            mse = mean_squared_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)
            r_corr, _ = pearsonr(y_te, y_pred)
            rmse = float(np.sqrt(mse))

            metrics.fold_r.append(float(r_corr))
            metrics.fold_r2.append(float(r2))
            metrics.fold_rmse.append(rmse)
            metrics.fold_mse.append(float(mse))
            metrics.best_params_per_fold.append(search.best_params_)
            best_estimators.append(search.best_estimator_)
            best_param_grids.append(search.best_params_)

            print(
                f"  Fold {fold_id}: R={r_corr:.4f} R²={r2:.4f} "
                f"RMSE={rmse:.4f} | best params: {search.best_params_}"
            )

        # Select best outer fold (highest R²) and retrain on full data
        best_idx = int(np.argmax(metrics.fold_r2))
        metrics.best_fold_index = best_idx

        # Build a fresh estimator with the best-fold params and refit on all data
        final_est, _ = self._build_search_estimator()
        final_est.set_params(**best_param_grids[best_idx])
        final_est.fit(X, y)
        self.estimator = final_est

        print(
            f"  ▶ Best fold = {best_idx + 1} | "
            f"mean R²={metrics.r2_mean:.4f} ± {metrics.r2_std:.4f} | "
            f"mean R={metrics.r_mean:.4f} ± {metrics.r_std:.4f}"
        )

        self.metrics = metrics
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Call .fit() before .predict()")
        return self.estimator.predict(np.asarray(X, dtype=float))

    def save(self, out_dir: str | Path, tag: str) -> None:
        """Persist the final estimator and metrics JSON."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dump(self.estimator, out / f"{tag}_final_model.pkl")
        with open(out / f"{tag}_metrics.json", "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2, default=str)
        print(f"  Saved → {out / f'{tag}_final_model.pkl'}")
        print(f"  Saved → {out / f'{tag}_metrics.json'}")
