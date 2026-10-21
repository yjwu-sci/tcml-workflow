"""Model + hyperparameter-space registry for tcml-workflow.

One configurable interface for XGBoost, SVR and GPR. Notes on the search ranges:

- XGBoost colsample_bytree is drawn from uniform(0.4, 0.6), i.e. [0.4, 1.0];
  values above 1.0 are not allowed.
- SVR epsilon is bounded above by 1.0, matching the spread of log_k.
- GPR length_scale is drawn from uniform(0, 1.0); features are standardised
  in pipeline.py, so the range refers to the standardised scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.svm import SVR
from xgboost import XGBRegressor


SUPPORTED_ALGORITHMS = ("XGBoost", "SVR", "GPR")


@dataclass
class ModelConfig:
    """Configuration for one ML algorithm."""

    algorithm: str
    n_iter_search: int = 200  # RandomizedSearchCV iterations
    outer_splits: int = 5
    inner_splits: int = 10
    random_state: int = 0
    extra_estimator_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {SUPPORTED_ALGORITHMS}, got {self.algorithm!r}"
            )
        # XGBoost gets a larger search budget (1,000 iterations)
        if self.algorithm == "XGBoost" and self.n_iter_search == 200:
            self.n_iter_search = 1000


def _xgboost_factory(random_state: int) -> XGBRegressor:
    return XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=1)


def _xgboost_param_space() -> dict:
    # uniform(loc, scale) → samples in [loc, loc + scale]
    return {
        "n_estimators": randint(50, 5000),
        "max_depth": randint(2, 21),
        "learning_rate": uniform(0.01, 0.59),       # [0.01, 0.60]
        "subsample": uniform(0.5, 0.5),             # [0.5, 1.0]
        "colsample_bytree": uniform(0.4, 0.6),      # [0.4, 1.0]   <-- bug fix
        "gamma": uniform(0.0, 0.4),
        "reg_alpha": uniform(0.0, 20.0),
        "reg_lambda": uniform(0.0, 20.0),
        "min_child_weight": randint(1, 11),
    }


def _svr_factory(random_state: int) -> SVR:
    # SVR has no random_state of its own; the kernel is deterministic.
    return SVR(kernel="rbf")


def _svr_param_space() -> dict:
    return {
        "C": loguniform(1e-5, 2e4),
        "gamma": loguniform(1e-5, 2e4),
        "epsilon": uniform(0.0, 1.0),               # tightened from [0, 10]
    }


def _gpr_factory(random_state: int) -> GaussianProcessRegressor:
    base_kernel = ConstantKernel() * RBF() + WhiteKernel()
    return GaussianProcessRegressor(
        kernel=base_kernel,
        normalize_y=True,
        random_state=random_state,
    )


def _gpr_param_space() -> dict:
    # When features are StandardScaler-transformed (done in pipeline.py),
    # length_scale ~ O(1) is meaningful.
    return {
        "kernel__k1__k1__constant_value": uniform(0.0, 1.0),
        "kernel__k1__k2__length_scale": uniform(0.1, 5.0),   # widened from [0, 1]
        "kernel__k2__noise_level": uniform(0.0, 1.0),
    }


_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "XGBoost": (_xgboost_factory, _xgboost_param_space),
    "SVR":     (_svr_factory, _svr_param_space),
    "GPR":     (_gpr_factory, _gpr_param_space),
}


def get_model_and_param_space(config: ModelConfig) -> tuple[Any, dict, bool]:
    """Return (estimator, param_distributions, needs_scaling).

    `needs_scaling` is True for distance/kernel-based models (SVR, GPR);
    pipeline.py wires this into a StandardScaler step in front of the
    estimator before RandomizedSearchCV.
    """
    factory, param_factory = _REGISTRY[config.algorithm]
    estimator = factory(config.random_state)
    if config.extra_estimator_kwargs:
        estimator.set_params(**config.extra_estimator_kwargs)
    needs_scaling = config.algorithm in {"SVR", "GPR"}
    return estimator, param_factory(), needs_scaling
