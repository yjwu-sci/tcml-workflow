"""Smoke tests for tcml_workflow.models and pipeline."""

import numpy as np
import pytest

from tcml_workflow import ModelConfig, NestedCVPipeline
from tcml_workflow.models import get_model_and_param_space


def _toy_data(n=80, n_features=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # log_k linear in first 3 features + small noise
    y = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("algo", ["XGBoost", "SVR", "GPR"])
def test_model_factory(algo):
    cfg = ModelConfig(algo, n_iter_search=2)
    est, params, needs_scaling = get_model_and_param_space(cfg)
    assert est is not None
    assert params and isinstance(params, dict)
    assert needs_scaling == (algo in {"SVR", "GPR"})


@pytest.mark.parametrize("algo", ["XGBoost", "SVR", "GPR"])
def test_nested_cv_smoke(algo):
    X, y = _toy_data(n=80)
    cfg = ModelConfig(
        algo, n_iter_search=2, outer_splits=3, inner_splits=2, random_state=0,
    )
    pipe = NestedCVPipeline(cfg)
    metrics = pipe.fit(X, y)

    assert metrics is not None
    assert len(metrics.fold_r2) == 3
    assert all(np.isfinite(r) for r in metrics.fold_r2)
    assert pipe.estimator is not None

    y_pred = pipe.predict(X[:5])
    assert y_pred.shape == (5,)


def test_xgboost_colsample_range_is_legal():
    """colsample_bytree must stay within the legal range [0.4, 1.0]."""
    cfg = ModelConfig("XGBoost", n_iter_search=1)
    _, params, _ = get_model_and_param_space(cfg)
    samples = params["colsample_bytree"].rvs(size=500, random_state=0)
    assert (samples >= 0.4).all() and (samples <= 1.0).all(), (
        f"colsample_bytree out of legal range: [{samples.min()}, {samples.max()}]"
    )


def test_svr_epsilon_range_tightened():
    """SVR epsilon is bounded above by 1.0."""
    cfg = ModelConfig("SVR", n_iter_search=1)
    _, params, _ = get_model_and_param_space(cfg)
    samples = params["epsilon"].rvs(size=500, random_state=0)
    assert samples.max() <= 1.0, f"SVR epsilon range too wide: max={samples.max()}"
