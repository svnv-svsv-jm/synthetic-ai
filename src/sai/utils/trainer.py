__all__ = ["get_metric"]

import typing as ty
from loguru import logger
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl

_OUT_METRIC = ty.Union[torch.Tensor, ty.Dict[str, torch.Tensor]]
_OUT_DICT = ty.Dict[str, _OUT_METRIC]


def logged_metrics(trainer: pl.Trainer) -> ty.Optional[_OUT_DICT]:
    """Tries to retrieve logged metrics from the trainer. Optionally looks for a MetricTrackerCallback callback and returns its logged metrics."""
    try:
        logged_metrics = trainer.logged_metrics
    except Exception:
        return None
    return logged_metrics


def _get_metric(key: str, logged_metrics: dict) -> float:
    """Get one metric."""
    metric: _OUT_METRIC = logged_metrics[key]
    # Ok if numerical
    if isinstance(metric, (int, float)):
        return float(metric)
    # If Tensor, convert to float
    assert isinstance(
        metric, torch.Tensor
    ), f"This is currently only supported for Tensor metrics but got {metric}."
    output = float(metric.mean())
    return output


def get_metric_to_optimize(trainer: pl.Trainer, optimize_metric: str) -> ty.Optional[float]:
    """For Optuna."""
    # Get stuff logged during training / validation / test
    metrics = logged_metrics(trainer)
    if metrics is None:
        return None
    # Now get the metric we want to run HPO for
    if isinstance(optimize_metric, str):
        # If one metric, just get it
        key = optimize_metric
        out = _get_metric(key, metrics)
        return out
    # If multiple metrics, then return average among them
    outs = []
    for key in optimize_metric:
        outs.append(_get_metric(key, metrics))
    return sum(outs) / len(outs)


def get_metric(
    pl_trainer: pl.Trainer,
    cfg: DictConfig = None,
    metric: str = None,
) -> ty.Optional[float]:
    """Get metric from `pl.Trainer`."""
    if cfg is None and metric is None:
        raise ValueError("Please provide a configuration or the metric to get.")
    try:
        if cfg is not None:
            m: str = cfg.get("optimize_metric")
            m = m.lower().strip()
            output = get_metric_to_optimize(pl_trainer, m)
        else:
            assert metric is not None
            output = get_metric_to_optimize(pl_trainer, metric)
    except Exception as e:
        logger.warning(e)
        output = 0.0
    return output
