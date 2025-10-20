"""mlflow experiment tracking utilities."""

from __future__ import annotations

import logging
import math
from typing import Union

import mlflow
from mlflow.tracking import MlflowClient

from utils import flatten_dict


def setup_mlflow(
        mlflow_cfg: dict,
        hyper_params: dict,
        logger: logging.Logger,
        extra_tags: dict | None = None,
):
    """Initialize mlflow run based on configuration.
    
    Args:
        mlflow_cfg: mlflow config dict, must contain:
            - tracking_uri: mlflow tracking server address
            - experiment: Experiment name
            - run_name: Run name
            - tags: (optional) Tags dict
        hyper_params: Hyperparameters dict (will be flattened and logged as mlflow params)
        logger: Logger instance
        extra_tags: (optional) Additional tags to merge with config tags
        
    Returns:
        mlflow module object, or None if initialization fails
        
    Raises:
        ValueError: If mlflow_cfg is missing required fields
        TypeError: If tags is not a dict
    """

    assert isinstance(mlflow_cfg, dict), "mlflow_cfg must be a dict"
    assert isinstance(hyper_params, dict), "hyper_params must be a dict"

    required_keys = ["tracking_uri", "experiment", "run_name"]
    missing_keys = [k for k in required_keys if not mlflow_cfg.get(k)]
    if missing_keys:
        raise ValueError(f"mlflow_cfg missing required fields: {missing_keys}")

    tracking_uri = mlflow_cfg["tracking_uri"]
    experiment_name = mlflow_cfg["experiment"]
    run_name = mlflow_cfg["run_name"]

    tags_source = mlflow_cfg.get("tags")
    if tags_source is not None and not isinstance(tags_source, dict):
        raise TypeError("mlflow_cfg['tags'] must be a dict")

    tags: dict = {}
    if isinstance(tags_source, dict):
        tags.update(tags_source)
    if extra_tags:
        tags.update(extra_tags)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        # Simple connectivity check by getting tracking URI
        client = MlflowClient()
        _ = client.get_experiment_by_name(experiment_name)  # Verify connection
        mlflow.set_experiment(experiment_name="default")
        mlflow.start_run(run_name=run_name, tags=tags or None)
    except Exception as exc:
        logger.error("mlflow initialization failed: %s", exc, exc_info=True)
        return None

    flat_params = flatten_dict(hyper_params)
    clean_params = {
        k: v.__name__ if isinstance(v, type) else v
        for k, v in flat_params.items()
    }
    mlflow.log_params(clean_params)

    logger.info("[mlflow] Experiment: %s | Run: %s", experiment_name, run_name)
    return mlflow


def prepare_metrics(result: dict) -> dict:
    """Prepare metrics for mlflow logging.
    
    Extracts numeric metrics from training result dict, filtering out non-finite
    values and non-numeric types.
    
    Args:
        result: Training result dict (typically contains sampler_results, info, etc.)
        
    Returns:
        Dict with numeric metrics (keys: metric names, values: floats)
    """
    metrics = {}
    metrics.update(flatten_dict(result.get("sampler_results", {})))
    info_flat = flatten_dict(result.get("info", {}))

    # Replay buffer statistics可能嵌套在"buffer"或info中
    buffer_stats = result.get("buffer") or info_flat.get("buffer")
    if buffer_stats:
        metrics.update(flatten_dict(buffer_stats))

    metrics.update(info_flat)

    # Keep only numeric finite values
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and v is not None and math.isfinite(v)
    }


