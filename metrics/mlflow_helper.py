"""mlflow experiment tracking utilities."""

from __future__ import annotations

import logging
import math
import zipfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from utils import flatten_dict


# ====== Section: MLflow Setup ======

def build_mlflow_tags(
        mlflow_cfg: dict,
        hyper_params: dict,
        extra_tags: dict | None = None,
) -> dict | None:
    """Build MLflow tags with the effective RLlib seed recorded."""

    tags_source = mlflow_cfg.get("tags")
    run_tags_source = mlflow_cfg.get("run_tags")
    if tags_source is not None and not isinstance(tags_source, dict):
        raise TypeError("mlflow_cfg['tags'] must be a dict")
    if run_tags_source is not None and not isinstance(run_tags_source, dict):
        raise TypeError("mlflow_cfg['run_tags'] must be a dict")

    tags: dict = {}
    if isinstance(run_tags_source, dict):
        tags.update(run_tags_source)
    if isinstance(tags_source, dict):
        tags.update(tags_source)
    if extra_tags:
        tags.update(extra_tags)

    seed = hyper_params.get("seed")
    if seed is not None:
        tags["seed"] = str(seed)

    if "user" in tags and "mlflow.user" not in tags:
        tags["mlflow.user"] = tags["user"]

    return tags or None


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

    tags = build_mlflow_tags(mlflow_cfg, hyper_params, extra_tags=extra_tags)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        # Simple connectivity check by getting tracking URI
        client = MlflowClient()
        _ = client.get_experiment_by_name(experiment_name)  # Verify connection
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.start_run(run_name=run_name, tags=tags)
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

    # Replay buffer statistics may be nested under "buffer" or info
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


def _is_checkpoint_path(relative_path: Path) -> bool:
    """Return true for checkpoint-like paths that should stay local by default."""

    return any(
        part == "checkpoint"
        or part == "checkpoints"
        or part.startswith("checkpoint_")
        or part.startswith("checkpoint-")
        for part in relative_path.parts
    )


def create_run_archive(
    log_dir: str | Path,
    *,
    archive_root: str | Path | None = None,
    include_checkpoints: bool = False,
) -> tuple[Path, int]:
    """Create one zip archive for a run directory.

    The archive is stored outside ``log_dir`` by default so repeated finalization
    cannot recursively include a previous archive. Checkpoint-like directories
    are excluded unless explicitly requested because they are not core evidence
    for the current RASPBERry experiments and can dominate artifact size.
    """

    resolved_log_dir = Path(log_dir).resolve()
    if not resolved_log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {resolved_log_dir}")

    resolved_archive_root = (
        Path(archive_root).resolve()
        if archive_root is not None
        else resolved_log_dir.parent / "_archives"
    )
    resolved_archive_root.mkdir(parents=True, exist_ok=True)

    archive_path = resolved_archive_root / f"{resolved_log_dir.name}.zip"
    tmp_path = archive_path.with_suffix(f"{archive_path.suffix}.tmp")

    file_count = 0
    with zipfile.ZipFile(
        tmp_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for path in sorted(resolved_log_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() in {archive_path, tmp_path}:
                continue

            relative_path = path.relative_to(resolved_log_dir)
            if not include_checkpoints and _is_checkpoint_path(relative_path):
                continue

            zf.write(path, arcname=str(relative_path))
            file_count += 1

    tmp_path.replace(archive_path)
    return archive_path, file_count


def log_final_artifact_archive(
    mlflow_module,
    log_dir: str | Path,
    logger: logging.Logger,
    mlflow_cfg: dict | None = None,
) -> Path | None:
    """Zip a completed run directory and upload it as one MLflow artifact."""

    cfg = mlflow_cfg or {}
    if not cfg.get("log_final_artifact_archive", True):
        logger.info("[mlflow] final artifact archive disabled")
        return None

    include_checkpoints = bool(cfg.get("include_checkpoints_in_archive", False))
    archive_root = cfg.get("archive_local_dir")
    artifact_path = cfg.get("final_artifact_path", "run_archive")

    for handler in logger.handlers:
        handler.flush()

    archive_path, file_count = create_run_archive(
        log_dir,
        archive_root=archive_root,
        include_checkpoints=include_checkpoints,
    )
    mlflow_module.log_artifact(str(archive_path), artifact_path=artifact_path)
    logger.info(
        "[mlflow] final artifact archive uploaded: %s (%d files)",
        archive_path,
        file_count,
    )
    return archive_path


