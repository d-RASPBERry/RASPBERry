#!/usr/bin/env python3
"""最小化的 SAC + PER 驱动脚本，直接构建并运行算法。

该脚本仿照 `run_sac_raspberry_algo.py`，但使用标准
`MultiAgentPrioritizedReplayBuffer`，方便与 RASPBERry 版本对照调试。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import ray
from ray.tune.registry import register_env

ROOT = Path("/home/seventheli/research/RASPBERry")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import convert_np_arrays, env_creator, flatten_dict, load_config
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.replay_buffers import MultiAgentPrioritizedReplayBuffer


# === 配置文件路径 === #
CONFIG_PATH = str((ROOT / "configs/sac_per.yml").resolve())
PATHS_CONFIG = str((ROOT / "configs/path.yml").resolve())
MLFLOW_CONFIG = str((ROOT / "configs/mlflow.yml").resolve())


def _build_run_name(run_cfg: dict, env_alias: str) -> str:
    template = run_cfg.get("run_name_template", "{env}_run")
    base_name = template.format(env_alias=env_alias, env_in=env_alias, env=env_alias)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def _setup_logger(run_name: str, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"SAC_PER.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s', datefmt='%H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _setup_mlflow(mlflow_cfg: dict, run_name: str, hyper_params: dict, logger: logging.Logger):
    """启动 MLflow 记录。如果未安装 mlflow，返回 None。"""
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow 未安装，跳过实验记录。")
        return None

    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow_cfg.get("experiment", "SAC")
    mlflow.set_experiment(experiment_name=experiment)

    run_tags = mlflow_cfg.get("tags", {})
    mlflow.start_run(run_name=run_name, tags=run_tags)

    flat_params = flatten_dict(hyper_params)
    clean_params = {}
    for key, value in flat_params.items():
        if isinstance(value, type):
            clean_params[key] = value.__name__
        else:
            clean_params[key] = value
    if clean_params:
        mlflow.log_params(clean_params)

    logger.info("[MLFLOW] 记录已开启 -> 实验: %s", experiment)
    return mlflow


def _prepare_metrics(result: dict) -> dict:
    metrics = {
        "episode_reward_mean": result.get("episode_reward_mean"),
        "timesteps_total": result.get("timesteps_total"),
        "episodes_total": result.get("episodes_total"),
    }
    metrics.update(flatten_dict(result.get("sampler_results", {})))
    metrics.update(flatten_dict(result.get("info", {})))

    numeric_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None and math.isfinite(value):
            numeric_metrics[key] = float(value)
    return numeric_metrics


def _write_iteration_json(log_dir: Path, iteration: int, result: dict) -> None:
    payload = {
        "iteration": iteration,
        "timestamp": time.time(),
        "result": result,
    }
    record = convert_np_arrays(payload)
    output_path = log_dir / f"result_{iteration:05d}.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2)


def build_algorithm(env_name: str, env_alias: str, config: dict):
    """构建并返回 SAC + PER 算法实例。"""

    hyper = config["hyper_parameters"]
    buffer_cfg = hyper["replay_buffer_config"].copy()

    game = env_creator({"id": env_name})
    obs_space = game.observation_space
    action_space = game.action_space
    game.close()

    buffer_cfg["obs_space"] = obs_space
    buffer_cfg["action_space"] = action_space
    if buffer_cfg.get("type") == "MultiAgentPrioritizedReplayBuffer":
        buffer_cfg["type"] = MultiAgentPrioritizedReplayBuffer

    for key in ("prioritized_replay_eps", "prioritized_replay_alpha", "prioritized_replay_beta"):
        if key in buffer_cfg:
            buffer_cfg[key] = float(buffer_cfg[key])

    env_config = {"id": env_name}
    register_env(env_alias, lambda cfg: env_creator(env_config))

    cfg = SACConfig()
    cfg = cfg.environment(env=env_alias, env_config=env_config)
    cfg = cfg.framework(hyper["framework"])

    cfg = cfg.training(
        lr=hyper["lr"],
        gamma=hyper["gamma"],
        tau=hyper.get("tau", 0.005),
        target_entropy=hyper.get("target_entropy", "auto"),
        initial_alpha=hyper.get("initial_alpha", 1.0),
        n_step=hyper.get("n_step", 1),
        train_batch_size=hyper["train_batch_size"],
        replay_buffer_config=buffer_cfg,
        num_steps_sampled_before_learning_starts=hyper["num_steps_sampled_before_learning_starts"],
    )

    cfg = cfg.reporting(
        min_sample_timesteps_per_iteration=hyper["min_sample_timesteps_per_iteration"]
    )

    cfg = cfg.resources(
        num_gpus=hyper["num_gpus"],
        num_gpus_per_worker=hyper.get("num_gpus_per_worker", 0),
        num_cpus_per_worker=hyper.get("num_cpus_per_worker", 1),
    )

    cfg = cfg.rollouts(
        num_rollout_workers=hyper.get("num_workers", 0),
        num_envs_per_worker=hyper.get("num_envs_per_worker", 1),
        rollout_fragment_length=hyper["rollout_fragment_length"],
    )

    return cfg.build()


def main() -> None:
    # 加载配置文件
    config = load_config(CONFIG_PATH)
    paths = load_config(PATHS_CONFIG)
    mlflow_base = load_config(MLFLOW_CONFIG)
    
    # 提取各模块配置
    env_cfg = config.get("env_config", {})
    hardware_cfg = config.get("hardware", {})
    logging_cfg = config.get("logging", {})
    run_cfg = config.get("run_config", {})
    ray_cfg = config.get("ray_config", {})
    hyper = config["hyper_parameters"]
    
    # 合并 MLflow 配置（算法配置覆盖全局配置）
    mlflow_cfg = mlflow_base.copy()
    if "mlflow" in config:
        mlflow_cfg.update(config["mlflow"])
        # 合并 tags
        if "run_tags" in mlflow_base and "tags" in config["mlflow"]:
            merged_tags = mlflow_base["run_tags"].copy()
            merged_tags.update(config["mlflow"]["tags"])
            mlflow_cfg["tags"] = merged_tags
        elif "tags" in config["mlflow"]:
            mlflow_cfg["tags"] = config["mlflow"]["tags"]
    
    # 命令行参数（用于覆盖 YAML 中的默认值）
    parser = argparse.ArgumentParser(description="SAC-PER 训练脚本")
    parser.add_argument(
        "--max-time",
        type=int,
        default=run_cfg.get("max_time_s", 7200),
        help="最大运行时间（秒）"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=run_cfg.get("max_iterations", 100000),
        help="最大迭代次数"
    )
    args = parser.parse_args()
    
    # 从配置中读取参数
    env_name = env_cfg.get("env_name", "BOX2D-CarRacing-v2")
    env_alias = env_cfg.get("env_alias", "SAC-PER-CarRacing")
    cuda_devices = hardware_cfg.get("cuda_devices", "0")
    
    # 构建日志路径：log_base_path（来自 path.yml）+ log_subdir（来自算法配置）
    log_base = paths.get("log_base_path", "/tmp/logs")
    log_subdir = logging_cfg.get("log_subdir", "")
    log_root = Path(log_base) / log_subdir if log_subdir else Path(log_base)
    log_every = logging_cfg.get("log_every", 10)

    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    os.environ["RAY_TMPDIR"] = os.path.abspath(paths["tmp_dir"])
    log_root.mkdir(parents=True, exist_ok=True)
    os.environ["TUNE_RESULTS_DIR"] = str(log_root)
    os.environ["TUNE_RESULT_DIR"] = str(log_root)

    # 设置日志
    run_name = _build_run_name(run_cfg, env_alias)
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_name, log_dir)
    logger.info("LOG_BASE_DIR=%s", log_dir)
    mlflow = _setup_mlflow(mlflow_cfg, run_name, hyper, logger)

    ray_temp_base = paths.get("ray_temp_dir", "/tmp/ray/")
    ray_temp_dir = f"{ray_temp_base}ray_{int(time.time())}"

    object_store_bytes = int(ray_cfg.get("object_store_memory_gb", 80) * 1024 * 1024 * 1024)
    system_config = {
        "maximum_gcs_destroyed_actor_cached_count": ray_cfg.get(
            "max_gcs_destroyed_actor_cached_count", 100
        )
    }

    ray.init(
        num_cpus=hyper.get("num_cpus", 10),
        num_gpus=hyper["num_gpus"],
        include_dashboard=ray_cfg.get("include_dashboard", False),
        object_store_memory=object_store_bytes,
        _temp_dir=ray_temp_dir,
        _system_config=system_config,
        ignore_reinit_error=ray_cfg.get("ignore_reinit_error", True),
    )

    algo = build_algorithm(env_name, env_alias, config)

    # 从 YAML 配置读取运行参数
    max_time_s = run_cfg.get("max_time_s", 7200)
    max_iterations = run_cfg.get("max_iterations", 100000)
    
    logger.info("[CONFIG] max_time=%ds max_iterations=%d log_every=%d (from YAML)", max_time_s, max_iterations, log_every)

    start_time = time.time()
    iteration = 0
    try:
        while iteration < max_iterations:
            if max_time_s and (time.time() - start_time) > max_time_s:
                logger.info("[INFO] 达到运行时长上限 (%ss)，结束训练。", max_time_s)
                break

            result = algo.train()
            iteration += 1

            _write_iteration_json(log_dir, iteration, result)

            if iteration % log_every == 0:
                reward = result.get("episode_reward_mean", "n/a")
                steps = result.get("timesteps_total", "n/a")
                logger.info("[Iter %05d] reward_mean=%s timesteps=%s", iteration, reward, steps)

                if mlflow:
                    metrics = _prepare_metrics(result)
                    step = result.get("episodes_total", iteration)
                    mlflow.log_metrics(metrics, step=step)

                    if iteration % run_cfg.get("mlflow_log_artifacts_every", 200) == 0:
                        mlflow.log_artifacts(str(log_dir))
    finally:
        algo.stop()
        ray.shutdown()
        if mlflow:
            mlflow.log_artifacts(str(log_dir))
            mlflow.end_run()


if __name__ == "__main__":
    main()


