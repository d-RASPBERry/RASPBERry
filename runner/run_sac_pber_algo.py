#!/usr/bin/env python3
"""SAC + PBER training script (minimal runnable version).

Directly builds SACRaspberryAlgo + PBER replay buffer for quick experiments.
File structure aligned with run_sac_per_algo.py / run_sac_raspberry_algo.py for easy comparison and reuse.
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ------ Subsection: Project root on sys.path ------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------ Subsection: Third-party ------
import mlflow
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# ------ Subsection: Local ------
from algorithms.sac_raspberry_algo import SACRaspberryAlgo
from metrics import write_iteration_json, attach_buffer_stats
from metrics.logger import redirect_stdio, setup_logger
from metrics.mlflow_helper import prepare_metrics, setup_mlflow
from models import SACLightweightCNN
from replay_buffer.d_pber_ray import MultiAgentPrioritizedBlockReplayBuffer
from utils import env_creator, infer_env_type, ConfigLoader

# ====== Section: Constants ======
DEFAULT_CONFIG_PATH = str((ROOT / "configs/sac_pber_image.yml").resolve())
RUNTIME_CONFIG = str((ROOT / "configs/runtime.yml").resolve())

# ====== Section: Algorithm Construction ======

def build_algorithm(env_id: str, env_short: str, config: dict) -> SACRaspberryAlgo:
    """Build SAC PBER algorithm instance.

    Uses dict-based config (not SACConfig) for direct construction.

    Args:
        env_id: Environment ID for env_creator routing, e.g. "Atari-BreakoutNoFrameskip-v4"
        env_short: Short name for registration, e.g. "Breakout"
        config: Complete config dict loaded from YAML

    Returns:
        Configured SACRaspberryAlgo instance with PBER buffer
    """

    hyper = config["hyper_parameters"].copy()
    
    # Register custom CNN model
    ModelCatalog.register_custom_model("SACLightweightCNN", SACLightweightCNN)

    # IMPORTANT: Pass the full YAML env_config to RLlib.
    # Passing only {"id": env_id} silently ignores key settings like
    # img_size/frame_skip/frame_stack/grayscale/normalize, causing
    # mismatch between experiment config and actual behavior.
    yaml_env_cfg = config.get("env_config", {}) or {}
    env_config = {**yaml_env_cfg, "id": yaml_env_cfg.get("id", env_id)}
    game = env_creator(env_config)
    register_env(env_short, env_creator)

    replay_buffer_config = {
        **hyper["replay_buffer_config"],
        "type": MultiAgentPrioritizedBlockReplayBuffer,
        "obs_space": game.observation_space,
        "action_space": game.action_space,
    }

    # Ensure priority-related params are floats
    for key in [
        "prioritized_replay_eps",
        "prioritized_replay_alpha",
        "prioritized_replay_beta",
    ]:
        if key in replay_buffer_config:
            replay_buffer_config[key] = float(replay_buffer_config[key])

    hyper["replay_buffer_config"] = replay_buffer_config
    hyper["env_config"] = env_config
    game.close()

    return SACRaspberryAlgo(config=hyper, env=env_short)

# ====== Section: CLI / Main ======

def main() -> None:
    # ------ Subsection: CLI args ------
    parser = argparse.ArgumentParser(description="SAC-PBER training script")
    parser.add_argument(
        "--env",
        type=str,
        default="Atari-Breakout",
        help="Environment name (e.g., Atari-Breakout, CarRacing)",
    )
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device ID")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config file (default: sac_pber_image.yml for CNN, or use sac_pber_vector.yml for MLP)",
    )
    args = parser.parse_args()

    # ------ Subsection: Config loading ------
    loader = ConfigLoader(runtime_config_path=RUNTIME_CONFIG)
    config = loader.load(args.config)

    # ------ Subsection: Runtime config ------
    paths = config['runtime']['paths']
    ray_cfg = config['runtime']['ray']
    mlflow_base = config['runtime'].get('mlflow', None)
    
    run_cfg = config["run_config"]
    hyper = config["hyper_parameters"]

    max_time_s = run_cfg.get("max_time_s", 7200)
    max_iterations = run_cfg.get("max_iterations", 10000)
    log_every = config.get("logging_config", {}).get("log_freq", 10)

    env_id = config.get("env_config", {}).get("id", args.env)
    env_alias = config.get("env_config", {}).get("env_alias", env_id)

    # ------ Subsection: Paths & env vars ------
    env_type = infer_env_type(env_id)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alias_slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in env_alias
    ).strip("_")
    if not alias_slug:
        alias_slug = "SAC-PBER"
    config_slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(args.config).stem
    ).strip("_")
    run_name_base = alias_slug
    if config_slug and config_slug.lower() not in alias_slug.lower():
        run_name_base = f"{alias_slug}-{config_slug}"
    run_name = f"{run_name_base}-{args.gpu}-{timestamp}"
    
    log_root = Path(paths["log_base_path"]) / env_type / env_id
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TUNE_RESULTS_DIR"] = str(log_root)

    # ------ Subsection: Logging ------
    redirect_stdio(log_dir)
    logger = setup_logger(run_name, log_dir)

    # Reuse the same handlers for replay buffer loggers to avoid logs scattering to stderr.
    buffer_loggers = [
        logging.getLogger("replay_buffer"),
        logging.getLogger("replay_buffer.d_raspberry_ray"),
        logging.getLogger("replay_buffer.raspberry_ray"),
    ]
    for buf_logger in buffer_loggers:
        buf_logger.handlers = []
        for handler in logger.handlers:
            buf_logger.addHandler(handler)
        buf_logger.setLevel(logging.INFO)
        buf_logger.propagate = False
    logger.info("=" * 60)
    logger.info("SAC-PBER Training Started")
    logger.info(
        "Env: %s (%s) | GPU: %s | Max time: %ds | Max iter: %d",
        env_id,
        env_type,
        args.gpu,
        max_time_s,
        max_iterations,
    )
    logger.info("Log dir: %s", log_dir)
    logger.info("=" * 60)

    # ------ Subsection: MLflow (optional) ------
    use_mlflow = run_cfg.get("use_mlflow", False)
    if mlflow_base and use_mlflow:
        # Get mlflow config from YAML (with experiment and tags)
        mlflow_cfg_from_yaml = config.get("mlflow", {})
        mlflow_experiment = mlflow_cfg_from_yaml.get("experiment", env_id)
        mlflow_tags_from_yaml = mlflow_cfg_from_yaml.get("tags", {})
        
        mlflow_cfg = {
            **mlflow_base,
            "experiment": mlflow_experiment,
            "run_name": f"{run_name_base}-{args.gpu}-{timestamp}",
        }
        extra_tags = {
            "algorithm": mlflow_tags_from_yaml.get("algorithm", "SAC"),
            "buffer": "PBER",
            "env": mlflow_tags_from_yaml.get("environment", env_id),
            "env_alias": env_alias,
            "obs_type": mlflow_tags_from_yaml.get("obs_type", "unknown"),
            "gpu": args.gpu,
        }
        mlflow_run = setup_mlflow(mlflow_cfg, hyper, logger, extra_tags=extra_tags)
    else:
        mlflow_run = None
        mlflow_cfg = None
        if not use_mlflow:
            logger.info("[mlflow] Disabled in run_config")
        else:
            logger.info("[mlflow] Not configured, skipping experiment tracking")

    # ------ Subsection: Ray init ------
    ray_temp_dir = f"{paths['ray_temp_dir']}ray_{int(time.time())}"
    object_store_bytes = int(
        ray_cfg.get("object_store_memory_gb", 90) * 1024 * 1024 * 1024
    )

    ray.init(
        num_cpus=hyper.get("num_cpus", 5),
        num_gpus=hyper.get("num_gpus", 0),
        include_dashboard=ray_cfg.get("include_dashboard", False),
        object_store_memory=object_store_bytes,
        _temp_dir=ray_temp_dir,
    )
    logger.info("Ray initialized (CPUs=%d, GPUs=%s)", hyper.get("num_cpus", 5), hyper.get("num_gpus", 0))

    # ------ Subsection: Algorithm init ------
    algo = build_algorithm(env_id, env_alias, config)
    logger.info("Algorithm built, starting training...")

    # ------ Subsection: Train loop ------
    start_time = time.time()
    iteration = 0
    try:
        while iteration < max_iterations:
            if time.time() - start_time > max_time_s:
                logger.info("⏰ Time limit reached (%ds), stopping", max_time_s)
                break

            result = algo.train()
            iteration += 1

            attach_buffer_stats(result, algo)
            write_iteration_json(log_dir, iteration, result)

            if iteration % log_every == 0:
                reward = result.get("episode_reward_mean", "n/a")
                steps = result.get("timesteps_total", "n/a")
                episodes = result.get("episodes_total", "n/a")
                logger.info(
                    "[Iter %05d] reward_mean=%s | timesteps=%s | episodes=%s",
                    iteration,
                    reward,
                    steps,
                    episodes,
                )

                if mlflow_run is not None and mlflow_cfg is not None:
                    metrics = prepare_metrics(result)
                    step = result.get("episodes_total", iteration)

                    try:
                        mlflow.log_metrics(metrics, step=step)
                    except Exception as e:
                        logger.warning("[mlflow] log_metrics failed: %s", e)
                    else:
                        try:
                            if iteration % mlflow_cfg.get("log_artifacts_every", 200) == 0:
                                mlflow.log_artifacts(str(log_dir))
                        except Exception as e:
                            logger.warning("[mlflow] log_artifacts failed: %s", e)

    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
    except Exception as e:
        logger.error("❌ Error during training: %s", e, exc_info=True)
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(
            "Training completed | Time: %.1fh | Iterations: %d",
            elapsed / 3600,
            iteration,
        )

        algo.stop()
        ray.shutdown()

        if mlflow_run is not None:
            try:
                mlflow.log_artifacts(str(log_dir))
            except Exception as e:
                logger.warning("[mlflow] log_artifacts failed: %s", e)
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning("[mlflow] end_run failed: %s", e)
            logger.info("mlflow run ended")

        logger.info("=" * 60)


if __name__ == "__main__":
    main()

