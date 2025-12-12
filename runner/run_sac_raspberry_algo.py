#!/usr/bin/env python3
"""Minimal training script for SAC + RASPBERry.

Direct algorithm construction without Trainer wrapper,
suitable for quick experiments and debugging.
"""

# Standard library imports
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Third-party imports
import mlflow
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Local imports
from algorithms.sac_raspberry_algo import SACRaspberryAlgo
from metrics import write_iteration_json
from metrics.logger import setup_logger
from metrics.mlflow_helper import prepare_metrics, setup_mlflow
from models import SACLightweightCNN
from replay_buffer.d_raspberry_ray import MultiAgentRASPBERryReplayBuffer
from utils import env_creator, infer_env_type, ConfigLoader
from utils.config_helper import load_buffer_dump_config
from utils.dump_helper import build_run_name, prepare_dump_dir, should_dump

DEFAULT_CONFIG_PATH = str((ROOT / "configs/sac_raspberry_image.yml").resolve())
RUNTIME_CONFIG = str((ROOT / "configs/runtime.yml").resolve())


def build_algorithm(env_name: str, env_short: str, config: dict) -> SACRaspberryAlgo:
    """Build SAC RASPBERry algorithm instance.

    Uses dict-based config (not SACConfig) for direct construction.

    Args:
        env_name: Full environment name, e.g. "Atari-BreakoutNoFrameskip-v4"
        env_short: Short name for registration, e.g. "Breakout"
        config: Complete config dict loaded from YAML

    Returns:
        Configured SACRaspberryAlgo instance
    """

    hyper = config["hyper_parameters"].copy()
    
    # Register custom CNN model
    ModelCatalog.register_custom_model("SACLightweightCNN", SACLightweightCNN)

    # IMPORTANT: Propagate YAML env_config into RLlib env_config.
    # Previously we only passed {"id": env_name}, silently ignoring keys like
    # img_size/frame_skip/frame_stack/grayscale/normalize, causing unexpected behavior
    # (e.g., BOX2DI default frame_skip=4 instead of config override).
    yaml_env_cfg = config.get("env_config", {}) or {}
    env_id = yaml_env_cfg.get("id") or yaml_env_cfg.get("env_name") or env_name
    env_config = {**yaml_env_cfg, "id": env_id}
    game = env_creator(env_config)
    register_env(env_short, env_creator)

    replay_buffer_config = {
        **hyper["replay_buffer_config"],
        "type": MultiAgentRASPBERryReplayBuffer,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC-RASPBERry training script")
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
        help="Path to config file (default: sac_raspberry_image.yml for CNN, or use sac_raspberry_vector.yml for MLP)",
    )
    args = parser.parse_args()

    # Load configs with ConfigLoader
    loader = ConfigLoader(runtime_config_path=RUNTIME_CONFIG)
    config = loader.load(args.config)

    # Extract configs
    paths = config['runtime']['paths']
    ray_cfg = config['runtime']['ray']
    mlflow_base = config['runtime'].get('mlflow', None)
    
    run_cfg = config["run_config"]
    hyper = config["hyper_parameters"]

    max_time_s = run_cfg.get("max_time_s", 7200)
    max_iterations = run_cfg.get("max_iterations", 10000)
    log_every = config.get("logging_config", {}).get("log_freq", 10)

    # Get environment name from config (override command-line default)
    env_name = config.get("env_config", {}).get("env_name", args.env)
    env_alias = config.get("env_config", {}).get("env_alias", env_name)

    # Infer environment type and construct paths dynamically
    env_type = infer_env_type(env_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name, run_name_base = build_run_name(
        env_alias,
        Path(args.config),
        args.gpu,
        timestamp,
        default_alias="SAC-RASPBERry",
    )

    log_root = Path(paths["log_base_path"]) / env_type / env_name
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    dump_root = Path(ROOT) / "logs" / "sub_buffer_size_test"
    dump_dir = prepare_dump_dir(dump_root, run_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TUNE_RESULTS_DIR"] = str(log_root)

    # Setup logging
    logger = setup_logger(run_name, log_dir)
    logger.info("=" * 60)
    logger.info("SAC-RASPBERry Training Started")
    logger.info(
        "Env: %s (%s) | GPU: %s | Max time: %ds | Max iter: %d",
        env_name,
        env_type,
        args.gpu,
        max_time_s,
        max_iterations,
    )
    logger.info("Log dir: %s", log_dir)
    logger.info("=" * 60)

    # Setup mlflow (if configured and enabled)
    use_mlflow = run_cfg.get("use_mlflow", False)  # 从run_config读取
    if mlflow_base and use_mlflow:
        # Get mlflow config from YAML (with experiment and tags)
        mlflow_cfg_from_yaml = config.get("mlflow", {})
        mlflow_experiment = mlflow_cfg_from_yaml.get("experiment", env_name)
        mlflow_tags_from_yaml = mlflow_cfg_from_yaml.get("tags", {})
        
        mlflow_cfg = {
            **mlflow_base,
            "experiment": mlflow_experiment,
            "run_name": f"{env_alias}-{args.gpu}-{timestamp}",  # env_alias + GPU + timestamp
        }
        extra_tags = {
            "algorithm": mlflow_tags_from_yaml.get("algorithm", "SAC"),
            "buffer": "RASPBERry",
            "env": mlflow_tags_from_yaml.get("environment", env_name),
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

    # Initialize Ray
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

    algo = build_algorithm(env_name, env_name, config)
    logger.info("Algorithm built, starting training...")

    start_time = time.time()
    iteration = 0
    dump_history = set()

    try:
        while iteration < max_iterations:
            if time.time() - start_time > max_time_s:
                logger.info("⏰ Time limit reached (%ds), stopping", max_time_s)
                break

            result = algo.train()
            iteration += 1
            
            # Dump buffer storage for verification (controlled by runtime.yml)
            dump_config = load_buffer_dump_config('sac', RUNTIME_CONFIG)
            do_dump, key, label = should_dump(dump_config, iteration, result, dump_history)
            if do_dump and key is not None:
                from utils.buffer_dump_utils import dump_buffer_content
                dump_history.add(key)
                dump_file = dump_dir / f"{label}.pkl" if label else dump_dir / "buffer_dump.pkl"
                try:
                    stats = dump_buffer_content(algo.local_replay_buffer, dump_file)
                    logger.info("📦 Buffer content dumped to %s", dump_file)
                    if stats:
                        for policy_id, policy_stats in stats.items():
                            logger.info(
                                "  [%s] Blocks: %s, Transitions: %s, Est. Memory: %.1f MB",
                                policy_id,
                                policy_stats.get('num_blocks', 0),
                                policy_stats.get('num_transitions', 0),
                                policy_stats.get('estimated_total_memory_mb', 0.0),
                            )
                except Exception as e:
                    logger.warning("Failed to dump buffer content: %s", e)
            
            # Attach replay buffer statistics to result
            if hasattr(algo, 'local_replay_buffer'):
                from utils import flatten_dict
                buffer_stats = flatten_dict(algo.local_replay_buffer.stats())
                # Convert bytes to GB for readability
                if "est_size_bytes" in buffer_stats:
                    buffer_stats["est_size_gb"] = buffer_stats["est_size_bytes"] / 1e9
                if "est_compressed_bytes" in buffer_stats:
                    buffer_stats["est_compressed_gb"] = buffer_stats["est_compressed_bytes"] / 1e9
                if "est_raw_bytes" in buffer_stats:
                    buffer_stats["est_raw_gb"] = buffer_stats["est_raw_bytes"] / 1e9
                # Ensure num_entries is logged (usually already in stats)
                if "num_entries" not in buffer_stats and hasattr(algo.local_replay_buffer, '_num_added'):
                    buffer_stats["num_entries"] = min(algo.local_replay_buffer._num_added, 
                                                      algo.local_replay_buffer.capacity)
                result["buffer"] = buffer_stats
            
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


