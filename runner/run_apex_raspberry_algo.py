#!/usr/bin/env python3
"""Minimal training script for APEX-DQN + RASPBERry.

Direct algorithm construction without Trainer wrapper,
suitable for quick experiments and debugging.
Distributed version with multiple workers and replay actors.
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
from ray.tune.registry import register_env

# Local imports
from algorithms.apex_dqn_raspberry_algo import ApexDQNRaspberryAlgo
from metrics import write_iteration_json
from metrics.logger import setup_logger
from metrics.mlflow_helper import setup_mlflow, prepare_metrics
from replay_buffer.d_raspberry_ray import MultiAgentRASPBERryReplayBuffer
from utils import env_creator, load_config, infer_env_type
from utils.config_helper import load_buffer_dump_config

CONFIG_PATH = str((ROOT / "configs/apex_raspberry_atari.yml").resolve())
RUNTIME_CONFIG = str((ROOT / "configs/runtime.yml").resolve())


def build_algorithm(env_name: str, env_short: str, config: dict) -> ApexDQNRaspberryAlgo:
    """Build APEX-DQN RASPBERry algorithm instance.

    Uses dict-based config (not ApexDQNConfig) for direct construction.

    Args:
        env_name: Full environment name, e.g. "Atari-BreakoutNoFrameskip-v4"
        env_short: Short name for registration, e.g. "Breakout"
        config: Complete config dict loaded from YAML

    Returns:
        Configured ApexDQNRaspberryAlgo instance
    """
    hyper = config["hyper_parameters"].copy()

    env_config = {"id": env_name}
    game = env_creator(env_config)
    register_env(env_short, env_creator)

    replay_buffer_config = {
        **hyper["replay_buffer_config"],
        "type": MultiAgentRASPBERryReplayBuffer,
        "obs_space": game.observation_space,
        "action_space": game.action_space,
    }

    # Ensure priority-related params are floats
    for key in ["prioritized_replay_eps", "prioritized_replay_alpha", "prioritized_replay_beta"]:
        if key in replay_buffer_config:
            replay_buffer_config[key] = float(replay_buffer_config[key])

    hyper["replay_buffer_config"] = replay_buffer_config
    hyper["env_config"] = env_config
    game.close()

    return ApexDQNRaspberryAlgo(config=hyper, env=env_short)


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-DQN-RASPBERry training script")
    parser.add_argument("--env", type=str, default="Atari-Breakout",
                        help="Atari environment name (e.g., Breakout, Pong)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device ID")
    args = parser.parse_args()

    # Load configs
    config = load_config(CONFIG_PATH)
    runtime = load_config(RUNTIME_CONFIG)

    required_runtime = ["paths", "ray"]
    missing = [k for k in required_runtime if k not in runtime]
    if missing:
        raise ValueError(f"runtime.yml missing required configs: {missing}")

    paths = runtime["paths"]
    ray_cfg = runtime["ray"]
    mlflow_base = runtime.get("mlflow", None)

    run_cfg = config["run_config"]
    hyper = config["hyper_parameters"]

    max_time_s = run_cfg.get("max_time_s", 7200)
    max_iterations = run_cfg.get("max_iterations", 10000)
    log_every = config.get("logging_config", {}).get("log_freq", 10)

    # Infer environment type and construct paths dynamically
    env_type = infer_env_type(args.env)
    run_name = f"APEX-RASPBERry-{datetime.now().timestamp()}"
    log_root = Path(paths["log_base_path"]) / env_type / args.env
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TUNE_RESULTS_DIR"] = str(log_root)

    # Setup logging
    logger = setup_logger(run_name, log_dir)
    logger.info("=" * 60)
    logger.info("APEX-DQN-RASPBERry Training Started")
    logger.info("Env: %s (%s) | GPU: %s | Max time: %ds | Max iter: %d",
                args.env, env_type, args.gpu, max_time_s, max_iterations)
    logger.info("Log dir: %s", log_dir)
    logger.info("=" * 60)

    # Setup mlflow (if configured)
    if mlflow_base:
        mlflow_cfg = {
            **mlflow_base,
            "experiment": args.env,
            "run_name": run_name,
        }
        extra_tags = {
            "algorithm": "APEX-DQN",
            "buffer": "RASPBERry",
            "env": args.env,
        }
        mlflow_run = setup_mlflow(mlflow_cfg, hyper, logger, extra_tags=extra_tags)
    else:
        mlflow_run = None
        mlflow_cfg = None
        logger.info("[mlflow] Not configured, skipping experiment tracking")

    # Initialize Ray
    ray_temp_dir = f"{paths['ray_temp_dir']}ray_{int(time.time())}"
    object_store_bytes = int(ray_cfg.get("object_store_memory_gb", 80) * 1024 * 1024 * 1024)

    ray.init(
        num_cpus=hyper.get("num_cpus", 5),
        num_gpus=hyper["num_gpus"],
        include_dashboard=ray_cfg.get("include_dashboard", False),
        object_store_memory=object_store_bytes,
        _temp_dir=ray_temp_dir,
    )
    logger.info("Ray initialized (CPUs=%d, GPUs=%d)", hyper["num_cpus"], hyper["num_gpus"])

    algo = build_algorithm(args.env, args.env, config)
    logger.info("Algorithm built, starting training...")

    start_time = time.time()
    iteration = 0

    try:
        while iteration < max_iterations:
            if time.time() - start_time > max_time_s:
                logger.info("⏰ Time limit reached (%ds), stopping", max_time_s)
                break

            result = algo.train()
            iteration += 1
            
            # Note: APEX uses distributed replay buffers, full dump not supported
            # Buffer statistics are automatically logged in result['info']['replay_shard_X']
            
            # Attach replay buffer statistics to result
            # For APEX, buffer stats come from result['info']['replay_shard_X']
            try:
                from utils import flatten_dict
                # Get stats from first replay shard (following 2024 notebook implementation)
                shard_0_stats = result.get('info', {}).get('replay_shard_0', {})
                
                if shard_0_stats:
                    # Extract buffer stats for default policy
                    policy_stats = shard_0_stats.get('policy_default_policy', {})
                    if policy_stats:
                        # Get number of shards from config
                        num_shards = algo.config.get("optimizer", {}).get("num_replay_buffer_shards", 4)
                        
                        # Multiply by number of shards to get total memory usage
                        buffer_stats = {
                            "est_size_bytes": policy_stats.get("est_size_bytes", 0) * num_shards,
                            "num_entries": policy_stats.get("num_entries", 0),
                            "added_count": policy_stats.get("added_count", 0),
                            "sampled_count": policy_stats.get("sampled_count", 0),
                            "num_shards": num_shards,
                        }
                        buffer_stats["est_size_gb"] = buffer_stats["est_size_bytes"] / 1e9
                        
                        # For RASPBERry, also include compression stats if available
                        if "est_compressed_bytes" in policy_stats:
                            buffer_stats["est_compressed_bytes"] = policy_stats["est_compressed_bytes"] * num_shards
                            buffer_stats["est_compressed_gb"] = buffer_stats["est_compressed_bytes"] / 1e9
                        if "est_raw_bytes" in policy_stats:
                            buffer_stats["est_raw_bytes"] = policy_stats["est_raw_bytes"] * num_shards
                            buffer_stats["est_raw_gb"] = buffer_stats["est_raw_bytes"] / 1e9
                        
                        result["buffer"] = buffer_stats
                        
                        # Log buffer stats every iteration (for distributed monitoring)
                        if iteration % log_every == 0:
                            logger.info("📊 APEX-RASPBERry Buffer Stats (Total across all shards):")
                            logger.info(f"  Shards: {num_shards}")
                            logger.info(f"  Total Memory: {buffer_stats['est_size_gb']:.2f} GB")
                            logger.info(f"  Entries per shard: {buffer_stats['num_entries']}")
                            if "est_compressed_gb" in buffer_stats:
                                logger.info(f"  Compressed Memory: {buffer_stats['est_compressed_gb']:.2f} GB")
                                compression_ratio = buffer_stats.get("est_raw_gb", 0) / buffer_stats.get("est_compressed_gb", 1) if buffer_stats.get("est_compressed_gb", 0) > 0 else 1.0
                                logger.info(f"  Compression Ratio: {compression_ratio:.2f}x")
            except Exception as e:
                logger.warning(f"Failed to get buffer stats from info: {e}")
            
            write_iteration_json(log_dir, iteration, result)

            if iteration % log_every == 0:
                reward = result.get("episode_reward_mean", "n/a")
                steps = result.get("timesteps_total", "n/a")
                episodes = result.get("episodes_total", "n/a")
                logger.info(
                    "[Iter %05d] reward_mean=%s | timesteps=%s | episodes=%s",
                    iteration, reward, steps, episodes
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
        logger.info("Training completed | Time: %.1fh | Iterations: %d",
                    elapsed / 3600, iteration)

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

