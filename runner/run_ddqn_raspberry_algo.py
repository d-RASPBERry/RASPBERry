#!/usr/bin/env python3
"""Minimal training script for DDQN + RASPBERry.

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
from ray.tune.registry import register_env

# Local imports
from algorithms.dqn_raspberry_algo import DQNRaspberryAlgo
from metrics import write_iteration_json
from metrics.logger import setup_logger
from metrics.mlflow_helper import setup_mlflow, prepare_metrics
from replay_buffer.d_raspberry_ray import MultiAgentPrioritizedBlockReplayBuffer
from utils import env_creator, load_config, infer_env_type

CONFIG_PATH = str((ROOT / "configs/ddqn_raspberry_atari.yml").resolve())
RUNTIME_CONFIG = str((ROOT / "configs/runtime.yml").resolve())


def build_algorithm(env_name: str, env_short: str, config: dict) -> DQNRaspberryAlgo:
    """Build DQN RASPBERry algorithm instance.

    Uses dict-based config (not DQNConfig) for direct construction.

    Args:
        env_name: Full environment name, e.g. "Atari-BreakoutNoFrameskip-v4"
        env_short: Short name for registration, e.g. "Breakout"
        config: Complete config dict loaded from YAML

    Returns:
        Configured DQNRaspberryAlgo instance
    """
    hyper = config["hyper_parameters"].copy()

    env_config = {"id": env_name}
    game = env_creator(env_config)
    register_env(env_short, env_creator)

    replay_buffer_config = {
        **hyper["replay_buffer_config"],
        "type": MultiAgentPrioritizedBlockReplayBuffer,
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

    return DQNRaspberryAlgo(config=hyper, env=env_short)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDQN-RASPBERry training script")
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
    run_name = f"DDQN-RASPBERry-{datetime.now().timestamp()}"
    log_root = Path(paths["log_base_path"]) / env_type / args.env
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TUNE_RESULTS_DIR"] = str(log_root)

    # Setup logging
    logger = setup_logger(run_name, log_dir)
    logger.info("=" * 60)
    logger.info("DDQN-RASPBERry Training Started")
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
            "algorithm": "DDQN",
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
            
            # Dump buffer storage at iteration 200 for verification
            if iteration == 200:
                from utils.buffer_dump_utils import dump_buffer_content
                dump_file = log_dir / "buffer_storage_iter200.pkl"
                try:
                    # Dump完整的buffer _storage用于验证
                    stats = dump_buffer_content(algo.local_replay_buffer, dump_file)
                    logger.info("📦 Buffer content dumped to %s", dump_file)
                    if stats:
                        for policy_id, policy_stats in stats.items():
                            logger.info(f"  [{policy_id}] Compression: {policy_stats.get('compression_ratio', 1.0):.2f}x, "
                                      f"Est. Memory: {policy_stats.get('estimated_total_memory_mb', 0):.1f} MB")
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
                    iteration, reward, steps, episodes
                )

                if mlflow_run is not None and mlflow_cfg is not None:
                    metrics = prepare_metrics(result)
                    step = result.get("episodes_total", iteration)
                    mlflow.log_metrics(metrics, step=step)

                    if iteration % mlflow_cfg.get("log_artifacts_every", 200) == 0:
                        mlflow.log_artifacts(str(log_dir))

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
            mlflow.log_artifacts(str(log_dir))
            mlflow.end_run()
            logger.info("mlflow run ended")

        logger.info("=" * 60)


if __name__ == "__main__":
    main()
