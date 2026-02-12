#!/usr/bin/env python3
"""Minimal training script for APEX-DQN + PBER.

Direct algorithm construction with PBER buffer,
suitable for quick experiments.
Distributed version with multiple workers and replay actors.
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import argparse
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
from ray.tune.registry import register_env  

# ------ Subsection: Local ------
from algorithms.apex_dqn_pber_algo import ApexDQNPberAlgo
from metrics import write_iteration_json
from metrics.logger import setup_logger
from metrics.mlflow_helper import setup_mlflow, prepare_metrics
from replay_buffer.d_pber_ray import MultiAgentPrioritizedBlockReplayBuffer
from utils import env_creator, infer_env_type, ConfigLoader

# ====== Section: Constants ======
DEFAULT_CONFIG_PATH = str((ROOT / "configs/apex_pber_atari.yml").resolve())
RUNTIME_CONFIG = str((ROOT / "configs/runtime.yml").resolve())


# ====== Section: Algorithm Construction ======
def build_algorithm(env_id: str, env_short: str, config: dict) -> ApexDQNPberAlgo:
    """Build APEX-DQN PBER algorithm instance.

    Uses dict-based config (not ApexDQNConfig) for direct construction.

    Args:
        env_id: Environment ID for env_creator routing, e.g. "Atari-BreakoutNoFrameskip-v4"
        env_short: Short name for registration, e.g. "Breakout"
        config: Complete config dict loaded from YAML

    Returns:
        Configured ApexDQNRaspberryAlgo instance with PBER buffer
    """
    hyper = config["hyper_parameters"].copy()

    # Pass full YAML env_config to preserve env-specific settings.
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
    for key in ["prioritized_replay_eps", "prioritized_replay_alpha", "prioritized_replay_beta"]:
        if key in replay_buffer_config:
            replay_buffer_config[key] = float(replay_buffer_config[key])

    hyper["replay_buffer_config"] = replay_buffer_config
    hyper["env_config"] = env_config
    game.close()

    return ApexDQNPberAlgo(config=hyper, env=env_short)


# ====== Section: CLI / Main ======
def main() -> None:
    # ------ Subsection: CLI args ------
    parser = argparse.ArgumentParser(description="APEX-DQN-PBER training script")
    parser.add_argument("--env", type=str, default="Atari-Breakout",
                        help="Atari environment name (e.g., Breakout, Pong)")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device ID")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to APEX-PBER config YAML (default: apex_pber_atari.yml)",
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

    # Infer environment type and construct paths dynamically
    env_id = config.get("env_config", {}).get("id", args.env)
    env_alias = config.get("env_config", {}).get("env_alias", env_id)

    env_type = infer_env_type(env_id)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alias_slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in env_alias
    ).strip("_")
    prefix = "APEX-PBER"
    config_slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(args.config).stem
    ).strip("_")
    if alias_slug:
        run_name_base = f"{prefix}-{alias_slug}"
    else:
        run_name_base = prefix
    if config_slug and config_slug.lower() not in run_name_base.lower():
        run_name_base = f"{run_name_base}-{config_slug}"
    run_name = f"{run_name_base}-{args.gpu}-{timestamp}"
    log_root = Path(paths["log_base_path"]) / env_type / env_id
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TUNE_RESULTS_DIR"] = str(log_root)

    # Setup logging
    logger = setup_logger(run_name, log_dir)
    logger.info("=" * 60)
    logger.info("APEX-DQN-PBER Training Started")
    logger.info("Env: %s (%s) | GPU: %s | Max time: %ds | Max iter: %d",
                env_id, env_type, args.gpu, max_time_s, max_iterations)
    logger.info("Log dir: %s", log_dir)
    logger.info("=" * 60)

    # Setup mlflow (if configured)
    use_mlflow = run_cfg.get("use_mlflow", False)
    if mlflow_base and use_mlflow:
        mlflow_cfg_from_yaml = config.get("mlflow", {})
        mlflow_experiment = mlflow_cfg_from_yaml.get("experiment", env_id)
        mlflow_tags_from_yaml = mlflow_cfg_from_yaml.get("tags", {})

        mlflow_cfg = {
            **mlflow_base,
            "experiment": mlflow_experiment,
            "run_name": f"{env_alias}-{args.gpu}-{timestamp}",
        }
        extra_tags = {
            "algorithm": mlflow_tags_from_yaml.get("algorithm", "APEX-DQN"),
            "buffer": mlflow_tags_from_yaml.get("buffer", "PBER"),
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

    # Initialize Ray
    ray_temp_dir = f"{paths['ray_temp_dir']}ray_{int(time.time())}"
    object_store_bytes = int(ray_cfg.get("object_store_memory_gb", 80) * 1024 * 1024 * 1024)

    num_cpus = hyper.get("num_cpus", 5)
    num_gpus = hyper.get("num_gpus", 0)
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        include_dashboard=ray_cfg.get("include_dashboard", False),
        object_store_memory=object_store_bytes,
        _temp_dir=ray_temp_dir,
    )
    logger.info("Ray initialized (CPUs=%d, GPUs=%s)", num_cpus, num_gpus)

    algo = build_algorithm(env_id, env_alias, config)
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
            
            # Attach replay buffer statistics to result
            # For APEX, buffer stats come from result['info']['replay_shard_X']
            try:
                from utils import flatten_dict
                # Get stats from first replay shard
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
                            "num_blocks": policy_stats.get("num_blocks", 0),
                            "added_count": policy_stats.get("added_count", 0),
                            "sampled_count": policy_stats.get("sampled_count", 0),
                            "num_shards": num_shards,
                        }
                        buffer_stats["est_size_gb"] = buffer_stats["est_size_bytes"] / 1e9
                        
                        result["buffer"] = buffer_stats
                        
                        # Log buffer stats every iteration (for distributed monitoring)
                        if iteration % log_every == 0:
                            logger.info("📊 APEX-PBER Buffer Stats (Total across all shards):")
                            logger.info(f"  Shards: {num_shards}")
                            logger.info(f"  Total Memory: {buffer_stats['est_size_gb']:.2f} GB")
                            logger.info(f"  Entries per shard: {buffer_stats['num_entries']}")
                            logger.info(f"  Blocks per shard: {buffer_stats.get('num_blocks', 0)}")
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

