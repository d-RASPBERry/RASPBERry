import os
import ray
from pathlib import Path
from dynaconf import Dynaconf
from tqdm import tqdm
from typing import Dict, Any
import time
import json

from utils import env_creator
from ray.tune.logger import JsonLogger
from ray.tune.registry import register_env
from algorithms.ddqn_pber import DDQNWithMPBER
from replay_buffer.mpber_ram_saver_v8 import MultiAgentPrioritizedBlockReplayBuffer
from utils import convert_np_arrays, flatten_dict


class RASPBERryDDQNTrainer:
    """DDQN with RASPBERry: Prioritized Experience Replay with Block-wise Compression"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_name = config.get("env_name", "BreakoutNoFrameskip-v4")
        self.run_name = config.get("run_name", "DDQN_RASPBERry_test")
        self.sub_buffer_size = config.get("sub_buffer_size", 8)
        self.compress_base = config.get("compress_base", -1)

        # Setup paths
        self.base_path = Path("~/data/BER/New/").expanduser()
        self.log_path = self.base_path / "logs" / self.run_name
        self.checkpoint_path = self.base_path / "checkpoints" / self.run_name

        # Create directories
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Setup training log file
        self.training_log_file = self.log_path / "training.log"

        print(f"🚀 Initialize RASPBERry DDQN Trainer")
        print(f"Environment: {self.env_name}")
        print(f"Sub-buffer size: {self.sub_buffer_size}")
        print(f"Compress base: {self.compress_base}")
        print(f"Log path: {self.log_path}")
        print(f"Checkpoint path: {self.checkpoint_path}")
        print(f"Training log file: {self.training_log_file}")

        # Initialize components
        self.env_example = None
        self.trainer = None
        self.hyper_parameters = None
        self.start_time = None

        # Training statistics
        self.best_reward = float("-inf")
        self.total_timesteps = 0
        self.total_episodes = 0

        # Keys to extract from results (like in notebook)
        self.keys_to_extract_sam = {
            "episode_reward_max",
            "episode_reward_min",
            "episode_reward_mean",
        }
        self.keys_to_extract_sta = {
            "num_agent_steps_sampled",
            "num_agent_steps_trained",
        }
        self.keys_to_extract_buf = {
            "add_batch_time_ms",
            "replay_time_ms",
            "update_priorities_time_ms",
        }

    def log_message(self, message: str, level: str = "INFO"):
        """Log message to both console and file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)

        # Also write to log file
        with open(self.training_log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def setup_ray(self):
        """Initialize Ray cluster"""
        self.log_message("Initializing Ray cluster...")

        ray_config = {
            "num_cpus": 16,
            "num_gpus": 1,
            "include_dashboard": False,
            "_system_config": {"maximum_gcs_destroyed_actor_cached_count": 50},
        }

        self.log_message(f"Ray config: {ray_config}")

        ray.init(**ray_config)

        self.log_message(f"✓ Ray initialized successfully")
        self.log_message(f"Ray cluster info: {ray.cluster_resources()}")

    def setup_environment(self):
        """Setup environment and register with Ray"""
        self.log_message("Setting up environment...")
        register_env("Atari", env_creator)

        # Create environment example for space information
        env_config = {"id": self.env_name}
        self.log_message(f"Creating environment with config: {env_config}")

        self.env_example = env_creator(env_config)
        obs, _ = self.env_example.reset()

        self.log_message(f"Action space: {self.env_example.action_space}")
        self.log_message(f"Observation space: {self.env_example.observation_space}")
        self.log_message(f"Observation shape: {obs.shape}")
        self.log_message(f"✓ Environment setup completed")

    def load_config(self):
        """Load configuration from YAML file with fallback to defaults"""
        self.log_message("Loading configuration from YAML file...")

        try:
            setting = Dynaconf(
                envvar_prefix="DYNACONF", settings_files="settings/ddqn_atari.yml"
            )

            self.log_message("✓ YAML file loaded successfully")

            # Safely extract configuration
            try:
                hyper_params = getattr(setting, "hyper_parameters", None)
                if hyper_params is None:
                    raise AttributeError("hyper_parameters not found in config")

                # Convert to dict
                try:
                    self.hyper_parameters = hyper_params.to_dict()  # type: ignore
                    self.log_message("✓ Configuration converted using to_dict()")
                except (AttributeError, TypeError):
                    try:
                        self.hyper_parameters = dict(hyper_params)  # type: ignore
                        self.log_message("✓ Configuration converted using dict()")
                    except (TypeError, ValueError):
                        self.log_message(
                            "Warning: Cannot convert config, using defaults", "WARN"
                        )
                        self.hyper_parameters = self._get_default_config()

            except AttributeError:
                self.log_message(
                    "Warning: Invalid config format, using defaults", "WARN"
                )
                self.hyper_parameters = self._get_default_config()

            # Set basic configuration
            self.hyper_parameters["logger_config"] = {
                "type": JsonLogger,
                "logdir": str(self.checkpoint_path),
            }
            self.hyper_parameters["env_config"] = {"id": self.env_name}

            # Log key configuration parameters
            key_params = [
                "lr",
                "train_batch_size",
                "target_network_update_freq",
                "num_workers",
                "num_envs_per_worker",
                "framework",
            ]
            self.log_message("Key configuration parameters:")
            for param in key_params:
                if param in self.hyper_parameters:
                    self.log_message(f"  {param}: {self.hyper_parameters[param]}")

            # Get max iterations
            try:
                log_config = getattr(setting, "log", None)
                if log_config is not None:
                    max_run_value = getattr(log_config, "max_run", None)
                    if max_run_value is not None:
                        self.max_run = int(max_run_value)
                        self.log_message(f"Max iterations from config: {self.max_run}")
                    else:
                        self.max_run = 10000
                        self.log_message(
                            f"Warning: max_run not found in config, using default {self.max_run}",
                            "WARN",
                        )
                else:
                    self.max_run = 10000
                    self.log_message(
                        f"Warning: log section not found in config, using default max_run {self.max_run}",
                        "WARN",
                    )
            except (TypeError, ValueError, AttributeError):
                self.max_run = 10000
                self.log_message(
                    f"Warning: Cannot parse max_run, using default {self.max_run}",
                    "WARN",
                )

        except Exception as e:
            self.log_message(f"Config loading failed: {e}", "ERROR")
            # Use default configuration
            self.hyper_parameters = self._get_default_config()
            self.max_run = 10000

        self.log_message("✓ Configuration loading completed")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default DDQN configuration for RLLib 2.8"""
        self.log_message("Using default DDQN configuration")
        return {
            # Basic DDQN parameters
            "double_q": True,
            "dueling": False,
            "noisy": False,
            "num_atoms": 1,
            "n_step": 1,
            # Network architecture
            "hiddens": [512],
            "framework": "torch",
            # Learning parameters
            "lr": 6.25e-5,
            "adam_epsilon": 0.00015,
            "train_batch_size": 32,
            "target_network_update_freq": 8000,
            # Exploration
            "exploration_config": {
                "epsilon_timesteps": 200000,
                "final_epsilon": 0.01,
            },
            # Sampling
            "rollout_fragment_length": 4,
            "min_sample_timesteps_per_iteration": 10000,
            "num_steps_sampled_before_learning_starts": 50000,
            # Resources
            "num_cpus": 5,
            "num_gpus": 1,
            "num_workers": 1,
            "num_envs_per_worker": 20,
            # Replay buffer (will be overridden by RASPBERry config)
            "replay_buffer_config": {
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 1000000,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 1.0,
            },
            # Logging
            "logger_config": {"type": JsonLogger, "logdir": str(self.checkpoint_path)},
            "env_config": {"id": self.env_name},
        }

    def setup_replay_buffer(self):
        """Configure RASPBERry replay buffer with block-wise compression"""
        self.log_message("Configuring RASPBERry replay buffer...")

        # Ensure hyper_parameters is a dictionary
        if not isinstance(self.hyper_parameters, dict):
            self.hyper_parameters = self._get_default_config()

        # Ensure env_example exists
        if self.env_example is None:
            raise ValueError(
                "Environment not initialized, call setup_environment() first"
            )

        # Get base replay buffer config
        base_config = self.hyper_parameters.get("replay_buffer_config", {})
        if hasattr(base_config, "to_dict"):
            try:
                base_config = base_config.to_dict()  # type: ignore
            except (AttributeError, TypeError):
                base_config = {}
        elif not isinstance(base_config, dict):
            try:
                base_config = dict(base_config) if base_config else {}
            except (TypeError, ValueError):
                base_config = {}

        self.log_message(f"Base replay buffer config: {base_config}")

        # Configure RASPBERry replay buffer
        replay_buffer_config = {
            **base_config,
            "type": MultiAgentPrioritizedBlockReplayBuffer,
            "obs_space": self.env_example.observation_space,
            "action_space": self.env_example.action_space,
            "sub_buffer_size": self.sub_buffer_size,
            "worker_side_prioritization": False,
            "replay_buffer_shards_colocated_with_driver": True,
            "rollout_fragment_length": self.sub_buffer_size,
            "num_save": 50,  # Reduced for faster compression triggering
            "split_mini_batch": 4,
            "compress_base": self.compress_base,
        }

        self.hyper_parameters["replay_buffer_config"] = replay_buffer_config

        # Log RASPBERry specific config
        raspberfy_params = [
            "sub_buffer_size",
            "num_save",
            "split_mini_batch",
            "compress_base",
        ]
        self.log_message("RASPBERry replay buffer parameters:")
        for param in raspberfy_params:
            if param in replay_buffer_config:
                self.log_message(f"  {param}: {replay_buffer_config[param]}")

        # Adjust training batch size for block structure
        # Each block contains sub_buffer_size samples
        # When trainer requests num_items, it gets num_items * sub_buffer_size samples
        original_train_batch_size = self.hyper_parameters.get("train_batch_size", 32)
        if hasattr(original_train_batch_size, "__int__"):
            original_train_batch_size = int(original_train_batch_size)
        elif isinstance(original_train_batch_size, (int, float)):
            original_train_batch_size = int(original_train_batch_size)
        else:
            original_train_batch_size = 32
            self.log_message(
                f"Warning: Cannot parse train_batch_size, using default {original_train_batch_size}",
                "WARN",
            )

        adjusted_train_batch_size = int(
            original_train_batch_size / self.sub_buffer_size
        )
        self.hyper_parameters["train_batch_size"] = adjusted_train_batch_size

        self.log_message("Batch size adjustment for block structure:")
        self.log_message(f"  Original train_batch_size: {original_train_batch_size}")
        self.log_message(f"  Sub-buffer size: {self.sub_buffer_size}")
        self.log_message(
            f"  Adjusted train_batch_size: {adjusted_train_batch_size} blocks"
        )
        self.log_message(
            f"  Expected final batch size: {adjusted_train_batch_size * self.sub_buffer_size} samples"
        )

        # Set replay buffer shards (simplified for single-node setup)
        if "optimizer" not in self.hyper_parameters:
            self.hyper_parameters["optimizer"] = {}
        self.hyper_parameters["optimizer"]["num_replay_buffer_shards"] = 1

        self.log_message("✓ RASPBERry replay buffer configuration completed")

    def create_trainer(self):
        """Create DDQN trainer with RASPBERry"""
        self.log_message("Creating DDQN trainer with RASPBERry...")

        # Ensure configuration exists
        if not isinstance(self.hyper_parameters, dict):
            self.hyper_parameters = self._get_default_config()

        try:
            # Log trainer creation details
            self.log_message(f"Using algorithm: DDQNWithMPBER")
            self.log_message(f"Environment registry: Atari")

            # Create trainer with type ignore for complex config type
            self.trainer = DDQNWithMPBER(config=self.hyper_parameters, env="Atari")  # type: ignore

            self.log_message("✓ DDQN trainer created successfully!")

            # Log trainer info
            try:
                trainer_info = {
                    "algorithm": "DDQN with RASPBERry",
                    "framework": self.hyper_parameters.get("framework", "unknown"),
                    "num_workers": self.hyper_parameters.get("num_workers", "unknown"),
                    "replay_buffer_type": "MultiAgentPrioritizedBlockReplayBuffer",
                }
                self.log_message(f"Trainer info: {trainer_info}")
            except Exception as e:
                self.log_message(f"Could not log trainer info: {e}", "WARN")

        except Exception as e:
            self.log_message(f"Trainer creation failed: {e}", "ERROR")
            raise

    def format_training_status(
        self,
        iteration: int,
        result: Dict[str, Any],
        iteration_time: float,
        elapsed_time: float,
    ) -> str:
        """Format training status for tqdm display and logging"""

        # Extract key metrics
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_reward_max = result.get("episode_reward_max", 0)
        episode_reward_min = result.get("episode_reward_min", 0)
        episodes_total = result.get("episodes_total", 0)
        timesteps_total = result.get("timesteps_total", 0)

        # Format main status line
        status = (
            f"Iter {iteration:4d} | "
            f"Reward: {episode_reward_mean:7.2f} "
            f"({episode_reward_min:5.1f}-{episode_reward_max:5.1f}) | "
            f"Episodes: {episodes_total:5d} | "
            f"Steps: {timesteps_total:7d} | "
            f"{iteration_time:4.1f}s | "
            f"Total: {elapsed_time/60:5.1f}min"
        )

        return status

    def extract_detailed_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed metrics like in the notebook"""
        metrics = {}

        try:
            # Extract sampler results
            sampler = result.get("sampler_results", {})
            sam = {
                key: sampler[key] for key in self.keys_to_extract_sam if key in sampler
            }
            metrics.update(sam)

            # Extract status info
            info = result.get("info", {})
            sta = {key: info[key] for key in self.keys_to_extract_sta if key in info}
            metrics.update(sta)

            # Extract buffer statistics if available
            try:
                if self.trainer is not None and hasattr(
                    self.trainer, "local_replay_buffer"
                ):
                    replay_buffer = getattr(self.trainer, "local_replay_buffer")
                    if replay_buffer is not None and hasattr(replay_buffer, "stats"):
                        buf = flatten_dict(replay_buffer.stats())
                        buf["est_size_gb"] = buf.get("est_size_bytes", 0) / 1e9
                        buf_filtered = {
                            key: buf[key]
                            for key in self.keys_to_extract_buf
                            if key in buf
                        }
                        buf_filtered["est_size_gb"] = buf["est_size_gb"]
                        metrics.update(buf_filtered)
            except Exception as e:
                pass  # Buffer stats might not be available

            # Extract learning stats
            if "learner" in info:
                learner_info = info["learner"]
                if "default_policy" in learner_info:
                    policy_info = learner_info["default_policy"]
                    if "learner_stats" in policy_info:
                        stats = policy_info["learner_stats"]
                        learning_stats = {
                            "td_error": stats.get("td_error", 0),
                            "mean_q": stats.get("mean_q", 0),
                            "cur_epsilon": stats.get("cur_epsilon", 0),
                            "policy_loss": stats.get("policy_loss", 0),
                        }
                        metrics.update(learning_stats)

        except Exception as e:
            pass  # Some metrics might not be available

        return metrics

    def train(self):
        """Execute training loop with enhanced progress display"""
        self.log_message(f"Starting training for {self.max_run} iterations...")
        self.start_time = time.time()

        # Ensure trainer exists
        if self.trainer is None:
            raise ValueError("Trainer not initialized, call create_trainer() first")

        # Create enhanced tqdm progress bar
        progress_bar = tqdm(
            range(self.max_run),
            desc="🚀 DDQN Training",
            ascii=True,
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for i in progress_bar:
            try:
                iteration_start = time.time()
                result = self.trainer.train()
                iteration_time = time.time() - iteration_start
                elapsed_time = time.time() - self.start_time

                # Update global statistics
                episodes_this_iter = result.get("episodes_this_iter", 0)
                timesteps_this_iter = result.get("timesteps_this_iter", 0)
                episode_reward_mean = result.get("episode_reward_mean", 0)

                self.total_timesteps += timesteps_this_iter
                self.total_episodes += episodes_this_iter

                # Update best reward
                if episode_reward_mean > self.best_reward:
                    self.best_reward = episode_reward_mean

                # Update progress bar description with current metrics
                progress_desc = (
                    f"🚀 DDQN | Reward: {episode_reward_mean:6.2f} | "
                    f"Episodes: {self.total_episodes:5d} | "
                    f"Best: {self.best_reward:6.2f}"
                )
                progress_bar.set_description(progress_desc)

                # Extract detailed metrics
                detailed_metrics = self.extract_detailed_metrics(result)

                # Save detailed results every iteration (fine-grained logging)
                result_file = self.log_path / f"result_{i:06d}.json"
                with open(result_file, "w") as f:
                    result_copy = result.copy()
                    result_copy["config"] = None  # Remove config to save space
                    result_copy["iteration"] = i
                    result_copy["detailed_metrics"] = detailed_metrics
                    json.dump(convert_np_arrays(result_copy), f, indent=2)

                # Console logging frequency: write every 10 iterations
                log_frequency = 10

                if i % log_frequency == 0:
                    status = self.format_training_status(
                        i, result, iteration_time, elapsed_time
                    )
                    # Use tqdm.write to avoid interfering with progress bar
                    tqdm.write(status)

                    # Show detailed learning metrics if available
                    if detailed_metrics:
                        detail_parts = []
                        if "td_error" in detailed_metrics:
                            detail_parts.append(
                                f"Loss: {detailed_metrics['td_error']:.4f}"
                            )
                        if "mean_q" in detailed_metrics:
                            detail_parts.append(f"Q: {detailed_metrics['mean_q']:.2f}")
                        if "cur_epsilon" in detailed_metrics:
                            detail_parts.append(
                                f"ε: {detailed_metrics['cur_epsilon']:.4f}"
                            )
                        if "est_size_gb" in detailed_metrics:
                            detail_parts.append(
                                f"Buffer: {detailed_metrics['est_size_gb']:.2f}GB"
                            )

                        if detail_parts:
                            tqdm.write(f"     └─ {' | '.join(detail_parts)}")

                # Also log significant reward improvements immediately
                if episode_reward_mean > self.best_reward * 1.05:  # 5% improvement
                    improvement_status = self.format_training_status(
                        i, result, iteration_time, elapsed_time
                    )
                    tqdm.write(f"🎯 New best reward! {improvement_status}")
                    if detailed_metrics:
                        detail_parts = []
                        if "td_error" in detailed_metrics:
                            detail_parts.append(
                                f"Loss: {detailed_metrics['td_error']:.4f}"
                            )
                        if "mean_q" in detailed_metrics:
                            detail_parts.append(f"Q: {detailed_metrics['mean_q']:.2f}")
                        if "cur_epsilon" in detailed_metrics:
                            detail_parts.append(
                                f"ε: {detailed_metrics['cur_epsilon']:.4f}"
                            )
                        if detail_parts:
                            tqdm.write(f"     └─ {' | '.join(detail_parts)}")

                # Save checkpoint every 1000 iterations
                if i % 1000 == 0 and i > 0:
                    checkpoint_path = self.trainer.save(
                        checkpoint_dir=str(self.checkpoint_path)
                    )
                    tqdm.write(f"     💾 Checkpoint saved: {checkpoint_path}")
                    self.log_message(
                        f"Checkpoint saved at iteration {i}: {checkpoint_path}"
                    )

                # Emergency break condition (like in notebook)
                time_used = result.get("time_total_s", elapsed_time)
                max_time = getattr(self, "max_time", 360000)  # 100 hours default
                if time_used >= max_time:
                    tqdm.write(
                        f"⏰ Maximum time reached ({max_time}s), stopping training..."
                    )
                    break

            except Exception as e:
                error_msg = f"Training iteration {i} failed: {e}"
                tqdm.write(f"❌ {error_msg}")
                self.log_message(error_msg, "ERROR")
                break

        progress_bar.close()

        # Training summary
        total_training_time = time.time() - self.start_time
        self.log_message("🎉 Training completed!")
        self.log_message(f"Training Summary:")
        self.log_message(f"  Total iterations: {i+1}")
        self.log_message(f"  Total episodes: {self.total_episodes}")
        self.log_message(f"  Total timesteps: {self.total_timesteps}")
        self.log_message(f"  Best mean reward: {self.best_reward:.2f}")
        self.log_message(
            f"  Total training time: {total_training_time:.1f}s ({total_training_time/3600:.2f}h)"
        )
        self.log_message(
            f"  Average time per iteration: {total_training_time/(i+1):.2f}s"
        )

    def run(self):
        """Run complete training pipeline"""
        try:
            self.log_message("=" * 60)
            self.log_message("Starting RASPBERry DDQN Training Pipeline")
            self.log_message("=" * 60)

            self.setup_ray()
            self.setup_environment()
            self.load_config()
            self.setup_replay_buffer()
            self.create_trainer()
            self.train()

            self.log_message("=" * 60)
            self.log_message("Training Pipeline Completed Successfully!")
            self.log_message("=" * 60)

        except Exception as e:
            self.log_message(f"Training pipeline failed: {e}", "ERROR")
            raise
        finally:
            # Clean up resources
            if ray.is_initialized():
                self.log_message("Shutting down Ray...")
                ray.shutdown()
                self.log_message("✓ Ray shutdown completed")


def main():
    """Main function"""
    config = {
        "env_name": "BreakoutNoFrameskip-v4",
        "run_name": "DDQN_RASPBERry_test",
        "sub_buffer_size": 8,
        "compress_base": -1,  # Smart compression
    }

    trainer = RASPBERryDDQNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
