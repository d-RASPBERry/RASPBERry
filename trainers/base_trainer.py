"""Base trainer class for RASPBERry project."""
import ray
import time
import logging
import warnings
from ray.tune.registry import register_env
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from utils import env_creator, load_config


class BaseTrainer(ABC):
    """Abstract base class for reinforcement learning trainers."""

    def __init__(self,
                 config: str,
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None):
        """Initialize common trainer state and I/O paths.

        Args:
            config: Path to YAML configuration file containing algorithm/env/buffer settings.
            env_name: Short environment name used in the `run_name`.
            log_path: Optional root directory to store logs; a subfolder `run_name` is created.
            checkpoint_path: Optional root directory to store checkpoints; a subfolder `run_name` is created.

        Notes:
            - `run_name` is `{env_name}_YYYYMMDD_HHMMSS`.
            - Subclasses must implement `_filter_result()` and `init_algorithm()`.
        """
        self._setup_config(config)
        self.start_time = time.time()
        self.env_name = env_name
        self.current_iteration = 0
        self.run_name = f"{run_name}_{time.time_ns()}"
        self.trainer = None
        self.log_freq = self.config.get("log_freq")
        self.checkpoint_freq = self.config.get("checkpoint_freq")
        self.log_path = Path(log_path) / self.run_name if log_path else None
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = Path(checkpoint_path) / self.run_name if checkpoint_path else None
        if self.checkpoint_path:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self.logger.info(f"🚀 Initializing {self.__class__.__name__}: {self.run_name}")

    def setup_environment(self) -> str:
        """设置和注册训练环境，返回环境ID。"""
        self.log(f"设置环境{self.env_name}", "TRAIN")
        register_env(self.env_name, env_creator)
        self.log(f"✓ 环境 {self.env_name} 注册成功", "TRAIN")
        return self.env_name

    def _setup_config(self, config: str):
        """Load configuration from YAML file.

        Args:
            config: Path to YAML configuration file containing all settings.

        Returns:
            Configuration dictionary.
        """
        # Load configuration from YAML file path
        config_dict = load_config(config)

        # Set default values for logging and checkpointing if not present
        if config_dict.get('log_freq') is None:
            config_dict['log_freq'] = 10
        if config_dict.get('checkpoint_freq') is None:
            config_dict['checkpoint_freq'] = 100
        self.config = config_dict

    def _setup_logger(self) -> None:
        """Configure a stream logger and optional file logger under `log_path`.

        The logger name is `RASPBERry.{run_name}` and uses a concise time-prefixed format.
        """
        self.logger = logging.getLogger(f"RASPBERry.{self.run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s', datefmt='%H:%M:%S')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if self.log_path:
            file_handler = logging.FileHandler(self.log_path / "training.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, message: str, category: str = "INFO") -> None:
        """Light wrapper to standardize log categories.

        Known categories: "TRAIN", "BUFFER", "CHECKPOINT". Any other value falls back to plain INFO.
        """
        if category in ["TRAIN", "BUFFER", "CHECKPOINT"]:
            self.logger.info(f"[{category}] {message}")
        else:
            self.logger.info(f"[UNKNOWN] {message}")

    @abstractmethod
    def _filter_result(self, iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Map RLlib training `result` into a compact stats dict for logging.

        Args:
            iteration: Current training iteration index.
            result: Raw dict returned by `self.trainer.train()`.

        Returns:
            A small dict containing at least the following keys for unified logging:
                - "reward": scalar reward metric (e.g., episode_reward_mean)
                - "timesteps": cumulative or per-iteration timesteps metric

        Subclasses decide the exact mapping from RLlib fields to these keys.
        """
        raise NotImplementedError("Subclasses must implement _filter_result()")

    def _train_single_iteration(self, iteration: int) -> None:
        """Run a single `train()` step, log progress, and checkpoint if configured."""
        result = self.trainer.train()
        self.current_iteration = iteration + 1
        stats = self._filter_result(iteration, result)
        # skip first 50 iter log
        if self.current_iteration > self.checkpoint_freq and iteration % self.log_freq == 0:
            reward = stats.get("reward", "N/A")
            timesteps = stats.get("timesteps", "N/A")
            self.log(f"Iter {iteration:4d} | Reward: {reward} | Timesteps: {timesteps}", "TRAIN")

        if self.current_iteration > self.checkpoint_freq and iteration % self.checkpoint_freq == 0 and self.checkpoint_path:
            checkpoint_path = self.checkpoint_path / f"checkpoint_{iteration}"
            self.trainer.save(str(checkpoint_path))
            self.log(f"Checkpoint saved: {checkpoint_path}", "CHECKPOINT")

    def setup_ray(self, num_cpus: int = 5, num_gpus: int = 1,
                  include_dashboard: bool = False) -> None:
        """Initialize a local Ray cluster if not already initialized.

        Args:
            num_cpus: Number of CPUs to allocate to Ray.
            num_gpus: Number of GPUs to allocate to Ray.
            include_dashboard: Whether to start the Ray dashboard.

        Notes:
            Tweaks `_system_config` for GC actor cache to avoid warnings on teardown.
        """
        self.log("Initializing Ray cluster...")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")

        if not ray.is_initialized():
            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                include_dashboard=include_dashboard,
                _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
                ignore_reinit_error=True
            )
            self.log(f"✓ Ray initialized with {num_cpus} CPUs, {num_gpus} GPUs")
        else:
            self.log("Ray already initialized")

    @abstractmethod
    def init_algorithm(self) -> Any:
        """Create and assign the concrete RLlib trainer to `self.trainer`.

        Subclasses typically build RLlib `Trainer` with `self.config` and environment bindings.
        Must set `self.trainer` before calling `train()`.
        """
        raise NotImplementedError("Subclasses must implement init_algorithm()")

    def train(self, max_iterations: Optional[int] = None, max_time: Optional[float] = None) -> None:
        """Main training loop with optional iteration/time limits.

        Args:
            max_iterations: Total iterations to run from the current iteration (default 1000 if None).
            max_time: Optional wall-clock seconds budget; stops early if exceeded.

        Behavior:
            - Logs progress every `log_freq` iterations using `_filter_result()` output.
            - Saves checkpoints every `checkpoint_freq` iterations if `checkpoint_path` is set.
        """
        if self.trainer is None:
            raise RuntimeError("Algorithm not setup. Call init_algorithm() first.")

        start_iter = self.current_iteration
        end_iter = start_iter + (max_iterations or 1000)
        train_start_time = time.time()

        time_info = f" (max {max_time}s)" if max_time else ""
        self.log(f"🚀 Training from iteration {start_iter} to {end_iter}{time_info}...", "TRAIN")

        for iteration in range(start_iter, end_iter):
            if max_time and (time.time() - train_start_time) >= max_time:
                self.log(f"⏰ Time limit reached ({max_time}s), stopping training", "TRAIN")
                break
            self._train_single_iteration(iteration)

        elapsed_time = time.time() - train_start_time
        self.log(f"✅ Training completed in {elapsed_time:.1f}s", "TRAIN")

    def run(self,
            max_iterations: Optional[int] = None,
            max_time: Optional[float] = None,
            initialize: bool = True) -> None:
        """Convenience entry point: optionally initialize, then train.

        Args:
            max_iterations: See `train`.
            max_time: See `train`.
            initialize: If True, calls `setup_ray()` and `init_algorithm()` before training.
        """
        if initialize:
            self.setup_ray()
            self.init_algorithm()

        self.train(max_iterations=max_iterations, max_time=max_time)
