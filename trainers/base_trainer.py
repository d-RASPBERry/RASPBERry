"""Base trainer class for RASPBERry project."""

import ray
import time
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dynaconf import Dynaconf
from pathlib import Path
from utils import load_config


class BaseTrainer(ABC):
    """Abstract base class for trainers supporting both RASPBERry and Ray PER modes."""

    def __init__(self,
                 config: str,
                 env_name: str,
                 buffer_type: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None):
        """Initialize common trainer state and I/O paths.

        Args:
            config: High-level str configuration for algorithm/env/buffer.
            env_name: Short environment name used in the `run_name`.
            buffer_type: Replay buffer type label (e.g., "RASPBERry", "PER").
            log_path: Optional root directory to store logs; a subfolder `run_name` is created.
            checkpoint_path: Optional root directory to store checkpoints; a subfolder `run_name` is created.

        Notes:
            - `run_name` is `{env_name}_YYYYMMDD_HHMMSS`.
            - Subclasses must later set up the concrete RLlib `self.trainer` in `init_algorithm()`.
        """
        self.start_time = time.time()
        self.env_name = env_name
        self.buffer_type = buffer_type

        self.config = self._setup_config(config, buffer_type)
        self.run_name = f"{env_name}_{time.time_ns()}"
        self.trainer = None

        self.log_freq = self.config.get("log_freq")
        self.checkpoint_freq = self.config.get("checkpoint_freq")
        self.current_iteration = 0
        self.log_path = Path(log_path) / self.run_name if log_path else None
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = Path(checkpoint_path) / self.run_name if checkpoint_path else None
        if self.checkpoint_path:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self.logger.info(f"🚀 Initializing {self.__class__.__name__}: {self.run_name} ({self.buffer_type})")

    def _setup_config(self, config: str, buffer_type: str) -> Dynaconf:
        """Load configuration and validate buffer compatibility.

        Default implementation provides common config loading using utils.config_loader.
        Subclasses can override to add algorithm-specific validation.

        Args:
            config: Path to YAML configuration file.
            buffer_type: Replay buffer type label for compatibility validation.

        Returns:
            Loaded Dynaconf configuration object.

        Expected responsibilities:
            - Load configuration from file path.
            - Validate `buffer_type` compatibility with the algorithm.
            - Apply/merge buffer-specific settings if needed.
            - Prepare any fields later used by `init_algorithm()`.
        """
        config_dict = load_config(config)
        return Dynaconf(settings=config_dict)

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
            self.logger.info(message)

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

        if iteration % self.log_freq == 0:
            reward = stats.get("reward", "N/A")
            timesteps = stats.get("timesteps", "N/A")
            self.log(f"Iter {iteration:4d} | Reward: {reward} | Timesteps: {timesteps}", "TRAIN")

        if iteration % self.checkpoint_freq == 0 and self.checkpoint_path:
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
