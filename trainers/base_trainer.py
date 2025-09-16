"""Base trainer class for RASPBERry project."""
import ray
import time
import logging
import warnings
from ray.tune.registry import register_env
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from utils import env_creator, load_config, flatten_dict
import mlflow


class BaseTrainer(ABC):
    """Abstract base class for reinforcement learning trainers."""

    def __init__(self,
                 config: str,
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 use_mlflow: Optional[str] = None):
        """Initialize common trainer state and I/O paths.

        Args:
            config: Path to YAML configuration file containing algorithm/env/buffer settings.
            env_name: Short environment name used in the `run_name`.
            log_path: Optional root directory to store logs; a subfolder `run_name` is created.
            checkpoint_path: Optional root directory to store checkpoints; a subfolder `run_name` is created.
            use_mlflow: Path to mlflow YAML configuration file, None to disable mlflow.

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

        # mlflow state
        self.use_mlflow = use_mlflow  # Path to mlflow config YAML or None
        self.mlflow_run = None

    def setup_environment(self) -> str:
        """Set up and register the training environment and return its ID."""
        env_str = self.env_name.split("-")[1].replace("NoFrameskip", "")
        register_env(env_str, env_creator)
        _env = env_creator({"id": self.env_name})
        reset_result = _env.reset()

        _env_rest = reset_result[0]
        self.log(f"✓ Environment {env_str} registered, obs: {_env_rest.shape}, act: {_env.action_space}", "TRAIN")

        return env_str

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
        # Log progress controlled purely by log_freq
        if self.log_freq and iteration % self.log_freq == 0:
            reward = stats.get("reward", "N/A")
            timesteps = stats.get("timesteps", "N/A")
            self.log(f"Iter {iteration:4d} | Reward: {reward} | Timesteps: {timesteps}", "TRAIN")
            # mlflow logging for this iteration (best-effort)
            self.log_mlflow_iteration(iteration, result)

        if self.current_iteration > self.checkpoint_freq and iteration % self.checkpoint_freq == 0 and self.checkpoint_path:
            checkpoint_path = self.checkpoint_path / f"checkpoint_{iteration}"
            self.trainer.save(str(checkpoint_path))
            self.log(f"Checkpoint saved: {checkpoint_path}", "CHECKPOINT")

        return result

    def _setup_mlflow(self) -> None:
        """Initialize mlflow from separate config file."""
        if not self.use_mlflow:
            return

        # Read separate mlflow configuration file
        mlflow_cfg = load_config(self.use_mlflow)

        if "tracking_uri" in mlflow_cfg:
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

        mlflow.set_experiment(experiment_name=self.env_name)

        run_tags = mlflow_cfg.get("run_tags", {})
        self.mlflow_run = mlflow.start_run(run_name=self.run_name, tags=run_tags)

        # Flatten and log hyperparameters
        flat_params = flatten_dict(self.config["hyper_parameters"])
        # Convert type objects to string names
        clean_params = {}
        for k, v in flat_params.items():
            if isinstance(v, type):
                clean_params[k] = str(v.__name__)
            else:
                clean_params[k] = v
        mlflow.log_params(clean_params)

        self.log("✓ mlflow ready", "TRAIN")

    def log_mlflow_iteration(self, iteration: int, result: Dict[str, Any]) -> None:
        """Log metrics to mlflow for a single iteration."""
        if not self.use_mlflow:
            return

        # Assemble metrics
        sampler = result.get("sampler_results", {})
        eva = result.get("evaluation", {})
        info = result.get("info", {})

        # Replay buffer statistics
        buf = {}
        if hasattr(self.trainer, 'local_replay_buffer'):
            buf = flatten_dict(self.trainer.local_replay_buffer.stats())
            if "est_size_bytes" in buf:
                buf["est_size_gb"] = buf["est_size_bytes"] / 1e9

        # Merge all numeric metrics
        metrics = {}
        for d in [sampler, info, buf]:
            metrics.update({k: v for k, v in d.items() if isinstance(v, (int, float))})

        # Add prefix for evaluation metrics
        for k, v in eva.items():
            if isinstance(v, (int, float)):
                metrics[f"eval_{k}"] = v

        step = result.get("episodes_total", iteration)
        mlflow.log_metrics(metrics, step=step)

        # Periodically upload artifacts
        mlflow_cfg = load_config(self.use_mlflow)
        log_artifacts_every = mlflow_cfg.get("log_artifacts_every", 200)
        if iteration % log_artifacts_every == 0:
            if self.log_path:
                mlflow.log_artifacts(str(self.log_path))
            if self.checkpoint_path:
                mlflow.log_artifacts(str(self.checkpoint_path))

    def finalize_mlflow(self) -> None:
        """Upload final artifacts and end mlflow run."""
        if not self.use_mlflow:
            return

        # if self.log_path:
        #     mlflow.log_artifacts(str(self.log_path))
        # if self.checkpoint_path:
        #     mlflow.log_artifacts(str(self.checkpoint_path))

        mlflow.end_run()

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
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")

        if not ray.is_initialized():
            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                include_dashboard=include_dashboard,
                _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
                ignore_reinit_error=True
            )
            self.log(f"✓ Ray ready ({num_cpus} CPUs, {num_gpus} GPUs)")

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
        self.log(f"🚀 Training {start_iter}→{end_iter}{time_info}", "TRAIN")

        # Setup mlflow on first train call
        if self.current_iteration == start_iter and self.use_mlflow:
            self._setup_mlflow()

        for iteration in range(start_iter, end_iter):
            if max_time and (time.time() - train_start_time) >= max_time:
                self.log(f"⏰ Time limit reached ({max_time}s), stopping training", "TRAIN")
                break
            result = self._train_single_iteration(iteration)

        elapsed_time = time.time() - train_start_time
        self.log(f"✅ Done ({elapsed_time:.1f}s)", "TRAIN")

        # Finalize mlflow
        self.finalize_mlflow()

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
            self.setup_ray(num_cpus=self.config["hyper_parameters"]["num_cpus"], num_gpus=self.config["hyper_parameters"]["num_gpus"])
            self.init_algorithm()

        self.train(max_iterations=max_iterations, max_time=max_time)
