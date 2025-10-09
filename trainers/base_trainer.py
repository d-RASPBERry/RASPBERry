"""Base trainer class for RASPBERry project."""
import ray
import time
import json
import math
import logging
import warnings
import datetime
from ray.tune.registry import register_env
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from utils import env_creator, flatten_dict, convert_np_arrays
import mlflow


class BaseTrainer(ABC):
    """Abstract base class for reinforcement learning trainers."""

    def __init__(self,
                 config: Dict[str, Any],
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 mlflow_cfg: Optional[Dict[str, Any]] = None):
        """Initialize common trainer state and I/O paths.

        Args:
            config: Configuration dictionary containing algorithm/env/buffer settings.
            env_name: Short environment name used in the `run_name`.
            log_path: Optional root directory to store logs; a subfolder `run_name` is created.
            checkpoint_path: Optional root directory to store checkpoints; a subfolder `run_name` is created.
            mlflow_cfg: Optional dictionary with MLflow configuration.

        Notes:
            - `run_name` is `{env_name}_YYYYMMDD_HHMMSS`.
            - Subclasses must implement `_filter_result()` and `init_algorithm()`.
        """
        # Set default values for logging and checkpointing if not present
        if config.get('log_freq') is None:
            config['log_freq'] = 10
        if config.get('checkpoint_freq') is None:
            config['checkpoint_freq'] = 100
        self.config = config
        self.start_time = time.time()
        self.env_name = env_name
        self.current_iteration = 0
        self.run_name = f"{run_name}_{time.time_ns()}"
        self.trainer = None
        self.log_freq = self.config.get("log_freq")
        self.checkpoint_freq = self.config.get("checkpoint_freq")
        self.log_start_iteration = self.config.get("log_start_iteration", 0)

        # mlflow state must be set before logger setup
        self.mlflow_cfg = mlflow_cfg
        self.mlflow_run = None

        self.log_path = Path(log_path) / self.run_name if log_path else None
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = Path(checkpoint_path) / self.run_name if checkpoint_path else None
        if self.checkpoint_path:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._setup_logger()
        self.logger.info(f"🚀 Initializing {self.__class__.__name__}: {self.run_name}")

    def setup_environment(self) -> str:
        """Set up and register the training environment and return its ID."""
        env_str = self.env_name.split("-")[1].replace("NoFrameskip", "")
        register_env(env_str, env_creator)
        _env = env_creator({"id": self.env_name})
        reset_result = _env.reset()

        _env_rest = reset_result[0]
        self.log(f"✓ Environment {env_str} registered, obs: {_env_rest.shape}, act: {_env.action_space}", "TRAIN")

        return env_str

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

    @staticmethod
    def _prepare_result_metadata(iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare result dictionary with standard metadata.
        
        This is a pure function that adds iteration, timestamp, and other common fields.
        
        Args:
            iteration: Current training iteration.
            result: Raw result dictionary from trainer.
            
        Returns:
            Result dictionary with added metadata.
        """
        data = result.copy()
        data["iteration"] = iteration
        data["timestamp"] = time.time()
        data["episodes"] = result.get("episodes_total", "N/A")
        data["time_s"] = result.get("time_total_s", "N/A")
        return data

    def _get_buffer_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract buffer statistics from the replay buffer.
        
        Uses `buffer.stats()` as the single source of truth. For custom compressed
        buffers (e.g., RASPBERry), compression-related metrics are expected to be
        included in the `stats()` output.
        """
        stats = {}
        if hasattr(self.trainer, 'local_replay_buffer') and self.trainer.local_replay_buffer is not None:
            buffer = self.trainer.local_replay_buffer
            # Single-node mode
            buffer_stats = buffer.stats()

            # Normalize/alias common keys
            # num_entries
            if "num_entries" not in buffer_stats:
                if "num_items" in buffer_stats and isinstance(buffer_stats["num_items"], (int, float)):
                    buffer_stats["num_entries"] = buffer_stats["num_items"]
                elif hasattr(buffer, '__len__'):
                    try:
                        buffer_stats["num_entries"] = len(buffer)
                    except Exception:
                        pass
            # est_size_bytes
            if "est_size_bytes" not in buffer_stats:
                if hasattr(buffer, 'get_estimated_size_in_bytes'):
                    try:
                        buffer_stats["est_size_bytes"] = buffer.get_estimated_size_in_bytes()
                    except Exception:
                        pass
                elif "size_bytes" in buffer_stats and isinstance(buffer_stats["size_bytes"], (int, float)):
                    buffer_stats["est_size_bytes"] = buffer_stats["size_bytes"]
                elif "total_size_bytes" in buffer_stats and isinstance(buffer_stats["total_size_bytes"], (int, float)):
                    buffer_stats["est_size_bytes"] = buffer_stats["total_size_bytes"]

            stats["buffer_stats"] = buffer_stats
        elif "info" in result and "replay_shard_0" in result["info"]:
            # Ape-X distributed mode (when available in `result`)
            stats["buffer_stats"] = result["info"]["replay_shard_0"]
        return stats

    def _process_train_result(self, iteration: int, result: Dict[str, Any]) -> (Dict[str, Any], Dict[str, float]):
        """Consolidates all result processing for logging and file saving.
        
        This is the new Single Source of Truth for handling the raw result dictionary.
        
        Args:
            iteration: Current training iteration.
            result: Raw result dictionary from trainer.
            
        Returns:
            A tuple containing:
            - full_log_data: A structured dict for file saving.
            - mlflow_metrics: A flattened dict of numeric metrics for MLflow.
        """
        # 1. Prepare metadata and buffer stats
        full_log_data = self._prepare_result_metadata(iteration, result)
        buffer_stats = self._get_buffer_stats(result)
        full_log_data.update(buffer_stats)

        # 2. Prepare metrics for MLflow by flattening all nested dictionaries
        sampler = result.get("sampler_results", {})
        eva = result.get("evaluation", {})
        info = result.get("info", {})

        # Flatten all potential sources of numeric data
        metrics: Dict[str, Any] = {}
        metrics.update(flatten_dict(sampler))
        metrics.update(flatten_dict(info))
        metrics.update(flatten_dict(buffer_stats))

        # Add derived/calculated metrics
        if "buffer_stats_est_size_bytes" in metrics:
            size_bytes = metrics["buffer_stats_est_size_bytes"]
            if isinstance(size_bytes, (int, float)):
                metrics["buffer_stats_est_size_gb"] = size_bytes / 1e9

        # Add prefix for evaluation metrics
        for k, v in flatten_dict(eva).items():
            metrics[f"eval_{k}"] = v

        # Filter for numeric types and remove non-finite values to prevent logger errors
        mlflow_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v is not None and math.isfinite(v)
        }

        return full_log_data, mlflow_metrics

    def _save_result_file(self, full_log_data: Dict[str, Any]) -> None:
        """Save the processed log data to a JSON file."""
        if not self.log_path:
            return

        try:
            iteration = full_log_data.get("iteration", 0)
            processed_data = convert_np_arrays(full_log_data)
            result_file = self.log_path / f"result_{iteration:06d}.json"

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)

        except (IOError, OSError) as e:
            self.log(f"结果文件保存失败: {e}", "BUFFER")
        except Exception as e:
            self.log(f"结果保存时发生意外错误: {e}", "BUFFER")

    def _train_single_iteration(self, iteration: int) -> None:
        """Run a single `train()` step, process results, log, and checkpoint."""
        # 1. Run a single training step
        result = self.trainer.train()
        self.current_iteration = iteration + 1

        # 2. Check if we should skip logging based on the configured start iteration
        if self.current_iteration < self.log_start_iteration:
            if self.log_freq and iteration % self.log_freq == 0:
                self.log(f"Iter {iteration:4d} | Skipping detailed logging until iteration {self.log_start_iteration}",
                         "TRAIN")
            return

        # 3. Process the raw result ONCE to get all necessary outputs
        full_log_data, mlflow_metrics = self._process_train_result(iteration, result)

        # 4. Save the complete, processed result to a file
        self._save_result_file(full_log_data)

        # 5. Log to console
        if self.log_freq and iteration % self.log_freq == 0:
            reward = result.get("episode_reward_mean", "N/A")
            timesteps = result.get("timesteps_total", "N/A")
            self.log(f"Iter {iteration:4d} | Reward: {reward} | Timesteps: {timesteps}", "TRAIN")

            # 6. Log processed metrics to MLflow
            self.log_mlflow_iteration(iteration, result, mlflow_metrics)

        # 7. Handle checkpointing
        if self.current_iteration > self.checkpoint_freq and iteration % self.checkpoint_freq == 0 and self.checkpoint_path:
            checkpoint_path = self.checkpoint_path / f"checkpoint_{iteration}"
            self.trainer.save(str(checkpoint_path))
            self.log(f"Checkpoint saved: {checkpoint_path}", "CHECKPOINT")

    def _setup_mlflow(self) -> None:
        """Initialize mlflow from the provided config dictionary."""
        if not self.mlflow_cfg:
            return

        try:
            import mlflow

            if "tracking_uri" in self.mlflow_cfg:
                mlflow.set_tracking_uri(self.mlflow_cfg["tracking_uri"])

            mlflow.set_experiment(experiment_name=self.env_name)

            run_tags = self.mlflow_cfg.get("run_tags", {})
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
        except ImportError:
            self.log("MLflow is configured, but the 'mlflow' package is not installed. Disabling MLflow.", "TRAIN")
            self.mlflow_cfg = None
        except Exception as e:
            self.log(f"Failed to initialize MLflow (is the server running?): {e}", "TRAIN")
            self.log("Disabling MLflow logging for this run.", "TRAIN")
            self.mlflow_cfg = None

    def log_mlflow_iteration(self, iteration: int, result: Dict[str, Any], mlflow_metrics: Dict[str, float]) -> None:
        """Log pre-processed metrics to mlflow for a single iteration."""
        if not self.mlflow_cfg:
            return

        step = result.get("episodes_total", iteration)
        mlflow.log_metrics(mlflow_metrics, step=step)

        # Periodically upload artifacts
        log_artifacts_every = self.mlflow_cfg.get("log_artifacts_every", 200)
        if iteration % log_artifacts_every == 0:
            if self.log_path:
                mlflow.log_artifacts(str(self.log_path))
            if self.checkpoint_path:
                mlflow.log_artifacts(str(self.checkpoint_path))

    def finalize_mlflow(self) -> None:
        """Upload final artifacts and end mlflow run."""
        if not self.mlflow_cfg:
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
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")

        if not ray.is_initialized():
            # Get Ray temp dir with timestamp
            ray_temp_base = self.config["paths"].get("ray_temp_dir", "/tmp/ray/")
            ray_temp_dir = f"{ray_temp_base}ray_{int(datetime.datetime.now().timestamp())}"

            # Get system config
            max_cached_count = self.config["ray_config"].get("max_gcs_destroyed_actor_cached_count", 100)

            # Get object store memory setting (in GB, convert to bytes)
            object_store_memory_gb = self.config["ray_config"].get("object_store_memory_gb", 100)
            object_store_memory_bytes = object_store_memory_gb * 1024 * 1024 * 1024

            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                include_dashboard=include_dashboard,
                object_store_memory=object_store_memory_bytes,
                _temp_dir=ray_temp_dir,
                _system_config={"maximum_gcs_destroyed_actor_cached_count": max_cached_count},
                ignore_reinit_error=True
            )
            self.log(
                f"✓ Ray ready ({num_cpus} CPUs, {num_gpus} GPUs, {object_store_memory_gb}GB Object Store, temp: {ray_temp_dir})")

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
        if self.current_iteration == start_iter and self.mlflow_cfg:
            self._setup_mlflow()

        for iteration in range(start_iter, end_iter):
            if max_time and (time.time() - train_start_time) >= max_time:
                self.log(f"⏰ Time limit reached ({max_time}s), stopping training", "TRAIN")
                break
            self._train_single_iteration(iteration)

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
            self.setup_ray(num_cpus=self.config["hyper_parameters"]["num_cpus"],
                           num_gpus=self.config["hyper_parameters"]["num_gpus"])
            self.init_algorithm()

        self.train(max_iterations=max_iterations, max_time=max_time)
