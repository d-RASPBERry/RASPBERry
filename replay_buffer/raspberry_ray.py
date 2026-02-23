"""RASPBERry replay buffer with Ray-based compression.

Prioritized block replay buffer with on-the-fly blosc compression.
"""

# ====== Section: Imports ======
import time
from typing import Any, Dict, List, Optional

# ------ Subsection: Third-party ------
import blosc
import numpy as np
import ray
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from ray.rllib.utils.typing import SampleBatchType

# ------ Subsection: Local ------
from replay_buffer.compress_replay_node import CompressReplayNode


# ====== Section: Ray Remote Compression Worker ======

@ray.remote
def compress_node_batch_ray(nodes_data: List[dict],
                            compress_config: dict) -> List[dict]:
    """Ray remote worker to compress a batch of nodes.

    Args:
        nodes_data: List of node data dicts (obs, new_obs, actions, etc.).
        compress_config: Compression config (cname, clevel, shuffle, compress_base,
            enable_compression).

    Returns:
        List of dicts with compressed_data (SampleBatch), weight, and byte counts.
    """
    batch_results = []
    
    for node_data in nodes_data:
        obs = node_data['obs']
        new_obs = node_data['new_obs']
        
        raw_obs_bytes = int(node_data.get('raw_obs_bytes', obs.nbytes))
        raw_new_obs_bytes = int(node_data.get('raw_new_obs_bytes', new_obs.nbytes))
        
        rank = len(obs.shape)
        if rank > 1:
            if compress_config['compress_base'] == -1:
                axes = list(range(1, rank)) + [0]
                obs = np.transpose(obs, axes)
                new_obs = np.transpose(new_obs, axes)
        
        obs = np.ascontiguousarray(obs)
        new_obs = np.ascontiguousarray(new_obs)
        
        compressed_obs = blosc.pack_array(
            obs,
            cname=compress_config['cname'],
            clevel=compress_config['clevel'],
            shuffle=compress_config['shuffle'],
        )
        compressed_new_obs = blosc.pack_array(
            new_obs,
            cname=compress_config['cname'],
            clevel=compress_config['clevel'],
            shuffle=compress_config['shuffle'],
        )
        
        compressed_data = SampleBatch({
            "obs": np.array([compressed_obs], dtype=object),
            "new_obs": np.array([compressed_new_obs], dtype=object),
            "actions": node_data['actions'],
            "rewards": node_data['rewards'],
            "terminateds": node_data['terminateds'],
            "truncateds": node_data['truncateds'],
            "weights": node_data['weights'],
            "compress_base": compress_config['compress_base'],
            "is_compressed": True,
        })
        
        result = {
            'compressed_data': compressed_data,
            'weight': node_data['weight'],
            'obs_bytes': len(compressed_obs),
            'new_obs_bytes': len(compressed_new_obs),
            'raw_obs_bytes': raw_obs_bytes,
            'raw_new_obs_bytes': raw_new_obs_bytes,
            'raw_total_bytes': raw_obs_bytes + raw_new_obs_bytes,
        }
        batch_results.append(result)
    
    return batch_results


# ====== Section: Decompression ======
def decompress_sample_batch(ma_batch: SampleBatch, compress_base: int = -1) -> SampleBatch:
    """Decompress obs/new_obs using blosc and restore axes.

    Args:
        ma_batch: Compressed batch with obs/new_obs as blosc-packed arrays
            (single or multiple blocks).
        compress_base: Axis transposition hint (-1 for auto, 0 for none).

    Returns:
        Decompressed SampleBatch with expanded weights/batch_indexes.

    Notes:
        weights/batch_indexes should already be expanded to transition-level
        by the buffer's sample() method before calling this function.
        Handles both single-block (DDQN via sample_min_n_steps_from_buffer)
        and multi-block (APEX direct sample) cases.
    """
    compress_base_used = ma_batch.get("compress_base", compress_base)

    obs_array = ma_batch["obs"]
    new_obs_array = ma_batch["new_obs"]
    # Support both single-block (DDQN) and multi-block (APEX) layouts.
    if len(obs_array) == 1 and isinstance(obs_array[0], bytes):
        decompressed_obs_transposed = blosc.unpack_array(obs_array[0])
        decompressed_new_obs_transposed = blosc.unpack_array(new_obs_array[0])
    else:
        obs_blocks = [blosc.unpack_array(block) for block in obs_array if isinstance(block, bytes)]
        new_obs_blocks = [blosc.unpack_array(block) for block in new_obs_array if isinstance(block, bytes)]
        
        if not obs_blocks:
            decompressed_obs_transposed = blosc.unpack_array(obs_array[0])
            decompressed_new_obs_transposed = blosc.unpack_array(new_obs_array[0])
        else:
            # Concatenate along the batch dimension (axis 0 for transposed, last axis after transpose)
            decompressed_obs_transposed = np.concatenate(obs_blocks, axis=-1)
            decompressed_new_obs_transposed = np.concatenate(new_obs_blocks, axis=-1)

    rank = len(decompressed_obs_transposed.shape)

    # Default path reverses the batch-axis move used for compression.
    if compress_base_used == -1:
        # Smart default: move axis 0 (batch) from end back to front
        if rank <= 1:
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
        else:
            # Multi-dim: move last axis (original axis 0) back to the front
            axes = [rank - 1] + list(range(rank - 1))
            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
    elif compress_base_used == 0:
        decompressed_obs = decompressed_obs_transposed
        decompressed_new_obs = decompressed_new_obs_transposed
    else:
        if compress_base_used >= rank or rank <= 1:
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
        else:
            # Insert the last axis back at compress_base position
            axes = list(range(rank))
            axes.pop()  # Remove last dimension
            axes.insert(compress_base_used, rank - 1)  # Insert at specified position

            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)

    # Simply pass through all other fields (weights and batch_indexes should already
    # be expanded to transition-level by PrioritizedBlockReplayBuffer.sample())
    data_dict = {
        "obs": decompressed_obs,
        "new_obs": decompressed_new_obs,
        "actions": ma_batch["actions"],
        "rewards": ma_batch["rewards"],
        "terminateds": ma_batch["terminateds"],
        "truncateds": ma_batch["truncateds"],
        "weights": ma_batch["weights"],
    }

    if "batch_indexes" in ma_batch:
        data_dict["batch_indexes"] = ma_batch["batch_indexes"]

    return SampleBatch(data_dict)


# ====== Section: RAM Saver Prioritized Block Replay Buffer ======

class RASPBERryReplayBuffer(PrioritizedReplayBuffer):
    """RASPBERry replay buffer with block-level compression.

    Uses Ray workers to compress blocks before storage.
    """

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 32,
            compress_base: int = -1,
            compress_pool_size: int = 5,
            num_ray_workers: int = 1,
            compression_algorithm: str = "zstd",
            compression_level: int = 5,
            compression_nthreads: int = 1,
            compression_mode: str = "C",  # A: sync, B: batch sync, C: async
            chunk_size: int = 10,
            max_inflight_tasks: Optional[int] = None,
            **kwargs,
    ):
        """Initialize the prioritized block replay buffer.

        Args:
            obs_space: Observation space.
            action_space: Action space.
            sub_buffer_size: Block size (transitions per block).
            compress_base: Axis to move for compression (-1 for auto).
            compress_pool_size: Legacy field (ignored for worker count).
            num_ray_workers: Max concurrent Ray compression tasks.
            compression_algorithm: Blosc algorithm (zstd, lz4, etc.).
            compression_level: Compression level (1-9).
            compression_nthreads: Blosc compression threads.
            compression_mode:
                - "A": synchronous compression
                - "B": batch synchronous compression
                - "C": async Ray compression
            chunk_size: Number of nodes per Ray task batch.
            max_inflight_tasks: Backpressure cap for async Ray tasks.
                If omitted, defaults to `num_ray_workers * 2`.
            **kwargs: Additional args for PrioritizedReplayBuffer.
        """
        # Map repo config keys to RLlib's PrioritizedReplayBuffer args.
        # - Configs use `prioritized_replay_alpha/beta`.
        # - RLlib expects `alpha`, and `beta` is provided in sample().
        if "prioritized_replay_alpha" in kwargs and "alpha" not in kwargs:
            kwargs["alpha"] = kwargs.pop("prioritized_replay_alpha")
        # Keep beta as an attribute so callers can omit beta in sample().
        if "prioritized_replay_beta" in kwargs:
            self.beta = float(kwargs.pop("prioritized_replay_beta"))

        super(RASPBERryReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        self.obs_space = obs_space
        self.action_space = action_space

        mode = str(compression_mode).upper()
        # External config mode (A/B/C) -> internal execution mode.
        # Internal modes remain B/C/D for minimal code churn:
        # - internal "B": sync compression
        # - internal "C": batch Ray sync
        # - internal "D": async Ray
        mode_map = {
            "A": "B",
            "B": "C",
            "C": "D",
        }
        self._compression_mode_input = mode
        if mode not in mode_map:
            raise ValueError(
                f"Invalid compression_mode={compression_mode!r}. "
                "Supported modes are: A (sync), B (batch sync), C (async)."
            )
        self._compression_mode = mode_map[mode]
        self._chunk_size = max(1, int(chunk_size))
        
        # All modes now use compression
        self._enable_compression = True
        
        if self._compression_mode == "B":
            self._num_ray_workers = 0  # No Ray for sync compression
        elif self._compression_mode in ["C", "D"]:
            self._num_ray_workers = max(1, int(num_ray_workers))
            if not ray.is_initialized():
                ray.init(num_cpus=self._num_ray_workers, ignore_reinit_error=True)

        # Keep legacy behavior by default, but allow explicit backpressure cap.
        if self._compression_mode == "D":
            default_max_inflight = self._num_ray_workers * 2
            if max_inflight_tasks is None:
                self._max_inflight_tasks = int(default_max_inflight)
            else:
                self._max_inflight_tasks = max(1, int(max_inflight_tasks))
        else:
            self._max_inflight_tasks = 0

        self._inflight_futures: List[ray.ObjectRef] = []
        self._pending_nodes: List[CompressReplayNode] = []

        self._metrics = {
            "compress_time_ms": 0.0,
            "compress_count": 0,
            "compress_bytes_obs": 0,
            "compress_bytes_new_obs": 0,
            "backpressure_wait_ms": 0.0,
            "decompress_time_ms": 0.0,
            "decompress_count": 0,
            "raw_bytes": 0,
        }

        # Track storage footprint (compressed vs estimated raw) to expose via stats.
        self._est_compressed_bytes = 0
        self._est_raw_bytes = 0

        self._compression_config = {
            'cname': compression_algorithm,
            'clevel': compression_level,
            'nthreads': compression_nthreads,
            'compress_base': compress_base,
            'shuffle': blosc.BITSHUFFLE,
            'enable_compression': self._enable_compression,
        }

        self.compress_node = self._create_compress_node()

    def _create_compress_node(self):
        """Create a new compress node with consistent parameters."""
        return CompressReplayNode(
            buffer_size=self.sub_buffer_size,
            obs_space=self.obs_space,
            action_space=self.action_space,
            compress_base=self.compress_base,
            cname=self._compression_config['cname'],
            compression_level=self._compression_config['clevel'],
            nthreads=self._compression_config['nthreads'],
            enable_compression=self._enable_compression,
        )

    def _node_to_dict(self, node: CompressReplayNode) -> dict:
        """Convert a CompressReplayNode to a serializable dict."""
        size = node.size()
        obs_slice = node.obs[:size].copy()
        new_obs_slice = node.new_obs[:size].copy()
        return {
            'obs': obs_slice,
            'new_obs': new_obs_slice,
            'actions': node.actions[:size].copy(),
            'rewards': node.rewards[:size].copy(),
            'terminateds': node.terminateds[:size].copy(),
            'truncateds': node.truncateds[:size].copy(),
            'weights': node.weights[:size].copy(),
            'weight': np.mean(node.weights[:size]),
            'raw_obs_bytes': int(obs_slice.nbytes),
            'raw_new_obs_bytes': int(new_obs_slice.nbytes),
        }

    def _compress_mode_b(self):
        """Internal mode B (external mode A): synchronous compression."""
        t0 = time.time()
        compressed_data, weight, metrics = self.compress_node.sample()
        self._metrics["compress_time_ms"] += (time.time() - t0) * 1000.0
        self._metrics["compress_bytes_obs"] += len(compressed_data["obs"][0])
        self._metrics["compress_bytes_new_obs"] += len(compressed_data["new_obs"][0])
        self._metrics["compress_count"] += 1
        self._add_compressed_batch(compressed_data, weight,
                                   len(compressed_data["obs"][0]), len(compressed_data["new_obs"][0]),
                                   metrics.get("raw_total_bytes"))
        self.compress_node.reset()

    def _compress_mode_c(self):
        """Internal mode C (external mode B): Ray batch sync compression."""
        nodes = self._pending_nodes[:]
        
        t_batch_start = time.time()
        
        nodes_data = [self._node_to_dict(node) for node in nodes]
        
        chunks = [
            nodes_data[i:i + self._chunk_size]
            for i in range(0, len(nodes_data), self._chunk_size)
        ]
        
        futures = [
            compress_node_batch_ray.remote(chunk, self._compression_config)
            for chunk in chunks
        ]
        
        all_results = ray.get(futures)
        
        for batch_results in all_results:
            for result in batch_results:
                total_raw = result.get("raw_total_bytes")
                if total_raw is None:
                    raw_obs = result.get("raw_obs_bytes")
                    raw_new_obs = result.get("raw_new_obs_bytes")
                    if raw_obs is not None and raw_new_obs is not None:
                        total_raw = int(raw_obs) + int(raw_new_obs)
                self._add_compressed_batch(result["compressed_data"], result["weight"],
                                           result["obs_bytes"], result["new_obs_bytes"],
                                           total_raw)
        
        actual_wall_time = (time.time() - t_batch_start) * 1000.0
        self._metrics["compress_time_ms"] += actual_wall_time

    def _compress_mode_d(self):
        """Internal mode D (external mode C): asynchronous Ray compression."""
        
        while len(self._pending_nodes) >= self._chunk_size:
            chunk_nodes = self._pending_nodes[:self._chunk_size]
            self._pending_nodes = self._pending_nodes[self._chunk_size:]
            
            nodes_data = [self._node_to_dict(node) for node in chunk_nodes]
            future = compress_node_batch_ray.remote(nodes_data, self._compression_config)
            self._inflight_futures.append(future)
            
        
        if self._inflight_futures:
            ready_refs, remaining_refs = ray.wait(
                self._inflight_futures, 
                num_returns=len(self._inflight_futures),
                timeout=0  # Non-blocking
            )
            
            if ready_refs:
                t_drain_start = time.time()
                for ref in ready_refs:
                    batch_results = ray.get(ref)
                    for result in batch_results:
                        total_raw = result.get("raw_total_bytes")
                        if total_raw is None:
                            raw_obs = result.get("raw_obs_bytes")
                            raw_new_obs = result.get("raw_new_obs_bytes")
                            if raw_obs is not None and raw_new_obs is not None:
                                total_raw = int(raw_obs) + int(raw_new_obs)
                        self._add_compressed_batch(result["compressed_data"], result["weight"],
                                                   result["obs_bytes"], result["new_obs_bytes"],
                                                   total_raw)
                
                drain_time = (time.time() - t_drain_start) * 1000.0
                self._metrics["compress_time_ms"] += drain_time
                
                self._inflight_futures = remaining_refs
                
        
        # Backpressure keeps in-flight tasks bounded to avoid memory growth.
        if self._max_inflight_tasks > 0 and len(self._inflight_futures) >= self._max_inflight_tasks:
            t_backpressure_start = time.time()
            
            oldest_ref = self._inflight_futures.pop(0)
            batch_results = ray.get(oldest_ref)
            
            for result in batch_results:
                total_raw = result.get("raw_total_bytes")
                if total_raw is None:
                    raw_obs = result.get("raw_obs_bytes")
                    raw_new_obs = result.get("raw_new_obs_bytes")
                    if raw_obs is not None and raw_new_obs is not None:
                        total_raw = int(raw_obs) + int(raw_new_obs)
                self._add_compressed_batch(result["compressed_data"], result["weight"],
                                           result["obs_bytes"], result["new_obs_bytes"],
                                           total_raw)
            
            backpressure_time = (time.time() - t_backpressure_start) * 1000.0
            self._metrics["compress_time_ms"] += backpressure_time
            self._metrics["backpressure_wait_ms"] += backpressure_time

    def _reset_pool(self):
        """Reset pending nodes pool."""
        self._pending_nodes = []

    def stats(self, debug: bool = False) -> Dict:
        """Compute memory usage statistics.

        Returns:
            Dict with estimated size, compression, and timing metrics.
        """
        total_size = 0
        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], "__getitem__"):
                obs_data = sample_batch["obs"]
                # obs is object array [bytes] - compressed block in obs[0]
                if isinstance(obs_data[0], bytes):
                    total_size += len(obs_data[0])
                else:
                    total_size += len(obs_data) if hasattr(obs_data, '__len__') else 0
                    
            if "new_obs" in sample_batch and hasattr(
                sample_batch["new_obs"], "__getitem__"
            ):
                new_obs_data = sample_batch["new_obs"]
                # new_obs is object array [bytes] - compressed block in new_obs[0]
                if isinstance(new_obs_data[0], bytes):
                    total_size += len(new_obs_data[0])
                else:
                    total_size += len(new_obs_data) if hasattr(new_obs_data, '__len__') else 0

        out = {
            "est_size_bytes": total_size,
            "est_compressed_bytes": self._est_compressed_bytes,
            "est_raw_bytes": self._est_raw_bytes,
            "compression_ratio": (self._est_compressed_bytes / self._est_raw_bytes)
            if self._est_raw_bytes > 0 else 0.0,
            "num_entries": len(self._storage) * self.sub_buffer_size,
            "compress_time_ms": self._metrics.get("compress_time_ms", 0.0),
            "compress_count": int(self._metrics.get("compress_count", 0)),
            "backpressure_wait_ms": self._metrics.get("backpressure_wait_ms", 0.0),
            "decompress_time_ms": self._metrics.get("decompress_time_ms", 0.0),
            "decompress_count": int(self._metrics.get("decompress_count", 0)),
        }
        return out

    def sample(self, num_items: int, beta: Optional[float] = None, **kwargs) -> Optional[SampleBatch]:
        """Sample blocks and expand metadata to transition-level.

        Args:
            num_items: Number of blocks to sample.
            beta: PER importance sampling exponent (defaults to self.beta if not provided).
            **kwargs: Additional sampling args.

        Returns:
            SampleBatch with expanded weights/batch_indexes, or None if empty.
        """
        if self._compression_mode == "D" and len(self._storage) == 0 and len(self._inflight_futures) > 0:
            self._drain_pending_futures()
        
        if len(self._storage) == 0:
            return None
        
        if beta is None:
            beta = getattr(self, "beta", 1.0)
        
        try:
            batch = super(RASPBERryReplayBuffer, self).sample(num_items, beta=beta, **kwargs)
        except ValueError as e:
            if "empty buffer" in str(e).lower():
                return None
            raise

        if batch is None:
            return None

        num_transitions = len(batch.get("actions", batch.get("rewards", [])))
        self._expand_block_field(batch, "weights", num_transitions)
        self._expand_block_field(batch, "batch_indexes", num_transitions)

        return batch
    
    def _expand_block_field(self, batch: SampleBatch, field_name: str, target_size: int) -> None:
        """Expand a block-level field to transition-level by replication.

        Args:
            batch: SampleBatch to modify in-place.
            field_name: Field to expand (weights, batch_indexes).
            target_size: Target size (number of transitions).
        """
        if field_name not in batch or len(batch[field_name]) == target_size:
            return
        
        replicate_factor = target_size // len(batch[field_name])
        batch[field_name] = np.repeat(batch[field_name], replicate_factor)
    
    def _drain_pending_futures(self) -> None:
        """Drain all pending Ray futures (internal mode D / external mode C)."""
        t_start = time.time()
        all_results = ray.get(self._inflight_futures)
        self._inflight_futures = []
        
        for batch_results in all_results:
            for result in batch_results:
                total_raw = result.get("raw_total_bytes")
                if total_raw is None:
                    raw_obs = result.get("raw_obs_bytes")
                    raw_new_obs = result.get("raw_new_obs_bytes")
                    if raw_obs is not None and raw_new_obs is not None:
                        total_raw = int(raw_obs) + int(raw_new_obs)
                self._add_compressed_batch(result["compressed_data"], result["weight"],
                                           result["obs_bytes"], result["new_obs_bytes"],
                                           total_raw)
        
        self._metrics["compress_time_ms"] += (time.time() - t_start) * 1000.0

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add a batch to the buffer, organizing transitions into blocks.

        Blocks are compressed as they fill up.

        Args:
            batch: SampleBatch with transitions to add.
            **kwargs: Additional args (unused).
        """
        if not isinstance(batch, SampleBatch):
            return

        idx = 0
        count = len(batch)
        while idx < count:
            space = self.compress_node.buffer_size - self.compress_node.pos
            take = min(space, count - idx)
            slice_batch = batch.slice(idx, idx + take)
            self.compress_node.add(slice_batch)
            idx += take

            if self.compress_node.is_ready():
                if self._compression_mode == "B":
                    # External mode A -> internal mode B
                    self._compress_mode_b()
                elif self._compression_mode == "C":
                    # External mode B -> internal mode C
                    self._pending_nodes.append(self.compress_node)
                    self.compress_node = self._create_compress_node()
                    if len(self._pending_nodes) >= self._chunk_size:
                        self._compress_mode_c()
                        self._reset_pool()
                elif self._compression_mode == "D":
                    # External mode C -> internal mode D
                    self._pending_nodes.append(self.compress_node)
                    self.compress_node = self._create_compress_node()
                    self._compress_mode_d()

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """Encode samples and return compressed data with metadata.

        Args:
            idxes: Block indices to retrieve.

        Returns:
            Concatenated SampleBatch with compress_base metadata.
        """
        batch_list = []
        for i in idxes:
            self._hit_count[i] += 1
            batch_list.append(self._storage[i])

        # All modes use compression - preserve metadata
        # Remove metadata fields before concat (they're scalars, can't be concatenated)
        compress_base_value = batch_list[0].get("compress_base", self.compress_base) if batch_list else self.compress_base
        is_compressed_value = batch_list[0].get("is_compressed", True) if batch_list else True
        
        for batch in batch_list:
            if "compress_base" in batch:
                del batch["compress_base"]
            if "is_compressed" in batch:
                del batch["is_compressed"]
        
        out = concat_samples(batch_list)
        
        out["compress_base"] = compress_base_value
        out["is_compressed"] = is_compressed_value

        return out

    def _estimate_raw_bytes(self, sample_batch: SampleBatch) -> int:
        """Estimate raw (uncompressed) bytes based on shapes and dtypes."""

        def _bytes(array: np.ndarray) -> int:
            return int(array.nbytes) if isinstance(array, np.ndarray) else 0

        obs_bytes = _bytes(sample_batch["obs"])
        new_obs_bytes = _bytes(sample_batch["new_obs"])
        act_bytes = _bytes(sample_batch["actions"])
        rew_bytes = _bytes(sample_batch["rewards"])
        term_bytes = _bytes(sample_batch["terminateds"])
        trunc_bytes = _bytes(sample_batch["truncateds"])
        weight_bytes = _bytes(sample_batch["weights"])
        return obs_bytes + new_obs_bytes + act_bytes + rew_bytes + term_bytes + trunc_bytes + weight_bytes

    def _add_compressed_batch(
        self,
        compressed_batch: SampleBatch,
        weight: float,
        obs_bytes: int,
        new_obs_bytes: int,
        raw_bytes: Optional[int],
    ) -> None:
        """Insert compressed batch and update size accounting."""

        prev_entry = None
        if self._next_idx < len(self._storage):
            prev_entry = self._storage[self._next_idx]
        self._add_single_batch(compressed_batch, weight=weight)

        total_comp = int(obs_bytes) + int(new_obs_bytes)
        self._metrics["compress_bytes_obs"] += int(obs_bytes)
        self._metrics["compress_bytes_new_obs"] += int(new_obs_bytes)
        self._metrics["compress_count"] += 1
        self._est_compressed_bytes += total_comp

        if raw_bytes is None:
            raw_bytes = self._estimate_raw_bytes(compressed_batch)
        self._est_raw_bytes += int(raw_bytes)

        if prev_entry is not None:
            self._est_compressed_bytes -= getattr(prev_entry, "_compressed_bytes", 0)
            self._est_raw_bytes -= getattr(prev_entry, "_raw_bytes", 0)

        compressed_batch._compressed_bytes = total_comp
        compressed_batch._raw_bytes = int(raw_bytes)


