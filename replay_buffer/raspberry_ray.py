"""RASPBERry Replay Buffer with Ray-based Compression.

Prioritized Block Replay Buffer with on-the-fly blosc compression.

Key features:
- Block-level storage: O(M/m) operations instead of O(M)
- Ray-based parallel compression: 2.3x faster than ThreadPool
- 60-95% memory reduction for Atari environments
- Compatible with RLlib's PER interface

See docs/raspberr_design.md for detailed architecture and design decisions.
"""

import logging
import numpy as np
import time
import blosc
import ray
from gymnasium.spaces import Space
from typing import Dict, Optional, Any, List
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from replay_buffer.compress_replay_node import CompressReplayNode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import concat_samples

logger = logging.getLogger(__name__)


# ============================================================================
# Ray Remote Compression Worker
# ============================================================================

@ray.remote
def compress_node_batch_ray(nodes_data: List[dict], 
                            compress_config: dict) -> List[dict]:
    """Ray remote worker to compress a batch of nodes.
    
    Args:
        nodes_data: List of node data dicts (obs, new_obs, actions, etc.)
        compress_config: Compression config (cname, clevel, shuffle, compress_base, enable_compression)
        
    Returns:
        List of dicts with compressed_data (SampleBatch), weight, and byte counts
    """
    batch_results = []
    enable_compression = compress_config.get('enable_compression', True)
    
    for node_data in nodes_data:
        # Reconstruct arrays
        obs = node_data['obs']
        new_obs = node_data['new_obs']
        
        raw_obs_bytes = int(node_data.get('raw_obs_bytes', obs.nbytes))
        raw_new_obs_bytes = int(node_data.get('raw_new_obs_bytes', new_obs.nbytes))
        
        # Mode C: No compression
        if not enable_compression:
            # Just wrap arrays in object array for compatibility (no transpose, no compression)
            compressed_data = SampleBatch({
                "obs": np.array([obs], dtype=object),
                "new_obs": np.array([new_obs], dtype=object),
                "actions": node_data['actions'],
                "rewards": node_data['rewards'],
                "terminateds": node_data['terminateds'],
                "truncateds": node_data['truncateds'],
                "weights": node_data['weights'],
                "compress_base": compress_config['compress_base'],
                "is_compressed": False,
            })
            result = {
                'compressed_data': compressed_data,
                'weight': node_data['weight'],
                'obs_bytes': raw_obs_bytes,  # No compression
                'new_obs_bytes': raw_new_obs_bytes,
                'raw_obs_bytes': raw_obs_bytes,
                'raw_new_obs_bytes': raw_new_obs_bytes,
                'raw_total_bytes': raw_obs_bytes + raw_new_obs_bytes,
            }
            batch_results.append(result)
            continue
        
        # Compression mode: Transpose for better compression (move batch dimension to end)
        rank = len(obs.shape)
        if rank > 1:
            if compress_config['compress_base'] == -1:
                axes = list(range(1, rank)) + [0]
                obs = np.transpose(obs, axes)
                new_obs = np.transpose(new_obs, axes)
        
        # Make contiguous
        obs = np.ascontiguousarray(obs)
        new_obs = np.ascontiguousarray(new_obs)
        
        # Compress with blosc
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
        
        # Reconstruct compressed batch
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


def decompress_sample_batch(ma_batch: SampleBatch, compress_base: int = -1) -> SampleBatch:
    """Decompress obs/new_obs using blosc and restore axes.
    
    Args:
        ma_batch: Compressed batch with obs/new_obs as blosc-packed arrays (single or multiple blocks)
        compress_base: Axis transposition hint (-1 for auto, 0 for none)
        
    Returns:
        Decompressed SampleBatch with expanded weights/batch_indexes
        
    Note:
        weights/batch_indexes should already be expanded to transition-level
        by the buffer's sample() method before calling this function.
        
        Handles both single-block (DDQN via sample_min_n_steps_from_buffer)
        and multi-block (APEX direct sample) cases.
    """
    t0 = time.time()

    # Prefer per-batch compress_base metadata if present
    compress_base_used = ma_batch.get("compress_base", compress_base)

    # Mode C (no compression): unwrap object arrays directly
    is_compressed = ma_batch.get("is_compressed", True)
    obs_array = ma_batch["obs"]
    new_obs_array = ma_batch["new_obs"]
    if not is_compressed:
        # obs/new_obs are object arrays containing numpy arrays per block; flatten along batch
        try:
            obs_list = [obs_array[i] for i in range(len(obs_array))]
            new_obs_list = [new_obs_array[i] for i in range(len(new_obs_array))]
            decompressed_obs = np.concatenate(obs_list, axis=0)
            decompressed_new_obs = np.concatenate(new_obs_list, axis=0)
        except Exception:
            # Fallback: single object element
            decompressed_obs = obs_array[0]
            decompressed_new_obs = new_obs_array[0]

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

    # Compressed path: Unpack compressed arrays - handle both single and multiple blocks
    # Check if we have multiple compressed blocks (APEX) or single block (DDQN)
    if len(obs_array) == 1 and isinstance(obs_array[0], bytes):
        # Single compressed block
        decompressed_obs_transposed = blosc.unpack_array(obs_array[0])
        decompressed_new_obs_transposed = blosc.unpack_array(new_obs_array[0])
    else:
        # Multiple compressed blocks - decompress and concatenate
        obs_blocks = [blosc.unpack_array(block) for block in obs_array if isinstance(block, bytes)]
        new_obs_blocks = [blosc.unpack_array(block) for block in new_obs_array if isinstance(block, bytes)]
        
        if not obs_blocks:
            # Fallback: try single block decompression
            decompressed_obs_transposed = blosc.unpack_array(obs_array[0])
            decompressed_new_obs_transposed = blosc.unpack_array(new_obs_array[0])
        else:
            # Concatenate along the batch dimension (axis 0 for transposed, last axis after transpose)
            decompressed_obs_transposed = np.concatenate(obs_blocks, axis=-1)
            decompressed_new_obs_transposed = np.concatenate(new_obs_blocks, axis=-1)

    # Rank info
    rank = len(decompressed_obs_transposed.shape)

    if compress_base_used == -1:
        # Smart default: move axis 0 (batch) from end back to front
        if rank <= 1:
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "no_transpose"
        else:
            # Multi-dim: move last axis (original axis 0) back to the front
            axes = [rank - 1] + list(range(rank - 1))
            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"default_dim0_to_front_{rank}D"
    elif compress_base_used == 0:
        # No transpose when compress_base=0
        decompressed_obs = decompressed_obs_transposed
        decompressed_new_obs = decompressed_new_obs_transposed
        transpose_type = "no_transpose_base0"
    else:
        # User-specified axis restoration
        if compress_base_used >= rank or rank <= 1:
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "no_transpose"
        else:
            # Insert the last axis back at compress_base position
            axes = list(range(rank))
            axes.pop()  # Remove last dimension
            axes.insert(compress_base_used, rank - 1)  # Insert at specified position

            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"dim_{compress_base_used}_from_end_{rank}D"

    t1 = time.time()
    logger.debug("[Decompression] Blosc unpack + %s transpose: %.4fs", 
                 transpose_type, t1 - t0)

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


# ============================================================================
# Prioritized Block Replay Buffer with Ray Compression
# ============================================================================

class PrioritizedBlockReplayBuffer(PrioritizedReplayBuffer):
    """Prioritized replay buffer with Ray-based on-the-fly compression."""

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 32,
            compress_base: int = -1,
            compress_pool_size: int = 5,
            compression_algorithm: str = "zstd",
            compression_level: int = 5,
            compression_nthreads: int = 1,
            compression_mode: str = "D",  # "A": PBER (no compression), "B": sync, "C": batch_ray, "D": async_ray
            chunk_size: int = 10,
            **kwargs,
    ):
        """Initialize the prioritized block replay buffer with Ray compression.
        
        Args:
            obs_space: Observation space
            action_space: Action space
            sub_buffer_size: Block size (transitions per block)
            compress_base: Axis to move for compression (-1 for auto)
            compress_pool_size: Number of Ray workers (default: 5)
            compression_algorithm: Blosc algorithm (zstd, lz4, etc.)
            compression_level: Compression level (1-9)
            compression_nthreads: Blosc compression threads
            compression_mode: "A" (PBER/no compression), "B" (sync), "C" (batch Ray), "D" (async Ray)
            chunk_size: Number of nodes per Ray task batch
            **kwargs: Additional args for PrioritizedReplayBuffer
        """
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        self.obs_space = obs_space
        self.action_space = action_space

        # Compression mode configuration
        self._compression_mode = compression_mode.upper()
        self._chunk_size = max(1, int(chunk_size))
        
        # Mode A: PBER baseline (no compression)
        self._enable_compression = self._compression_mode != "A"
        
        # Ray setup based on mode
        if self._compression_mode in ["A", "B"]:
            self._num_ray_workers = 0  # No Ray for PBER or sync compression
        elif self._compression_mode in ["C", "D"]:
            self._num_ray_workers = max(4, int(compress_pool_size)) if compress_pool_size > 0 else 5
            if not ray.is_initialized():
                logger.debug("Initializing Ray with %d CPUs for compression", self._num_ray_workers)
                ray.init(num_cpus=self._num_ray_workers, ignore_reinit_error=True)

        self._inflight_futures: List[ray.ObjectRef] = []
        self._pending_nodes: List[CompressReplayNode] = []

        # Cumulative metrics
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

        # Persist compression configuration
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

    def _compress_mode_A(self):
        """Mode A: PBER - No compression, direct storage."""
        # Directly extract raw data without any compression
        size = self.compress_node.size()
        raw_batch = SampleBatch({
            "obs": self.compress_node.obs[:size].copy(),
            "new_obs": self.compress_node.new_obs[:size].copy(),
            "actions": self.compress_node.actions[:size].copy(),
            "rewards": self.compress_node.rewards[:size].copy(),
            "terminateds": self.compress_node.terminateds[:size].copy(),
            "truncateds": self.compress_node.truncateds[:size].copy(),
            "weights": self.compress_node.weights[:size].copy(),
        })
        
        # Calculate block-level weight
        weight = float(np.mean(raw_batch["weights"]))
        if np.isnan(weight) or weight <= 0:
            weight = 0.01
        
        # Add directly to storage (no compression, no metadata)
        self._add_single_batch(raw_batch, weight=weight)
        self.compress_node.reset()

    def _compress_mode_B(self):
        """Mode B: Synchronous compression (no Ray)."""
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

    def _compress_mode_C(self):
        """Mode C: Batch processing with Ray (synchronous wait)."""
        nodes = self._pending_nodes[:]
        
        t_batch_start = time.time()
        
        # Convert nodes to serializable dicts
        nodes_data = [self._node_to_dict(node) for node in nodes]
        
        # Split into chunks for Ray tasks
        chunks = [
            nodes_data[i:i + self._chunk_size]
            for i in range(0, len(nodes_data), self._chunk_size)
        ]
        
        # Submit Ray tasks
        futures = [
            compress_node_batch_ray.remote(chunk, self._compression_config)
            for chunk in chunks
        ]
        
        # Wait for all results (synchronous)
        all_results = ray.get(futures)
        
        # Flatten results and add to buffer
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
        
        # Record wall time
        actual_wall_time = (time.time() - t_batch_start) * 1000.0
        self._metrics["compress_time_ms"] += actual_wall_time

    def _compress_mode_D(self):
        """Mode D: True asynchronous processing with Ray."""
        
        # 1. Submit new tasks when we have enough pending nodes
        while len(self._pending_nodes) >= self._chunk_size:
            chunk_nodes = self._pending_nodes[:self._chunk_size]
            self._pending_nodes = self._pending_nodes[self._chunk_size:]
            
            # Convert to dicts and submit
            nodes_data = [self._node_to_dict(node) for node in chunk_nodes]
            future = compress_node_batch_ray.remote(nodes_data, self._compression_config)
            self._inflight_futures.append(future)
            
            logger.debug("Mode D: Submitted chunk (%d nodes), %d futures pending", 
                        len(chunk_nodes), len(self._inflight_futures))
        
        # 2. Check for completed tasks (non-blocking)
        if self._inflight_futures:
            ready_refs, remaining_refs = ray.wait(
                self._inflight_futures, 
                num_returns=len(self._inflight_futures),
                timeout=0  # Non-blocking
            )
            
            # Process completed tasks
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
                
                # Update futures list
                self._inflight_futures = remaining_refs
                
                logger.debug("Mode D: Processed %d futures, buffer=%d, pending=%d", 
                           len(ready_refs), len(self._storage), len(self._inflight_futures))
        
        # 3. Backpressure: if too many pending futures, wait for oldest
        if len(self._inflight_futures) >= self._num_ray_workers * 2:
            logger.debug("Mode D: Backpressure triggered, waiting for oldest future")
            t_backpressure_start = time.time()
            
            # Wait for the oldest future
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

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """Compute memory usage and expose timing metrics.
        
        Returns:
            Dict with est_size_bytes, num_entries, and timing metrics (ms)
        """
        total_size = 0
        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], "__getitem__"):
                obs_data = sample_batch["obs"]
                # Check data structure:
                # Mode A: obs is numpy array (batch_size, *obs_shape) - entire block
                # Mode B/C/D: obs is object array [bytes] - compressed block in obs[0]
                if isinstance(obs_data[0], bytes):
                    # Mode B/C/D: obs[0] is bytes containing entire compressed block
                    total_size += len(obs_data[0])
                elif hasattr(obs_data, 'nbytes'):
                    # Mode A: obs is numpy array containing entire block
                    total_size += obs_data.nbytes
                else:
                    # Fallback
                    total_size += len(obs_data) if hasattr(obs_data, '__len__') else 0
                    
            if "new_obs" in sample_batch and hasattr(
                sample_batch["new_obs"], "__getitem__"
            ):
                new_obs_data = sample_batch["new_obs"]
                # Same logic for new_obs
                if isinstance(new_obs_data[0], bytes):
                    # Mode B/C/D: new_obs[0] is bytes containing entire compressed block
                    total_size += len(new_obs_data[0])
                elif hasattr(new_obs_data, 'nbytes'):
                    # Mode A: new_obs is numpy array containing entire block
                    total_size += new_obs_data.nbytes
                else:
                    # Fallback
                    total_size += len(new_obs_data) if hasattr(new_obs_data, '__len__') else 0

        out = {
            "est_size_bytes": total_size,
            "est_compressed_bytes": self._est_compressed_bytes,
            "est_raw_bytes": self._est_raw_bytes,
            "compression_ratio": (self._est_compressed_bytes / self._est_raw_bytes)
            if self._est_raw_bytes > 0 else 0.0,
            "num_entries": len(self._storage) * self.sub_buffer_size,
            "compress_time_ms": self._metrics.get("compress_time_ms", 0.0),
            "backpressure_wait_ms": self._metrics.get("backpressure_wait_ms", 0.0),
            "decompress_time_ms": self._metrics.get("decompress_time_ms", 0.0),
        }
        return out

    def sample(self, num_items: int, beta: float, **kwargs) -> Optional[SampleBatch]:
        """Sample blocks and expand metadata to transition-level for DQN training.
        
        Args:
            num_items: Number of blocks to sample
            beta: PER importance sampling exponent
            **kwargs: Additional sampling args
            
        Returns:
            SampleBatch with expanded weights/batch_indexes, or None if buffer empty
        """
        # Drain pending async tasks if buffer empty (Mode D only)
        if self._compression_mode == "D" and len(self._storage) == 0 and len(self._inflight_futures) > 0:
            self._drain_pending_futures()
        
        if len(self._storage) == 0:
            return None
        
        try:
            batch = super(PrioritizedBlockReplayBuffer, self).sample(num_items, beta=beta, **kwargs)
        except ValueError as e:
            if "empty buffer" in str(e).lower():
                return None
            raise

        if batch is None:
            return None

        # Expand block-level weights/batch_indexes to transition-level
        num_transitions = len(batch.get("actions", batch.get("rewards", [])))
        self._expand_block_field(batch, "weights", num_transitions)
        self._expand_block_field(batch, "batch_indexes", num_transitions)

        return batch
    
    def _expand_block_field(self, batch: SampleBatch, field_name: str, target_size: int) -> None:
        """Expand block-level field to transition-level by replicating values.
        
        Args:
            batch: SampleBatch to modify in-place
            field_name: Field to expand (weights, batch_indexes)
            target_size: Target size (number of transitions)
        """
        if field_name not in batch or len(batch[field_name]) == target_size:
            return
        
        replicate_factor = target_size // len(batch[field_name])
        batch[field_name] = np.repeat(batch[field_name], replicate_factor)
    
    def _drain_pending_futures(self) -> None:
        """Drain all pending Ray futures and add to storage (Mode D async only)."""
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
        """Add a batch to the buffer, compressing blocks as they fill up.
        
        Args:
            batch: SampleBatch with transitions to add
            **kwargs: Additional args (unused)
        """
        if not isinstance(batch, SampleBatch):
            return

        # Use CompressReplayNode to fill blocks
        idx = 0
        count = len(batch)
        while idx < count:
            space = self.compress_node.buffer_size - self.compress_node.pos
            take = min(space, count - idx)
            slice_batch = batch.slice(idx, idx + take)
            self.compress_node.add(slice_batch)
            idx += take

            if self.compress_node.is_ready():
                if self._compression_mode == "A":
                    # Mode A: PBER - no compression
                    self._compress_mode_A()
                elif self._compression_mode == "B":
                    # Mode B: Synchronous compression (no Ray)
                    self._compress_mode_B()
                elif self._compression_mode == "C":
                    # Mode C: Accumulate nodes then batch process with Ray
                    self._pending_nodes.append(self.compress_node)
                    self.compress_node = self._create_compress_node()
                    if len(self._pending_nodes) >= self._chunk_size:
                        self._compress_mode_C()
                        self._reset_pool()
                elif self._compression_mode == "D":
                    # Mode D: True async processing with Ray
                    self._pending_nodes.append(self.compress_node)
                    self.compress_node = self._create_compress_node()
                    self._compress_mode_D()

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """Encode samples - returns data (compressed for B/C/D, raw for A).
        
        Args:
            idxes: Block indices to retrieve
            
        Returns:
            Concatenated SampleBatch (with compress_base metadata for compressed modes)
        """
        batch_list = []
        for i in idxes:
            self._hit_count[i] += 1
            batch_list.append(self._storage[i])

        # Mode A: Raw data, no metadata needed
        if self._compression_mode == "A":
            return concat_samples(batch_list)
        
        # Modes B/C/D: Compressed data with metadata
        # Remove metadata fields before concat (they're scalars, can't be concatenated)
        compress_base_value = batch_list[0].get("compress_base", self.compress_base) if batch_list else self.compress_base
        is_compressed_value = batch_list[0].get("is_compressed", True) if batch_list else True
        
        for batch in batch_list:
            if "compress_base" in batch:
                del batch["compress_base"]
            if "is_compressed" in batch:
                del batch["is_compressed"]
        
        # Concatenate batches
        out = concat_samples(batch_list)
        
        # Add metadata back (after concatenation)
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

