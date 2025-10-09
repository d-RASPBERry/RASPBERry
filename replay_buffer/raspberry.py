import logging
import numpy as np
import time
import blosc
import sys
from gymnasium.spaces import Space
from typing import Dict, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, Future
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from replay_buffer.compress_replay_node import CompressReplayNode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import concat_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decompress_sample_batch(
    ma_batch: SampleBatch, compress_base: int = -1
) -> SampleBatch:
    """Decompress a SampleBatch using blosc.unpack_array and restore axes.

    Supports arbitrary observation ranks, not limited to images.

    IMPORTANT: This function assumes the input is ALWAYS compressed.
    If you pass uncompressed data, it will fail with a clear error.

    Args:
        ma_batch: compressed batch data (must be compressed!)
        compress_base: axis moved to the end during compression
                       -1: smart default (move axis 0 to the end when compressing)
                       >=0: move the last axis back to the given position
    """
    t0 = time.time()

    # Prefer per-batch compress_base metadata if present
    compress_base_used = ma_batch.get("compress_base", compress_base)

    # Unpack compressed arrays (will fail if not compressed - by design!)
    decompressed_obs_transposed = blosc.unpack_array(ma_batch["obs"][0])
    decompressed_new_obs_transposed = blosc.unpack_array(ma_batch["new_obs"][0])

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
            axes.pop()  # 移除最后一个维度
            axes.insert(compress_base_used, rank - 1)  # insert at specified position

            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"dim_{compress_base_used}_from_end_{rank}D"

    t1 = time.time()
    logger.debug(
        f"[Decompression] Blosc unpack_array & {transpose_type} transpose took: {t1 - t0:.4f}s"
    )

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


class PrioritizedBlockReplayBuffer(PrioritizedReplayBuffer):
    """Prioritized replay buffer with on-the-fly compression."""

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 32,
            compress_base: int = -1,
            compress_pool_size: int = 0,
        compression_algorithm: str = "zstd",
        compression_level: int = 5,
        compression_nthreads: int = 1,
        compression_mode: str = "A",  # "A": sync, "B": batch_pool, "D": batch_async
        chunk_size: int = 10,
        **kwargs,
    ):
        # Strictly follow Ray: pass prioritized parameters
        # (prioritized_replay_alpha/prioritized_replay_beta) via **kwargs
        # directly to the parent without aliasing or defaults here.
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        self.obs_space = obs_space
        self.action_space = action_space

        # Compression mode configuration
        self._compression_mode = compression_mode.upper()

        self._chunk_size = max(1, int(chunk_size))

        # Thread pool setup based on mode
        if self._compression_mode == "A":
            self._compress_pool = None  # No pool for pure sync
        elif self._compression_mode in ["B", "D"]:
            # Both B and D use compress_pool_size
            pool_size = max(4, int(compress_pool_size)) if compress_pool_size > 0 else 8
            self._compress_pool = ThreadPoolExecutor(max_workers=pool_size)

        self._inflight: List[Future] = []
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
        }

        # Persist compression configuration for subsequent nodes
        self._compression_algorithm = compression_algorithm
        self._compression_level = compression_level
        self._compression_nthreads = compression_nthreads

        self.compress_node = self._create_compress_node()


    def _create_compress_node(self):
        """Create a new compress node with consistent parameters."""
        return CompressReplayNode(
            buffer_size=self.sub_buffer_size,
            obs_space=self.obs_space,
            action_space=self.action_space,
            compress_base=self.compress_base,
            cname=self._compression_algorithm,
            compression_level=self._compression_level,
            nthreads=self._compression_nthreads,
        )

    def _compress_mode_A(self):
        """Mode A: Pure synchronous compression."""
        t0 = time.time()
        compressed_data, weight, metrics = self.compress_node.sample()
        self._metrics["compress_time_ms"] += (time.time() - t0) * 1000.0
        self._metrics["compress_bytes_obs"] += len(compressed_data["obs"][0])
        self._metrics["compress_bytes_new_obs"] += len(compressed_data["new_obs"][0])
        self._metrics["compress_count"] += 1
        self._add_single_batch(compressed_data, weight=weight)
        self.compress_node.reset()


    def _compress_mode_B(self):
        """Mode B: Batch processing with thread pool (synchronous)."""
        nodes = self._pending_nodes[:]
        
        # Measure actual wall time for the entire batch processing
        t_batch_start = time.time()

        def _compress_batch_worker(nodes_batch):
            """Worker function that processes a batch of nodes."""
            batch_results = []

            for node in nodes_batch:
                compressed_data, weight, metrics = node.sample()

                batch_results.append(
                    {
                        "compressed_data": compressed_data,
                        "weight": weight,
                        "obs_bytes": len(compressed_data["obs"][0]),
                        "new_obs_bytes": len(compressed_data["new_obs"][0]),
                    }
                )

            return batch_results

        # Split nodes into batches
        batches = [
            nodes[i : i + self._chunk_size]
            for i in range(0, len(nodes), self._chunk_size)
        ]

        # Process batches in parallel using thread pool
        futures = []
        for batch in batches:
            future = self._compress_pool.submit(_compress_batch_worker, batch)
            futures.append(future)

        # Collect results synchronously
        for future in futures:
            batch_results = future.result()

            for result in batch_results:
                self._add_single_batch(
                    result["compressed_data"], weight=result["weight"]
                )
                self._metrics["compress_bytes_obs"] += result["obs_bytes"]
                self._metrics["compress_bytes_new_obs"] += result["new_obs_bytes"]
                self._metrics["compress_count"] += 1
        
        # Record actual wall time (not cumulative CPU time)
        actual_wall_time = (time.time() - t_batch_start) * 1000.0
        self._metrics["compress_time_ms"] += actual_wall_time

    def _compress_mode_D(self):
        """Mode D: True asynchronous processing with chunked submission."""
        
        def _compress_batch_worker(nodes_batch):
            """Worker function that processes a batch of nodes."""
            batch_results = []
            for node in nodes_batch:
                compressed_data, weight, metrics = node.sample()
                batch_results.append({
                    "compressed_data": compressed_data,
                    "weight": weight,
                    "obs_bytes": len(compressed_data["obs"][0]),
                    "new_obs_bytes": len(compressed_data["new_obs"][0]),
                })
            return batch_results
        
        # 1. 检查pending_nodes是否构成chunk，构成则提交异步任务
        while len(self._pending_nodes) >= self._chunk_size:
            chunk = self._pending_nodes[:self._chunk_size]
            self._pending_nodes = self._pending_nodes[self._chunk_size:]
            
            future = self._compress_pool.submit(_compress_batch_worker, chunk)
            self._inflight.append(future)
            logger.debug(f"Mode D: Submitted chunk of {len(chunk)} nodes, {len(self._inflight)} futures pending")
        
        # 2. 检查inflight中是否有完成的任务，有的话立即处理
        completed_indices = []
        for i, future in enumerate(self._inflight):
            if future.done():
                completed_indices.append(i)
        
        # 3. 处理完成的任务（倒序移除以保持索引有效）
        for i in reversed(completed_indices):
            t_drain_start = time.time()
            future = self._inflight[i]
            batch_results = future.result()
            drain_time = (time.time() - t_drain_start) * 1000.0
            
            # 立即添加到主buffer
            for result in batch_results:
                self._add_single_batch(
                    result["compressed_data"], weight=result["weight"]
                )
                self._metrics["compress_bytes_obs"] += result["obs_bytes"]
                self._metrics["compress_bytes_new_obs"] += result["new_obs_bytes"]
                self._metrics["compress_count"] += 1
            
            # 更新metrics
            self._metrics["compress_time_ms"] += drain_time
            self._inflight.pop(i)  # 移除已完成的future
            
            logger.debug(f"Mode D: Processed completed future, buffer size: {len(self._storage)}, pending: {len(self._inflight)}")
        
        # 4. 反压机制：如果pending futures太多，等待最老的完成
        if len(self._inflight) >= self._compress_pool._max_workers:
            logger.debug(f"Mode D: Backpressure triggered, waiting for oldest future")
            t_backpressure_start = time.time()
            future = self._inflight.pop(0)  # 取最老的
            batch_results = future.result()
            backpressure_time = (time.time() - t_backpressure_start) * 1000.0
            
            for result in batch_results:
                self._add_single_batch(
                    result["compressed_data"], weight=result["weight"]
                )
                self._metrics["compress_bytes_obs"] += result["obs_bytes"]
                self._metrics["compress_bytes_new_obs"] += result["new_obs_bytes"]
                self._metrics["compress_count"] += 1
            
            self._metrics["compress_time_ms"] += backpressure_time
            self._metrics["backpressure_wait_ms"] += backpressure_time

    def _reset_pool(self):
        """Reset pending nodes pool."""
        self._pending_nodes = []
    
    def _drain_pending_if_needed(self):
        """For Mode D: Drain pending futures if needed during warmup."""
        if self._compression_mode != "D" or len(self._inflight) == 0:
            return
            
        # During warmup, be more aggressive about draining
        max_drain = min(2, len(self._inflight))  # Drain at most 2 batches
        drained = 0
        
        while drained < max_drain and len(self._inflight) > 0:
            t_drain_start = time.time()
            future = self._inflight.pop(0)
            batch_results = future.result()
            drain_time = (time.time() - t_drain_start) * 1000.0

            for result in batch_results:
                self._add_single_batch(
                    result["compressed_data"], weight=result["weight"]
                )
                self._metrics["compress_bytes_obs"] += result["obs_bytes"]
                self._metrics["compress_bytes_new_obs"] += result["new_obs_bytes"]
                self._metrics["compress_count"] += 1
            
            self._metrics["compress_time_ms"] += drain_time
            drained += 1
            
        logger.debug(f"Mode D warmup drain: {drained} batches, buffer size: {len(self._storage)}, pending: {len(self._inflight)}")

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """Compute memory usage and expose timing metrics for monitoring."""
        total_size = 0
        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], "__getitem__"):
                total_size += len(sample_batch["obs"][0])
            if "new_obs" in sample_batch and hasattr(
                sample_batch["new_obs"], "__getitem__"
            ):
                total_size += len(sample_batch["new_obs"][0])

        out = {
            "est_size_bytes": total_size,
            "num_entries": len(self._storage) * self.sub_buffer_size,
            "compress_time_ms": self._metrics.get("compress_time_ms", 0.0),
            "backpressure_wait_ms": self._metrics.get("backpressure_wait_ms", 0.0),
            "decompress_time_ms": self._metrics.get("decompress_time_ms", 0.0),
        }
        # Include optional fine-grained timings when present
        for k in (
            "prepare_ms",
            "transpose_ms",
            "contig_ms",
            "compress_obs_ms",
            "compress_new_obs_ms",
            "assemble_ms",
        ):
            if k in self._metrics:
                out[k] = self._metrics[k]
        return out

    def sample(self, num_items: int, beta: float, **kwargs) -> Optional[SampleBatch]:
        """Override sampling to properly expand weights (block -> transition)."""
        # For Mode D: drain pending futures if buffer is empty
        if self._compression_mode == "D" and len(self._storage) == 0 and len(self._inflight) > 0:
            logger.debug(f"Mode D: Buffer empty, draining {len(self._inflight)} pending futures")
            while len(self._inflight) > 0 and len(self._storage) == 0:
                t_drain_start = time.time()
                future = self._inflight.pop(0)
                batch_results = future.result()
                drain_time = (time.time() - t_drain_start) * 1000.0

                for result in batch_results:
                    self._add_single_batch(
                        result["compressed_data"], weight=result["weight"]
                    )
                    self._metrics["compress_bytes_obs"] += result["obs_bytes"]
                    self._metrics["compress_bytes_new_obs"] += result["new_obs_bytes"]
                    self._metrics["compress_count"] += 1
                
                self._metrics["compress_time_ms"] += drain_time
                logger.debug(f"Mode D emergency drain: buffer size now {len(self._storage)}")
                
                # Break if we have enough data
                if len(self._storage) >= num_items:
                    break
        
        # Check if buffer is still empty
        if len(self._storage) == 0:
            logger.debug(f"Buffer is empty, cannot sample {num_items} items")
            return None
            
        try:
            batch = super(PrioritizedBlockReplayBuffer, self).sample(
                num_items, beta=beta, **kwargs
            )
        except ValueError as e:
            if "empty buffer" in str(e).lower():
                logger.debug(f"Buffer not ready for sampling: {e}")
                return None
            else:
                raise e

        if batch is not None:
            # Expand block-level weights to per-transition
            if "weights" in batch and 0 < batch.count != len(batch["weights"]):
                num_blocks = len(batch["weights"])
                samples_per_block = batch.count // num_blocks
                expanded_weights = np.repeat(batch["weights"], samples_per_block)
                batch["weights"] = expanded_weights
            # Expand block-level batch_indexes to per-transition for RLlib priority updates
            if "batch_indexes" in batch and 0 < batch.count != len(
                batch["batch_indexes"]
            ):
                num_blocks_idx = len(batch["batch_indexes"])
                samples_per_block_idx = batch.count // num_blocks_idx
                expanded_idx = np.repeat(batch["batch_indexes"], samples_per_block_idx)
                batch["batch_indexes"] = expanded_idx

        return batch

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add a batch to the buffer."""
        if not isinstance(batch, SampleBatch):
            return

        # Use CompressReplayNode to compress on-the-fly:
        # fill blocks by slicing; when ready, write immediately.
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
                    self._compress_mode_A()
                elif self._compression_mode == "B":
                    # Mode B: Batch processing (accumulate then process)
                    self._pending_nodes.append(self.compress_node)
                    self.compress_node = self._create_compress_node()
                    if len(self._pending_nodes) >= self._chunk_size:
                        self._compress_mode_B()
                        self._reset_pool()
                elif self._compression_mode == "D":
                    # Mode D: True async processing (每次都调用)
                    self._pending_nodes.append(self.compress_node)
                    self.compress_node = self._create_compress_node()
                    self._compress_mode_D()  # 每次add都调用

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """Encode samples - returns COMPRESSED data (decompression happens at higher level)."""
        compressed_list = []
        for i in idxes:
            self._hit_count[i] += 1
            compressed_list.append(self._storage[i])

        # Remove compress_base metadata before concat (it's scalar, can't be concatenated)
        compress_base_value = compressed_list[0].get("compress_base", self.compress_base) if compressed_list else self.compress_base
        for batch in compressed_list:
            if "compress_base" in batch:
                del batch["compress_base"]
        
        # Concatenate compressed batches
        out = concat_samples(compressed_list)
        
        # Add compress_base metadata back
        out["compress_base"] = compress_base_value

        return out
