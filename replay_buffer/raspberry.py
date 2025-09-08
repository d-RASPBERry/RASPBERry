import logging
import numpy as np
import time
import blosc
import sys
from gymnasium.spaces import Space
from typing import Dict, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, Future
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.compress_replay_node import CompressReplayNode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import concat_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decompress_sample_batch(ma_batch: SampleBatch, compress_base: int = -1) -> SampleBatch:
    """Decompress a SampleBatch using blosc.unpack_array and restore axes.

    Supports arbitrary observation ranks, not limited to images.

    Args:
        ma_batch: compressed batch data
        compress_base: axis moved to the end during compression
                       -1: smart default (move axis 0 to the end when compressing)
                       >=0: move the last axis back to the given position
    """
    t0 = time.time()

    # Prefer per-batch compress_base metadata if present
    compress_base_used = ma_batch.get("compress_base", compress_base)

    # Unpack compressed arrays
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
    logger.debug(f"[Decompression] Blosc unpack_array & {transpose_type} transpose took: {t1 - t0:.4f}s")

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
            compression_algorithm: str = 'zstd',
            compression_level: int = 5,
            compression_nthreads: int = 1,
            **kwargs
    ):
        # Strictly follow Ray: pass prioritized parameters
        # (prioritized_replay_alpha/prioritized_replay_beta) via **kwargs
        # directly to the parent without aliasing or defaults here.
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        self._compress_pool_size = max(0, int(compress_pool_size))
        self._compress_pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self._compress_pool_size)
            if self._compress_pool_size > 0 else None
        )
        self._inflight: List[Future] = []

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

        self.compress_node = CompressReplayNode(
            buffer_size=sub_buffer_size,
            obs_space=obs_space,
            action_space=action_space,
            compress_base=compress_base,
            cname=self._compression_algorithm,
            compression_level=self._compression_level,
            nthreads=self._compression_nthreads,
        )

    def _drain_one(self) -> None:
        """Process one completed future if available (non-blocking)."""
        if not self._inflight:
            return
        done_indices = [i for i, f in enumerate(self._inflight) if f.done()]
        if not done_indices:
            return
        idx = done_indices[0]
        future = self._inflight.pop(idx)
        try:
            result = future.result()
            if isinstance(result, tuple) and len(result) >= 3:
                compressed_data, weight, dt = result[0], result[1], result[2]
                self._metrics["compress_time_ms"] += dt * 1000.0
                if len(result) == 4 and isinstance(result[3], dict):
                    info = result[3]
                    for k in ("prepare_ms", "transpose_ms", "contig_ms", "compress_obs_ms", "compress_new_obs_ms", "assemble_ms"):
                        if k in info:
                            self._metrics[k] = self._metrics.get(k, 0.0) + float(info[k])
            else:
                compressed_data, weight = result
            self._add_single_batch(compressed_data, weight=weight)
            try:
                self._metrics["compress_bytes_obs"] += len(compressed_data["obs"][0])
                self._metrics["compress_bytes_new_obs"] += len(compressed_data["new_obs"][0])
            except Exception:
                pass
            self._metrics["compress_count"] += 1
        except Exception:
            logger.exception("[RB] compression future failed")

    def _submit_compress(self, node: CompressReplayNode) -> None:
        assert self._compress_pool is not None
        # Backpressure: if inflight at capacity, drain one (wait for earliest)
        if len(self._inflight) >= self._compress_pool_size:
            t_bp0 = time.time()
            # Block on the oldest future
            future0 = self._inflight.pop(0)
            try:
                result0 = future0.result()
                # Support both (data, weight) and (data, weight, dt[, info])
                if isinstance(result0, tuple) and len(result0) >= 3:
                    compressed_data, weight, dt = result0[0], result0[1], result0[2]
                    self._metrics["compress_time_ms"] += dt * 1000.0
                    if len(result0) == 4 and isinstance(result0[3], dict):
                        info = result0[3]
                        for k in ("prepare_ms", "transpose_ms", "contig_ms", "compress_obs_ms", "compress_new_obs_ms", "assemble_ms"):
                            if k in info:
                                self._metrics[k] = self._metrics.get(k, 0.0) + float(info[k])
                else:
                    compressed_data, weight = result0
                self._add_single_batch(compressed_data, weight=weight)
                self._metrics["backpressure_wait_ms"] += (time.time() - t_bp0) * 1000.0
                try:
                    logger.debug(
                        f"[RB.bp] wait_ms={(time.time() - t_bp0)*1000.0:.3f} inflight_after={len(self._inflight)}"
                    )
                except Exception:
                    pass
                try:
                    self._metrics["compress_bytes_obs"] += len(compressed_data["obs"][0])
                    self._metrics["compress_bytes_new_obs"] += len(compressed_data["new_obs"][0])
                except Exception:
                    pass
                self._metrics["compress_count"] += 1
            except Exception:
                logger.exception("[RB] compression future failed (blocking)")

        def _worker(n: CompressReplayNode):
            t0 = time.time()
            out = n.sample()  # (compressed_batch, weight) or (compressed_batch, weight, info)
            dt = time.time() - t0
            if isinstance(out, tuple) and len(out) == 3 and isinstance(out[2], dict):
                return out[0], out[1], dt, out[2]
            return out[0], out[1], dt

        fut = self._compress_pool.submit(_worker, node)
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[RB.submit] inflight={len(self._inflight)} pool_size={self._compress_pool_size}")
        except Exception:
            pass
        self._inflight.append(fut)

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """Compute actual memory usage of compressed blocks and expose metrics."""
        total_size = 0
        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], '__getitem__'):
                try:
                    total_size += len(sample_batch["obs"][0])
                except Exception:
                    total_size += sys.getsizeof(sample_batch["obs"][0])
            if "new_obs" in sample_batch and hasattr(sample_batch["new_obs"], '__getitem__'):
                    try:
                        total_size += len(sample_batch["new_obs"][0])
                    except Exception:
                        total_size += sys.getsizeof(sample_batch["new_obs"][0])
        # Only expose timing-related metrics as requested
        out = {
            "compress_time_ms": self._metrics.get("compress_time_ms", 0.0),
            "backpressure_wait_ms": self._metrics.get("backpressure_wait_ms", 0.0),
            "decompress_time_ms": self._metrics.get("decompress_time_ms", 0.0),
        }
        # Include optional fine-grained timings when present
        for k in ("prepare_ms", "transpose_ms", "contig_ms", "compress_obs_ms", "compress_new_obs_ms", "assemble_ms"):
            if k in self._metrics:
                out[k] = self._metrics[k]
        return out

    def sample(self, num_items: int, beta: float, **kwargs) -> Optional[SampleBatch]:
        """Override sampling to properly expand weights (block -> transition)."""
        batch = super(PrioritizedBlockReplayBuffer, self).sample(num_items, beta=beta, **kwargs)

        if batch is not None:
            # Expand block-level weights to per-transition
            if "weights" in batch and 0 < batch.count != len(batch["weights"]):
                num_blocks = len(batch["weights"])
                samples_per_block = batch.count // num_blocks
                expanded_weights = np.repeat(batch["weights"], samples_per_block)
                batch["weights"] = expanded_weights
            # Expand block-level batch_indexes to per-transition for RLlib priority updates
            if "batch_indexes" in batch and 0 < batch.count != len(batch["batch_indexes"]):
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
                if self._compress_pool is None:
                    t0 = time.time()
                    compressed_data, weight = self.compress_node.sample()
                    self._metrics["compress_time_ms"] += (time.time() - t0) * 1000.0
                    try:
                        self._metrics["compress_bytes_obs"] += len(compressed_data["obs"][0])
                        self._metrics["compress_bytes_new_obs"] += len(compressed_data["new_obs"][0])
                    except Exception:
                        pass
                    self._metrics["compress_count"] += 1
                    try:
                        comp_obs_len = len(compressed_data["obs"][0])
                        comp_new_obs_len = len(compressed_data["new_obs"][0])
                        logger.debug(
                            f"[RB.add] block_ready: size={self.sub_buffer_size}, block_weight={weight:.4f}, "
                            f"comp_obs_bytes={comp_obs_len}, comp_new_obs_bytes={comp_new_obs_len}"
                        )
                    except Exception:
                        pass
                    self._add_single_batch(compressed_data, weight=weight)
                    self.compress_node.reset()
                else:
                    # Swap out full node and submit compression to pool
                    node_to_compress = self.compress_node
                    self.compress_node = CompressReplayNode(
                        buffer_size=self.sub_buffer_size,
                        obs_space=node_to_compress.obs_space,
                        action_space=node_to_compress.action_space,
                        compress_base=self.compress_base,
                        cname=self._compression_algorithm,
                        compression_level=self._compression_level,
                        nthreads=self._compression_nthreads,
                    )
                    self._submit_compress(node_to_compress)

            # Opportunistically drain one completed future to bound latency
            if self._compress_pool is not None:
                self._drain_one()

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """Encode samples (sync decompress + one-shot concat). Called by parent."""
        t0 = time.time()
        compressed_list = []
        for i in idxes:
            self._hit_count[i] += 1
            compressed_list.append(self._storage[i])

        t_dec0 = time.time()
        decompressed_list: List[SampleBatch] = [
            decompress_sample_batch(compressed_sample, self.compress_base)
            for compressed_sample in compressed_list
        ]
        out = concat_samples(decompressed_list)
        dt = (time.time() - t0) * 1000.0
        self._metrics["decompress_time_ms"] += (time.time() - t_dec0) * 1000.0
        self._metrics["decompress_count"] += len(compressed_list)
        try:
            logger.debug(
                f"[RB._encode_sample] blocks={len(idxes)}, transitions={out.count}, time_ms={dt:.2f}"
            )
        except Exception:
            pass
        return out
