# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import logging
import time
from typing import Tuple

# ------ Subsection: Third-party ------
import blosc
import numpy as np
from gymnasium import spaces
from ray.rllib.policy.sample_batch import SampleBatch

# ------ Subsection: Local ------
from utils import get_obs_shape

# ====== Section: Module State ======
logger = logging.getLogger(__name__)

# ====== Section: Classes ======
class CompressReplayNode(object):
    """
    Compressed replay node.

    Design principles:
    1) Pre-allocate all necessary memory at initialization time
    2) Clear dtype and shape definitions for all tensors
    3) Built-in compression to produce a compressed SampleBatch in one step
    4) Support different observation/action space types
    5) Robust memory monitoring and error handling
    """

    def __init__(
            self,
            buffer_size: int,
            obs_space: spaces.Space,
            action_space: spaces.Space,
            compress_base: int = -1,
            compression_level: int = 5,
            cname: str = 'zstd',
            shuffle: int = blosc.BITSHUFFLE,
            nthreads: int = 4,
            enable_compression: bool = True,
    ):
        """
        Initialize the compressed replay node.
        Args:
            buffer_size: secondary block size
            obs_space: observation space (supports Box, Discrete, etc.)
            action_space: action space (supports Discrete, Box, etc.)
            compress_base: base axis for compression
                -1: smart default (move batch axis to the end)
                 0: no transpose (preserve current layout)
                >0: move that axis to the end (if < rank)
            compression_level: compression level (1-9, default 5)
            enable_compression: whether to enable compression (False for Mode C - Control)
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got: {buffer_size}")

        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.action_space = action_space
        self.compress_base = compress_base
        self.compression_level = max(1, min(9, compression_level))
        self.cname = cname
        self.shuffle = shuffle
        self.nthreads = max(1, int(nthreads))
        self.enable_compression = enable_compression
        
        try:
            blosc.set_nthreads(self.nthreads)
        except Exception:
            # Some environments do not support setting the thread count; ignore
            pass

        self.pos = 0
        self.full = False

        self.obs_shape = get_obs_shape(obs_space)

        obs_buffer_shape = (self.buffer_size,) + tuple(self.obs_shape)
        self.obs = np.zeros(obs_buffer_shape, dtype=obs_space.dtype)
        self.new_obs = np.zeros(obs_buffer_shape, dtype=obs_space.dtype)

        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros(self.buffer_size, dtype=action_space.dtype)
        else:
            self.actions = np.zeros((self.buffer_size, *action_space.shape), dtype=action_space.dtype)

        # Standard RL fields (keep dtypes aligned with BaseBuffer)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.truncateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.weights = np.zeros(self.buffer_size, dtype=np.float32)
        # Last compression metrics snapshot (for callers who need details)
        self.last_metrics = {
            "compress_time_ms": 0.0,
            "compress_prep_ms": 0.0,
            "compress_pack_obs_ms": 0.0,
            "compress_pack_new_obs_ms": 0.0,
        }

    def add(self, batch: SampleBatch) -> None:
        """Add a SampleBatch to the node's buffers."""
        if not isinstance(batch, SampleBatch) or len(batch) == 0:
            raise ValueError

        batch_size = len(batch["obs"])
        available_space = self.buffer_size - self.pos
        actual_size = min(batch_size, available_space)

        if actual_size == 0:
            raise ValueError

        end_pos = self.pos + actual_size
        slice_range = slice(self.pos, end_pos)
        data_slice = slice(None, actual_size)

        self.obs[slice_range] = batch["obs"][data_slice]
        self.new_obs[slice_range] = batch["new_obs"][data_slice]
        self.actions[slice_range] = batch["actions"][data_slice]
        self.rewards[slice_range] = batch["rewards"][data_slice]
        self.terminateds[slice_range] = batch["terminateds"][data_slice]
        self.truncateds[slice_range] = batch["truncateds"][data_slice]

        weights = batch.get("weights")
        self.weights[slice_range] = weights[data_slice] if weights is not None else 1.0

        self.pos = end_pos
        if self.pos >= self.buffer_size:
            self.full = True

    def _prepare_for_compression(self) -> Tuple[SampleBatch, float, float]:
        """Prepare data for compression.

        Returns:
            tuple: (prepared_batch, prepare_ms, transpose_ms)
        """
        t0 = time.time()
        size = self.size()
        obs_data = self.obs[:size]
        new_obs_data = self.new_obs[:size]

        # Transpose for better compression locality if needed
        t_tr0 = time.time()
        transpose_ms = 0.0
        rank = len(obs_data.shape)
        
        if self.enable_compression and rank > 1:
            if self.compress_base == -1:
                # Smart default: move batch dimension to the end
                axes = list(range(1, rank)) + [0]
                obs_data = np.transpose(obs_data, axes)
                new_obs_data = np.transpose(new_obs_data, axes)
            elif 0 < self.compress_base < rank:
                # User-specified axis to move to the end (skip when 0)
                axes = list(range(rank))
                to_move = axes.pop(self.compress_base)
                axes.append(to_move)
                obs_data = np.transpose(obs_data, axes)
                new_obs_data = np.transpose(new_obs_data, axes)
        transpose_ms = (time.time() - t_tr0) * 1000.0

        prepared = SampleBatch({
            "obs": obs_data,
            "new_obs": new_obs_data,
            "actions": self.actions[:size],
            "rewards": self.rewards[:size],
            "terminateds": self.terminateds[:size],
            "truncateds": self.truncateds[:size],
            "weights": self.weights[:size],
            "compress_base": self.compress_base,
        })
        prepare_ms = (time.time() - t0) * 1000.0
        return prepared, prepare_ms, transpose_ms

    def sample(self) -> Tuple[SampleBatch, float, dict]:
        """
        Sample and return compressed data.

        Core steps:
        1) Prepare data (including preprocessing)
        2) Compute block-level weight
        3) Perform compression
        4) Return compressed SampleBatch and weight

        Returns:
            Tuple[compressed_sample_batch, block_weight]

        Raises:
            ValueError: if the node is empty
        """
        if not self.full and self.pos == 0:
            raise ValueError("Node is empty; cannot sample compressed data")

        prepared_batch, prep_ms, transpose_ms = self._prepare_for_compression()

        block_weight = float(np.mean(prepared_batch["weights"]))
        if np.isnan(block_weight) or block_weight <= 0:
            # Guard against invalid/zero priorities to keep sampling stable.
            block_weight = 0.01

        try:
            compressed_batch, pack_obs_ms, pack_new_obs_ms, contig_ms, assemble_ms, raw_obs_bytes, raw_new_obs_bytes = self._compress_sample_batch(prepared_batch)
            metrics = {
                "prepare_ms": float(prep_ms),
                "transpose_ms": float(transpose_ms),
                "contig_ms": float(contig_ms),
                "compress_obs_ms": float(pack_obs_ms),
                "compress_new_obs_ms": float(pack_new_obs_ms),
                "assemble_ms": float(assemble_ms),
                "raw_obs_bytes": float(raw_obs_bytes),
                "raw_new_obs_bytes": float(raw_new_obs_bytes),
            }
            self.last_metrics = {
                "compress_time_ms": float(prep_ms + contig_ms + pack_obs_ms + pack_new_obs_ms + assemble_ms),
                **metrics,
            }
            metrics["raw_total_bytes"] = float(raw_obs_bytes + raw_new_obs_bytes)
            return compressed_batch, block_weight, metrics
        except Exception as e:
            logger.exception("Compression failed")
            raise RuntimeError(f"Compression failed: {e}")

    def _compress_sample_batch(self, sample_batch: SampleBatch) -> Tuple[SampleBatch, float, float, float, float, int, int]:
        """Compress the SampleBatch and return a compressed SampleBatch with timings.

        Returns:
            tuple: (compressed_batch, pack_obs_ms, pack_new_obs_ms, contig_ms, assemble_ms, raw_obs_bytes, raw_new_obs_bytes)
        """
        obs_array = sample_batch["obs"]
        new_obs_array = sample_batch["new_obs"]

        # Ensure contiguous layout to avoid implicit copies inside blosc
        t_contig0 = time.time()
        obs_array = np.ascontiguousarray(obs_array)
        new_obs_array = np.ascontiguousarray(new_obs_array)
        contig_ms = (time.time() - t_contig0) * 1000.0

        raw_obs_bytes = int(obs_array.nbytes)
        raw_new_obs_bytes = int(new_obs_array.nbytes)

        # Mode C: No compression, just wrap in object array for compatibility
        if not self.enable_compression:
            t_asm0 = time.time()
            out_batch = SampleBatch({
                "obs": np.array([obs_array], dtype=object),
                "new_obs": np.array([new_obs_array], dtype=object),
                "actions": sample_batch["actions"],
                "rewards": sample_batch["rewards"],
                "terminateds": sample_batch["terminateds"],
                "truncateds": sample_batch["truncateds"],
                "weights": sample_batch["weights"],
                "compress_base": self.compress_base,
                "is_compressed": False,  # Mark as uncompressed
            })
            assemble_ms = (time.time() - t_asm0) * 1000.0
            return out_batch, 0.0, 0.0, contig_ms, assemble_ms, raw_obs_bytes, raw_new_obs_bytes

        t_obs0 = time.time()
        compressed_obs = blosc.pack_array(
            obs_array,
            cname=self.cname,
            clevel=self.compression_level,
            shuffle=self.shuffle,
        )
        t_obs1 = time.time()
        compressed_new_obs = blosc.pack_array(
            new_obs_array,
            cname=self.cname,
            clevel=self.compression_level,
            shuffle=self.shuffle,
        )
        t_obs2 = time.time()
        pack_obs_ms = (t_obs1 - t_obs0) * 1000.0
        pack_new_obs_ms = (t_obs2 - t_obs1) * 1000.0

        t_asm0 = time.time()
        out_batch = SampleBatch({
            "obs": np.array([compressed_obs], dtype=object),
            "new_obs": np.array([compressed_new_obs], dtype=object),
            "actions": sample_batch["actions"],
            "rewards": sample_batch["rewards"],
            "terminateds": sample_batch["terminateds"],
            "truncateds": sample_batch["truncateds"],
            "weights": sample_batch["weights"],
            "compress_base": self.compress_base,
            "is_compressed": True,  # Mark as compressed
        })
        assemble_ms = (time.time() - t_asm0) * 1000.0

        return out_batch, pack_obs_ms, pack_new_obs_ms, contig_ms, assemble_ms, raw_obs_bytes, raw_new_obs_bytes

    def size(self) -> int:
        """Return current number of stored samples."""
        return self.buffer_size if self.full else self.pos

    def reset(self) -> None:
        """Reset node state (keep allocated memory)."""
        self.pos = 0
        self.full = False

    def is_ready(self) -> bool:
        """Whether the node is ready to compress (buffer full)."""
        return self.full

    def __repr__(self) -> str:
        return (f"CompressReplayNode(size={self.pos}/{self.buffer_size}, "
                f"obs_shape={self.obs_shape}, compression_level={self.compression_level})")
