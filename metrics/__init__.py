"""实验观测与记录工具模块。

包含日志记录、mlflow 跟踪和结果持久化等功能。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from utils import convert_np_arrays

# 导出子模块的功能
from .logger import setup_logger
from .mlflow_helper import setup_mlflow, prepare_metrics

__all__ = [
    "setup_logger",
    "setup_mlflow",
    "prepare_metrics",
    "write_iteration_json",
    "attach_buffer_stats",
]


def write_iteration_json(log_dir: Path, iteration: int, result: dict) -> None:
    """保存训练结果为 JSON 文件。
    
    Args:
        log_dir: 日志目录路径
        iteration: 当前迭代次数
        result: 训练结果字典，包含指标、状态等信息
    """
    sanitized_result = dict(result) if isinstance(result, dict) else result
    if isinstance(sanitized_result, dict):
        config = sanitized_result.get("config")
        if isinstance(config, dict):
            config = dict(config)
            replay_cfg = config.get("replay_buffer_config")
            if isinstance(replay_cfg, dict):
                replay_cfg = dict(replay_cfg)
                replay_cfg.pop("obs_space", None)
                replay_cfg.pop("action_space", None)
                config["replay_buffer_config"] = replay_cfg
            sanitized_result["config"] = config

    payload = {
        "iteration": iteration,
        "timestamp": time.time(),
        "result": sanitized_result,
    }
    record = convert_np_arrays(payload)
    output_path = log_dir / f"result_{iteration:05d}.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(record, fp, ensure_ascii=False, separators=(",", ":"))


def attach_buffer_stats(result: dict, algo: Any) -> None:
    """Attach replay buffer stats into training result dict in-place.

    Why:
    - RLlib's `algo.train()` result does not always include replay buffer memory stats.
    - Our replay buffers already expose `stats()` with `est_size_bytes`, `est_raw_bytes`, etc.
    - The runners log/save the `result` dict; so we attach stats here for JSON/MLflow visibility.

    Writes:
    - `result["buffer"]`: dict, merged/created
    - Adds derived GB metrics when corresponding *_bytes exist:
      `est_size_gb`, `est_raw_gb`, `est_compressed_gb`
    """

    if not isinstance(result, dict) or algo is None:
        return

    # Prefer existing buffer stats if already present.
    buffer_obj = result.get("buffer")
    if isinstance(buffer_obj, dict):
        buffer_stats: dict = buffer_obj
    else:
        buffer_stats = {}

    if not buffer_stats:
        # 1) Local algorithms (SAC/DQN) usually have local_replay_buffer.
        rb = getattr(algo, "local_replay_buffer", None)
        if rb is not None and hasattr(rb, "stats"):
            try:
                s = rb.stats()
            except Exception:
                s = None
            if isinstance(s, dict) and s:
                buffer_stats.update(s)

    if not buffer_stats:
        # 2) Distributed (APEX) algorithms expose shard0 replay stats.
        get_stats = getattr(algo, "_get_shard0_replay_stats", None)
        if callable(get_stats):
            try:
                s = get_stats()
            except Exception:
                s = None
            if isinstance(s, dict) and s:
                buffer_stats.update(s)

    if not buffer_stats:
        return

    # Some multi-agent buffers (e.g., RLlib's `MultiAgentPrioritizedReplayBuffer`)
    # only expose size counters under per-policy keys ("policy_*"). For logging
    # and downstream consumers, also surface aggregate totals at the top level.
    if "est_size_bytes" not in buffer_stats:
        per_policy_stats = [
            v
            for k, v in buffer_stats.items()
            if isinstance(k, str) and k.startswith("policy_") and isinstance(v, dict)
        ]

        def _sum_numeric(key: str) -> int | None:
            total = 0
            found = False
            for ps in per_policy_stats:
                v = ps.get(key)
                if isinstance(v, (int, float)):
                    total += int(v)
                    found = True
            return total if found else None

        if per_policy_stats:
            for k in (
                "est_size_bytes",
                "est_raw_bytes",
                "est_compressed_bytes",
                "num_entries",
                "added_count",
                "sampled_count",
            ):
                agg = _sum_numeric(k)
                if agg is not None:
                    buffer_stats[k] = agg

            if (
                "compression_ratio" not in buffer_stats
                and isinstance(buffer_stats.get("est_raw_bytes"), (int, float))
                and buffer_stats["est_raw_bytes"] > 0
                and isinstance(buffer_stats.get("est_compressed_bytes"), (int, float))
            ):
                buffer_stats["compression_ratio"] = float(
                    buffer_stats["est_compressed_bytes"]
                ) / float(buffer_stats["est_raw_bytes"])

    # Ensure it's attached at the expected top-level key.
    if not isinstance(result.get("buffer"), dict):
        result["buffer"] = {}
    result["buffer"].update(buffer_stats)

    # Derived metrics for human-friendly inspection.
    bytes_per_gb = 1024.0 ** 3
    for bytes_key, gb_key in (
        ("est_size_bytes", "est_size_gb"),
        ("est_raw_bytes", "est_raw_gb"),
        ("est_compressed_bytes", "est_compressed_gb"),
    ):
        v = result["buffer"].get(bytes_key)
        if isinstance(v, (int, float)):
            result["buffer"][gb_key] = float(v) / bytes_per_gb

