"""实验观测与记录工具模块。

包含日志记录、mlflow 跟踪和结果持久化等功能。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from utils import convert_np_arrays

# 导出子模块的功能
from .logger import setup_logger
from .mlflow_helper import setup_mlflow, prepare_metrics

__all__ = [
    "setup_logger",
    "setup_mlflow",
    "prepare_metrics",
    "write_iteration_json",
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
        json.dump(record, fp, indent=2)

