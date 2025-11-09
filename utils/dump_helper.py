"""Shared helpers for buffer dump management in SAC runners."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple


def slugify(text: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    if not text:
        return ""
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text
    ).strip("_")


def build_run_name(env_alias: str, config_path: Path, gpu: str, timestamp: str, default_alias: str) -> Tuple[str, str]:
    """Generate run name and slug base from alias and config.

    Args:
        env_alias: alias string from config.
        config_path: Path to config file (used to derive slug).
        gpu: GPU identifier string.
        timestamp: Timestamp string appended to run name.
        default_alias: Fallback alias if env_alias becomes empty after slugify.

    Returns:
        Tuple of (run_name, run_name_base) where run_name is full
        name including GPU/timestamp, and run_name_base is the slug
        without GPU/timestamp.
    """

    alias_slug = slugify(env_alias)
    if not alias_slug:
        alias_slug = default_alias

    config_slug = slugify(Path(config_path).stem)

    run_name_base = alias_slug
    if config_slug and config_slug.lower() not in alias_slug.lower():
        run_name_base = f"{alias_slug}-{config_slug}"

    run_name = f"{run_name_base}-{gpu}-{timestamp}"
    return run_name, run_name_base


def prepare_dump_dir(root: Path, run_name: str) -> Path:
    """Create and return dump directory under given root."""

    dump_dir = root / run_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def should_dump(
    dump_config: dict,
    iteration: int,
    result: dict,
    history: set,
) -> Tuple[bool, Optional[Tuple], Optional[str]]:
    """Decide whether to trigger a dump based on config and history.

    Args:
        dump_config: Config dictionary from runtime.yml (buffer_verification).
        iteration: Current training iteration (int).
        result: Training result dict from algo.train().
        history: Set storing keys of already-triggered dumps to avoid repeats.

    Returns:
        Tuple of (should_dump, key, label). key is added to history when
        dumping, label is used for filename suffix (e.g. buffer_t0006000).
    """

    if not dump_config or not dump_config.get("enable_dump", False):
        return False, None, None

    condition = dump_config.get("dump_condition", "iteration").lower()
    key = None
    label = None
    should = False

    if condition == "timesteps":
        target_steps = dump_config.get("dump_timesteps")
        steps_total = result.get("timesteps_total")
        if target_steps and steps_total is not None and steps_total >= target_steps:
            key = ("timesteps", target_steps)
            label = f"buffer_t{int(steps_total):07d}"
            should = True
    else:
        target_iter = dump_config.get("dump_iteration")
        if target_iter and iteration >= target_iter:
            key = ("iteration", target_iter)
            label = f"buffer_iter{iteration:05d}"
            should = True

    if should and key not in history:
        return True, key, label

    return False, None, None

