"""Configuration helper utilities for RASPBERry project.

Provides shared configuration loading functions used across multiple runner scripts.
"""

from pathlib import Path
from typing import Dict

# Lazy import to avoid circular dependencies
_load_config = None


def _get_load_config():
    """Lazy load the load_config function to avoid circular import."""
    global _load_config
    if _load_config is None:
        from utils import load_config
        _load_config = load_config
    return _load_config


def load_buffer_dump_config(algorithm: str, runtime_config_path: str = None) -> Dict:
    """Load buffer dump configuration from runtime.yml.
    
    This function provides a unified interface for all runner scripts to load
    buffer verification settings.
    
    Args:
        algorithm: Algorithm name ('sac', 'ddqn', 'apex') used to select the
                   appropriate dump_iteration value.
        runtime_config_path: Optional path to runtime.yml. If None, uses default
                             path relative to project root.
    
    Returns:
        dict: Configuration for buffer dumping with keys:
            - enable_dump (bool): Whether to enable buffer dumping
            - dump_condition (str): 'iteration' or 'timesteps'
            - dump_iteration (int): Iteration number to dump buffer at
            - dump_timesteps (int): Timestep count to dump buffer at
            - dump_type (str): 'full' or 'summary'
    
    Example:
        >>> dump_config = load_buffer_dump_config('sac')
        >>> if dump_config['enable_dump'] and iteration == dump_config['dump_iteration']:
        ...     # Perform buffer dump
    """
    load_config = _get_load_config()
    
    if runtime_config_path is None:
        # Default: configs/runtime.yml relative to project root
        project_root = Path(__file__).resolve().parents[1]
        runtime_config_path = str((project_root / "configs/runtime.yml").resolve())
    
    runtime_config = load_config(runtime_config_path)
    buffer_config = runtime_config.get('buffer_verification', {})
    
    # Algorithm-specific defaults for dump_iteration
    algo_defaults = {
        'sac': {'dump_iteration': 200, 'dump_timesteps': 100000, 'dump_type': 'full'},
        'ddqn': {'dump_iteration': 100, 'dump_timesteps': 100000, 'dump_type': 'full'},
        'apex': {'dump_iteration': 50, 'dump_timesteps': 100000, 'dump_type': 'summary'},
    }
    
    algo = algorithm.lower()
    defaults = algo_defaults.get(algo, algo_defaults['sac'])
    
    return {
        'enable_dump': buffer_config.get('enable_dump', False),
        'dump_condition': buffer_config.get('dump_condition', 'iteration'),
        'dump_iteration': buffer_config.get('dump_iteration', {}).get(algo, defaults['dump_iteration']),
        'dump_timesteps': buffer_config.get('dump_timesteps', {}).get(algo, defaults['dump_timesteps']),
        'dump_type': buffer_config.get('dump_type', defaults['dump_type']),
    }



