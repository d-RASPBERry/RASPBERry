from gymnasium import spaces
from gymnasium.wrappers import PixelObservationWrapper, ResizeObservation, TimeLimit
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from typing import Dict, Tuple, Union
import gymnasium
import numpy as np
import os
import yaml

agent_dir = {
    0: '>',
    1: 'V',
    2: '<',
    3: '^',
}


def split_list_into_n_parts(lst, n=10):
    """Split a list into n interleaved parts.
    
    Args:
        lst: List to split
        n: Number of parts (default: 10)
        
    Returns:
        List of n sub-lists with interleaved elements
        
    Example:
        >>> split_list_into_n_parts([0,1,2,3,4,5], n=2)
        [[0,2,4], [1,3,5]]
    """
    return [lst[i::n] for i in range(n)]


def check_path(path):
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to check and create
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def convert_np_arrays(obj):
    """Convert numpy arrays and filter out non-JSON-serializable objects.
    
    Args:
        obj: Object to convert (dict, list, numpy array, etc.)
        
    Returns:
        JSON-serializable version of obj (lists, primitives, dicts)
        
    Note:
        Recursively converts nested structures, filters out type objects
    """
    if obj is None:
        return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, dict):
        # Filter out non-serializable values
        result = {}
        for key, value in obj.items():
            if not isinstance(value, type):  # Skip type objects (ABCMeta, etc.)
                converted = convert_np_arrays(value)
                if converted is not None or value is None:
                    result[key] = converted
        return result
    elif isinstance(obj, (list, tuple)):
        # Filter out non-serializable items
        result = []
        for item in obj:
            if not isinstance(item, type):  # Skip type objects
                converted = convert_np_arrays(item)
                if converted is not None or item is None:
                    result.append(converted)
        return result
    elif isinstance(obj, type):
        # Convert type objects to string representation
        return str(obj)
    elif hasattr(obj, '__dict__') and not callable(obj):
        # Convert objects with __dict__ to dict representation
        try:
            return convert_np_arrays(obj.__dict__)
        except (AttributeError, TypeError):
            return str(obj)
    else:
        # For basic types (int, float, str, bool), return as is
        # For other objects, convert to string
        try:
            import json
            json.dumps(obj)  # Test if it's JSON serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)


def flatten_dict(d):
    """Flatten a nested dictionary by one level.
    
    Args:
        d: Dictionary to flatten
        
    Returns:
        Flattened dictionary (nested values promoted to top level)
        
    Example:
        >>> flatten_dict({'a': 1, 'b': {'c': 2, 'd': 3}})
        {'a': 1, 'c': 2, 'd': 3}
    """
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for subkey, sub_value in value.items():
                flat_dict[subkey] = sub_value
        else:
            flat_dict[key] = value
    return flat_dict


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_obs_shape(observation_space: spaces.Space) -> Union[
    None, Dict[str, Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]], Tuple[int], Tuple[int, ...]]:
    """Get the shape of the observation (useful for buffers).

    Args:
        observation_space: Gymnasium observation space
        
    Returns:
        Tuple representing observation shape, or dict for Dict spaces
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return 1,
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return int(len(observation_space.nvec)),
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return int(observation_space.n),
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
    else:
        return None


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_action_dim(action_space: spaces.Space) -> int:
    """Get the dimension of the action space.

    Args:
        action_space: Gymnasium action space
        
    Returns:
        Integer dimension of action space
        
    Raises:
        NotImplementedError: If action space type is not supported
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def minigrid_env_creator(env_config):
    """Create a MiniGrid environment with standard wrappers.
    
    Args:
        env_config: Dict with 'id', 'tile_size', 'img_size', 'max_steps'
        
    Returns:
        Wrapped MiniGrid environment (RGB, resized, time-limited)
    """
    env = gymnasium.make(env_config["id"], render_mode="rgb_array")
    env = RGBImgObsWrapper(env, tile_size=env_config["tile_size"])
    env = ImgObsWrapper(env)
    env = ResizeObservation(env, (env_config["img_size"], env_config["img_size"]))
    env = TimeLimit(env, max_episode_steps=env_config["max_steps"])
    return env


def dicts_to_structured_array(dict_list):
    """Convert list of dicts to numpy structured array.
    
    Args:
        dict_list: List of dicts with same keys and numeric values
        
    Returns:
        Numpy structured array with fields from dict keys
    """
    keys = dict_list[0].keys()
    dtype = [(key, 'float32') for key in keys]
    structured_array = np.array([tuple(d.values()) for d in dict_list], dtype=dtype)
    return structured_array


def calculate_average_with_numpy(dict_list):
    """Calculate average values across list of dicts using numpy.
    
    Args:
        dict_list: List of dicts with same keys and numeric values
        
    Returns:
        Dict mapping keys to mean values
    """
    structured_array = dicts_to_structured_array(dict_list)
    averages = {dtype[0]: structured_array[dtype[0]].mean() for dtype in structured_array.dtype.descr}
    return averages


def translate_state(state):
    """Extract agent view, map, and battery from state dict.
    
    Args:
        state: State dict with 'agent_view', 'whole_map', 'battery' keys
        
    Returns:
        Tuple of (agent_view, whole_map, battery)
    """
    return state["agent_view"], state["whole_map"], state["battery"]


def copy_params(offline, online):
    """Copy parameters from offline network to online network (MXNet-specific).
    
    Args:
        offline: Offline network with parameters to copy
        online: Online network to update
        
    Note:
        Legacy function for MXNet-based networks
    """
    layer = list(offline.collect_params().values())
    for i in layer:
        _1 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()
        online.collect_params().get("_".join(i.name.split("_")[1:])).set_data(
            i.data())
        _2 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()


def check_dir(i):
    """Create directory if it doesn't exist (legacy function, use check_path instead).
    
    Args:
        i: Directory name to create in current directory
    """
    if not os.path.exists("./{}/".format(i)):
        os.mkdir("./{}/".format(i))


def get_goal(array, agent):
    """Find closest goal location (value 3 or 4) to agent.
    
    Args:
        array: 2D grid array with goal locations marked as 3 or 4
        agent: Agent position [x, y, ...] (only first 2 elements used)
        
    Returns:
        2D binary array with 1 at closest goal location, 0 elsewhere
    """
    _min = 999
    _location = np.zeros([array.shape[0], array.shape[1]])
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            if value in (3, 4):
                _dis = sum(np.abs(np.array(agent[:2]) - np.array((i, j))))
                if _dis < _min:
                    _location = np.zeros([array.shape[0], array.shape[1]])
                    _location[i][j] = 1
                    _min = _dis
    return _location


def to_numpy(grid, allow, agent, vis_mask=None):
    """Convert MiniGrid grid to numpy array representation.
    
    Args:
        grid: MiniGrid grid object
        allow: Dict mapping object types to integer codes
        agent: Agent position [x, y, direction] or None
        vis_mask: Optional visibility mask (default: all visible)
        
    Returns:
        2D numpy array with integer codes for each grid cell
    """
    shape = (grid.width, grid.height)
    grid = grid.grid
    if vis_mask is None:
        vis_mask = np.ones(len(grid), dtype=bool)
    else:
        vis_mask = vis_mask.flatten()
    map_img = []
    for i, j in zip(grid, vis_mask):
        if i is not None and i.type in allow.keys() and j:
            map_img.append(allow[i.type])
        else:
            map_img.append(0)
    map_img = np.array(map_img).reshape(shape)
    if agent is not None:
        map_img[agent[0], agent[1]] = allow[agent_dir[agent[2]]]
    return map_img


def env_creator(env_config):
    """Create environment with appropriate wrappers based on type.
    
    Args:
        env_config: Dict with 'id' key specifying environment
                   (MiniGrid-*, Atari-*, BOX2D-*, Pendulum-*)
        
    Returns:
        Wrapped environment instance
        
    Raises:
        NotImplementedError: If environment type not supported
    """
    if env_config["id"][0:8] == "MiniGrid":
        return minigrid_env_creator(env_config)
    elif env_config["id"][0:5] == "Atari":
        env_id = env_config["id"].replace("Atari-", "")
        env = gymnasium.make(env_id)
        return wrap_deepmind(env)
    elif env_config["id"].startswith("Pendulum-"):
        env_id = env_config["id"].replace("Pendulum-", "", 1)
        img_size = env_config.get("img_size", 84)
        pixels_only = env_config.get("pixels_only", True)
        env = gymnasium.make(env_id, render_mode="rgb_array")
        env = PixelObservationWrapper(env, pixels_only=pixels_only)
        env = ResizeObservation(env, (img_size, img_size))
        return env
    elif env_config["id"][0:5] == "BOX2D":
        return gymnasium.make(env_config["id"].replace("BOX2D-", ""))
    else:
        raise NotImplementedError(f"Environment {env_config['id']} not supported")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file with support for extends inheritance.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dict with inheritance resolved via 'extends' key
        
    Note:
        If 'extends' key is present, recursively loads base config
        and deep-merges with current config (current overrides base)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle extends inheritance
    if 'extends' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['extends'])
        base_config = load_config(base_path)
        # Remove extends key
        del config['extends']
        # Merge configs (current config overrides base)
        config = deep_merge_config(base_config, config)

    return config


def deep_merge_config(base: Dict, override: Dict) -> Dict:
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration dict
        override: Override configuration dict (takes precedence)

    Returns:
        Merged configuration dict (nested dicts are recursively merged)
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result


