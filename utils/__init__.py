from gymnasium import spaces
from gymnasium.wrappers import (
    ResizeObservation,
    TimeLimit,
    TransformObservation,
    GrayScaleObservation,
    PixelObservationWrapper,
)
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from ray.rllib.env.wrappers.atari_wrappers import (
    FrameStack as RLlibFrameStack,
    MaxAndSkipEnv,
    ScaledFloatFrame,
    wrap_deepmind,
)
from typing import Dict, Tuple, Union
import gymnasium
import numpy as np
import os
import yaml


class ClipObservationWrapper(gymnasium.Wrapper):
    """包装器：裁剪观察值到观察空间的合法范围内。
    
    用于修复某些环境（如 LunarLanderContinuous）在极端情况下
    返回超出定义范围的观察值的问题。
    """
    
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError("ClipObservationWrapper 只支持 Box 观察空间")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs, reward, terminated, truncated, info


class CarRacingActionWrapper(gymnasium.ActionWrapper):
    """Normalize CarRacing actions to [-1, 1] for SAC agents.
    
    The raw CarRacing environment has an asymmetric action space:
    - Steering: [-1, 1]
    - Gas: [0, 1]
    - Brake: [0, 1]
    
    This wrapper presents a unified [-1, 1] action space to the agent,
    which is crucial for algorithms like SAC that typically output tanh-squashed
    actions in [-1, 1].
    
    Mapping strategy:
    - Steering: direct map [-1, 1] -> [-1, 1]
    - Gas: clip(action, 0, 1) -> [0, 1] (negative values are ignored/idling)
    - Brake: clip(action, 0, 1) -> [0, 1] (negative values are ignored/idling)
    
    This strategy ensures that an agent initialized with 0-mean outputs (common in SAC)
    starts with 0 gas and 0 brake (idling), rather than 0.5 gas/brake which causes
    conflicting inputs and stalling.
    """
    def __init__(self, env):
        super().__init__(env)
        # Define the unified symmetric action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def action(self, action):
        # Receive [-1, 1] action from agent
        steering = action[0]
        # Clip negative values to 0 for gas/brake (ReLU-style)
        gas = max(0.0, float(action[1]))
        brake = max(0.0, float(action[2]))
        
        return np.array([steering, gas, brake], dtype=np.float32)


class RewardScaleWrapper(gymnasium.RewardWrapper):
    """统一的奖励缩放包装器，便于在外层脚本中检查 scale."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale


class PixelObsExtractWrapper(gymnasium.ObservationWrapper):
    """Extract 'pixels' key from PixelObservationWrapper's dict observation.
    
    PixelObservationWrapper returns {"pixels": image} even with pixels_only=True.
    This wrapper converts it to just the image array for downstream wrappers.
    """
    def __init__(self, env, key="pixels"):
        super().__init__(env)
        self._key = key
        # Update observation space to be just the pixels
        self.observation_space = env.observation_space[key]
    
    def observation(self, obs):
        return obs[self._key]


class SkipInitialFramesWrapper(gymnasium.Wrapper):
    """在 reset 后自动跳过固定数量帧，常用于 CarRacing 等环境."""

    def __init__(self, env, skip_frames: int = 0):
        super().__init__(env)
        self.skip_frames = max(0, int(skip_frames))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.skip_frames <= 0:
            return obs, info

        action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        for _ in range(self.skip_frames):
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

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


def resolve_dtype(dtype_value):
    """Resolve dtype from string or numpy dtype spec."""
    if dtype_value is None:
        return None
    if isinstance(dtype_value, str):
        return np.dtype(dtype_value).type
    return dtype_value


def wrap_sac_like_deepmind(
    env,
    *,
    img_size: int = 84,
    frame_skip: int = 4,
    frame_stack: int = 4,
    grayscale: bool = True,
    normalize: bool = True,
    dtype=None,
):
    """DeepMind-style preprocessing tailored for BOX2DI SAC runs (e.g., CarRacing).
    
    Mirrors wrap_deepmind ordering: MaxAndSkip → Resize/Gray → (optional) clip →
    FrameStack → normalize/dtype cast. Defaults keep channel-last tensors to
    match RLlib's CNN expectations.
    """
    dtype = resolve_dtype(dtype)
    if not normalize and dtype is None:
        dtype = np.uint8

    if frame_skip and frame_skip > 1:
        env = MaxAndSkipEnv(env, skip=frame_skip)

    env = ResizeObservation(env, (img_size, img_size))
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)

    if frame_stack and frame_stack > 1:
        env = RLlibFrameStack(env, frame_stack)

    if normalize:
        env = ScaledFloatFrame(env)
    elif dtype is not None:
        env = TransformObservation(env, lambda obs: np.asarray(obs).astype(dtype))

    return env


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
    if env_config["id"][0:5] == "Atari":
        env_id = env_config["id"].replace("Atari-", "")
        env = gymnasium.make(env_id)
        return wrap_deepmind(env)
    elif env_config["id"][0:7] == "BOX2DV-":
        # BOX2DV: Vector observation environments (LunarLander)
        # Note: BipedalWalker has been moved to Image-based (BOX2DI)
        env_id = env_config["id"].replace("BOX2DV-", "")
        env = gymnasium.make(env_id)
        # Apply observation clipping wrapper to handle edge cases where
        # the environment returns observations outside the defined space bounds
        # (e.g., LunarLander angle/velocity can exceed bounds in extreme situations)
        env = ClipObservationWrapper(env)
        return env
    elif env_config["id"][0:7] == "BOX2DI-":
        # BOX2DI: Image observation environments (CarRacing, BipedalWalker-Image)
        env_id = env_config["id"].replace("BOX2DI-", "")
        
        # Determine if we need to force pixel observation
        # CarRacing is native pixels; BipedalWalker/LunarLander are native vectors.
        needs_pixel_wrapper = "CarRacing" not in env_id
        
        # Create env (render_mode="rgb_array" is key for both cases)
        env = gymnasium.make(env_id, render_mode="rgb_array")

        if needs_pixel_wrapper:
            # Wrap vector env to output pixels
            env = PixelObservationWrapper(env, pixels_only=True)
            env = PixelObsExtractWrapper(env, key="pixels")

        # Allow configuration overrides (fall back to defaults used previously)
        img_size = env_config.get("img_size", 84)
        frame_skip = env_config.get("frame_skip", 4)
        frame_stack = env_config.get("frame_stack", 4)
        grayscale = env_config.get("grayscale", True)
        
        # [Alignment Fix] Enforce normalization for CarRacing
        # SAC requires observation intensity in [0, 1] to match standard 
        # weight initialization and temperature (alpha) scaling.
        if "CarRacing" in env_id:
            normalize = True
            dtype = np.float32
        else:
            normalize = env_config.get("normalize", True)
            dtype = env_config.get("dtype", None)

        env = wrap_sac_like_deepmind(
            env,
            img_size=img_size,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            grayscale=grayscale,
            normalize=normalize,
            dtype=dtype,
        )

        # [CRITICAL Fix] Apply Action Wrapper for CarRacing to fix initialization stalling
        if "CarRacing" in env_id:
            env = CarRacingActionWrapper(env)
        
        # [Stability Fix] Scale rewards for ALL Image tasks (configurable)
        if "CarRacing" in env_id:
            # CarRacing 默认缩放 0.01，仍允许 YAML 覆写
            default_reward_scale = 0.01
        else:
            # BipedalWalker/Others 默认缩放 0.05，可通过 env_config.reward_scale 调整
            default_reward_scale = 0.05
        reward_scale = env_config.get("reward_scale", default_reward_scale)
        env = RewardScaleWrapper(env, reward_scale)

        # Default skip 50 frames for CarRacing (skip initial zoom-in/black frames)
        default_skip = 50 if "CarRacing" in env_id else 0
        reset_skip = env_config.get("reset_skip_frames", default_skip)
        if reset_skip > 0:
            env = SkipInitialFramesWrapper(env, reset_skip)

        return env
    elif env_config["id"][0:8] == "MUJOCOI-":
        # MUJOCOI: MuJoCo with IMAGE observations (render_mode="rgb_array")
        # MuJoCo envs by default return vector state; we need PixelObservationWrapper
        # to replace the state observation with rendered RGB images.
        env_id = env_config["id"].replace("MUJOCOI-", "")
        env = gymnasium.make(env_id, render_mode="rgb_array")
        
        # Convert state observation to pixel observation
        # pixels_only=True replaces state with image
        # For MuJoCo envs in Gymnasium, render() does not accept width/height directly.
        # They are usually set at env creation or via camera configuration.
        # However, MuJoCo envs respect the render_mode="rgb_array" and default resolution.
        # Let's try without explicit size args first, or rely on gym's default.
        env = PixelObservationWrapper(
            env, 
            pixels_only=True,
            # render_kwargs={} # Rely on default rendering
        )
        # Extract pixels from dict observation {"pixels": image} -> image
        env = PixelObsExtractWrapper(env, key="pixels")

        # Apply image preprocessing (similar to BOX2DI)
        img_size = env_config.get("img_size", 84)
        frame_skip = env_config.get("frame_skip", 4)
        frame_stack = env_config.get("frame_stack", 4)
        grayscale = env_config.get("grayscale", True)
        normalize = env_config.get("normalize", True)
        dtype = env_config.get("dtype", None)

        env = wrap_sac_like_deepmind(
            env,
            img_size=img_size,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            grayscale=grayscale,
            normalize=normalize,
            dtype=dtype,
        )
        # [Stability Fix] Scale rewards for MuJoCo Image tasks (configurable)
        # HalfCheetah rewards can be large (~3000+), scaling keeps Q-values reasonable
        reward_scale = env_config.get("reward_scale", 0.01)
        env = RewardScaleWrapper(env, reward_scale)
        
        return env
    else:
        raise NotImplementedError(f"Environment {env_config['id']} not supported")


def infer_env_type(env_name: str) -> str:
    """Infer environment type from environment name for path organization.
    
    Simply extracts the first part before '-' as the environment type.
    
    Args:
        env_name: Environment name (e.g., "Atari-Breakout", "CarRacing-v2")
        
    Returns:
        Environment type string (first part before '-', or full name if no '-')
        
    Examples:
        >>> infer_env_type("Atari-Breakout")
        'Atari'
        >>> infer_env_type("CarRacing-v2")
        'CarRacing'
        >>> infer_env_type("MiniGrid-Empty-8x8-v0")
        'MiniGrid'
    """
    return env_name.split("-")[0] if "-" in env_name else env_name


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


class ConfigLoader:
    """简化的配置加载器 - 统一配置管理的最小化实现。
    
    设计原则:
        - 简单: 只做必要的事（加载、合并、统一访问）
        - 直接: 不做隐式转换和魔法操作
        - 明确: 必须显式指定 runtime 路径
    
    核心功能:
        1. 加载实验配置（自动处理 extends 继承）
        2. 合并 runtime 配置
        3. 提供统一的访问接口
    
    使用方法:
        loader = ConfigLoader(runtime_config_path="configs/runtime.yml")
        config = loader.load("configs/experiments/sac/raspberry/pendulum.yml")
        
        # 统一访问路径
        paths = config['runtime']['paths']
        ray_cfg = config['runtime']['ray']
        hyper = config['hyper_parameters']
        log_freq = config.get('log_freq', 10)  # 不强制规范化
    
    返回结构:
        {
            'runtime': {              # 来自 runtime.yml
                'paths': {...},
                'ray': {...},
                'mlflow': {...},
                'logging': {...}
            },
            'env_config': {...},      # 来自 experiment.yml
            'hyper_parameters': {...},
            'run_config': {...},
            'mlflow': {...},
            'logging': {...}          # 如果存在
            # ... 其他字段保持不变
        }
    """
    
    def __init__(self, runtime_config_path: str):
        """初始化 ConfigLoader.
        
        Args:
            runtime_config_path: runtime.yml 的完整路径（必须显式指定）
        
        Raises:
            FileNotFoundError: 如果 runtime.yml 不存在
        """
        if not os.path.exists(runtime_config_path):
            raise FileNotFoundError(f"Runtime config not found: {runtime_config_path}")
        
        self.runtime_config = load_config(runtime_config_path)
    
    def load(self, experiment_config_path: str) -> Dict:
        """加载并合并配置.
        
        Args:
            experiment_config_path: 实验配置文件路径
            
        Returns:
            合并后的完整配置字典
            
        处理步骤:
            1. 加载实验配置（load_config 自动处理 extends）
            2. 注入 runtime 配置到 config['runtime']
            3. 返回合并后的配置
        """
        # Step 1: 加载实验配置（load_config 自动处理 extends）
        config = load_config(experiment_config_path)
        
        # Step 2: 注入 runtime 配置
        config['runtime'] = self.runtime_config
        
        return config


