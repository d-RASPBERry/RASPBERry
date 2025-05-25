from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, ResizeObservation
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from typing import Dict, Tuple, Union
import gymnasium
import numpy as np
import os

agent_dir = {
    0: '>',
    1: 'V',
    2: '<',
    3: '^',
}


def split_list_into_n_parts(lst, n=10):
    return [lst[i::n] for i in range(n)]


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def convert_np_arrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays(item) for item in obj]
    else:
        return obj


def flatten_dict(d):
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
def get_obs_shape(
        observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
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


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
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
    env = gymnasium.make(env_config["id"], render_mode="rgb_array")
    env = RGBImgObsWrapper(env, tile_size=env_config["tile_size"])
    env = ImgObsWrapper(env)
    env = ResizeObservation(env, (env_config["img_size"], env_config["img_size"]))
    env = TimeLimit(env, max_episode_steps=env_config["max_steps"])
    return env


def dicts_to_structured_array(dict_list):
    keys = dict_list[0].keys()
    dtype = [(key, 'float32') for key in keys]
    structured_array = np.array([tuple(d.values()) for d in dict_list], dtype=dtype)
    return structured_array


def calculate_average_with_numpy(dict_list):
    structured_array = dicts_to_structured_array(dict_list)
    averages = {dtype[0]: structured_array[dtype[0]].mean() for dtype in structured_array.dtype.descr}
    return averages


def translate_state(state):
    return state["agent_view"], state["whole_map"], state["battery"]


def copy_params(offline, online):
    layer = list(offline.collect_params().values())
    for i in layer:
        _1 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()
        online.collect_params().get("_".join(i.name.split("_")[1:])).set_data(
            i.data())
        _2 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()


def check_dir(i):
    # create required path
    if not os.path.exists("./{}/".format(i)):
        os.mkdir("./{}/".format(i))


def get_goal(array, agent):
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
    """
    Produce a pretty string of the environment.txt grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
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
    if env_config["id"][0:8] == "MiniGrid":
        return minigrid_env_creator(env_config)
    else:
        env = gymnasium.make(env_config["id"])
        return wrap_deepmind(env, noframeskip=True)
