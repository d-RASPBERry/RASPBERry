import os
import ray
import torch
import argparse
from dynaconf import Dynaconf
from utils import check_path, env_creator
from ray.tune.logger import JsonLogger
from ray.tune.registry import register_env
from algorithms.ddqn_pber import DDQNWithMPBER
from replay_buffer.mpber_ram_saver import MultiAgentPrioritizedBlockReplayBuffer
from run_trainer import run_loop

torch.manual_seed(10)
parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)
parser.add_argument("-E", "--env", dest="env_name", type=str)
parser.add_argument("-SBZ", "--sbz", dest="sub_buffer_size", type=int)

# Config path
env_name = parser.parse_args().env_name
env_name += "NoFrameskip-v4"
run_name = str(parser.parse_args().run_name)
run_name = "DDQN_%s" % env_name + "_PBER_RAM_SAVER_%s" % run_name
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
sub_buffer_size = int(parser.parse_args().sub_buffer_size)

# Init Ray
ray.init(
    num_cpus=5, num_gpus=1,
    include_dashboard=False,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

# Check path available
check_path(log_path)
log_path = str(os.path.join(log_path, run_name))
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

setting = parser.parse_args().setting_path
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

# Set Env
hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
hyper_parameters["env_config"] = {
    "id": env_name,
}
env_example = env_creator(hyper_parameters["env_config"])
obs, _ = env_example.reset()
step = env_example.step(1)
print(env_example.action_space, env_example.observation_space)
print(env_example)
print("log path: %s; check_path: %s" % (log_path, checkpoint_path))
register_env("Atari", env_creator)

# Set Replay Buffer
replay_buffer_config = {
    **hyper_parameters["replay_buffer_config"],
    "type": MultiAgentPrioritizedBlockReplayBuffer,
    "obs_space": env_example.observation_space,
    "action_space": env_example.action_space,
    "sub_buffer_size": sub_buffer_size,
    "worker_side_prioritization": False,
    "replay_sequence_length": 1,
    "rollout_fragment_length": hyper_parameters["rollout_fragment_length"],
    "num_save": 100,
    "split_mini_batch": 1,
    "store": 3000
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config

# Set Trainer
trainer = DDQNWithMPBER(config=hyper_parameters, env="Atari")

run_loop(trainer=trainer,
         log=setting.log.log,
         max_run=setting.log.max_run,
         max_time=setting.log.max_time,
         checkpoint_path=checkpoint_path,
         log_path=log_path,
         run_name=run_name,
         use_normal=True)
