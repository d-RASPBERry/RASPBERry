"""
DQN Trainer implementation using Ray RLlib.

This module provides DQNTrainer that uses Ray's native PER replay buffer.
"""

from typing import Dict, Any
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from utils import env_creator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config_loader import load_config
from ray.rllib.utils.replay_buffers import MultiAgentPrioritizedReplayBuffer
import time
import json

from .base_trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    """
    DQN trainer using Ray's native MultiAgent PER.
    
    This trainer creates DQN algorithms with Ray's built-in prioritized
    experience replay buffer.
    """

    def __init__(self, config: str, env_name: str = "Atari", **kwargs):
        """
        Initialize DQN trainer.
        
        Args:
            config: 配置文件路径
            env_name: 环境名称
            **kwargs: 传递给BaseTrainer的其他参数
        """
        super().__init__(config, env_name, buffer_type="PER", **kwargs)

    def _setup_config(self, config: str, buffer_type: str) -> None:
        """从YAML配置文件中读取配置，支持继承和导入。"""
        # 使用config_loader加载配置
        self.yaml_config = load_config(config)
        
        # 设置日志和检查点频率
        self.log_freq = self.yaml_config.get("log_freq", 10)
        self.checkpoint_freq = self.yaml_config.get("checkpoint_freq", 100)
        
        # 验证缓冲区类型兼容性
        hyper_params = self.yaml_config.get("hyper_parameters", {})
        buffer_config = hyper_params.get("replay_buffer_config", {})
        config_buffer_type = buffer_config.get("type", "")
        
        # 验证配置中的buffer类型是否与期望的PER兼容
        if config_buffer_type and config_buffer_type != "MultiAgentPrioritizedReplayBuffer":
            self.log(f"警告: 配置中的buffer类型 '{config_buffer_type}' 与PER trainer不匹配", "TRAIN")
        
        # 确保使用PER buffer类型
        if "type" not in buffer_config:
            buffer_config["type"] = "MultiAgentPrioritizedReplayBuffer"
            self.log("设置默认PER buffer类型", "BUFFER")

    def setup_environment(self) -> None:
        """设置和注册训练环境。"""
        self.log("设置环境...", "TRAIN")
        # 使用格式化后的环境名称
        env_id = f"RLEnvironment_{self.env_name}"
        register_env(env_id, env_creator)
        self.log(f"✓ 环境 {env_id} 注册成功", "TRAIN")
        self.env_id = env_id



    def init_algorithm(self) -> Any:
        """
        创建并返回RL算法实例。
        
        Returns:
            配置好的DQN算法实例
        """
        self.log("创建DQN算法...", "TRAIN")
        
        # 设置环境
        self.setup_environment()

        # 获取超参数
        hyper_parameters = self.yaml_config.get("hyper_parameters", {})
        
        # 直接使用YAML中的回放缓冲区配置
        buffer_config = hyper_parameters.get("replay_buffer_config", {})
        self.log(f"使用缓冲区配置: {buffer_config}", "BUFFER")
        
        # 确保buffer_config中的type是对象而不是字符串
        if "type" in buffer_config and isinstance(buffer_config["type"], str):
            if buffer_config["type"] == "MultiAgentPrioritizedReplayBuffer":
                buffer_config["type"] = MultiAgentPrioritizedReplayBuffer

        # 环境配置
        env_config = {
            "id": self.env_name if "Atari" not in self.env_name else "Atari-PongNoFrameskip-v4"
        }

        # 创建DQN配置
        dqn_config = DQNConfig()
        
        # 环境设置
        dqn_config = dqn_config.environment(
            env=self.env_id,
            env_config=env_config
        )
        
        # 框架设置
        dqn_config = dqn_config.framework(
            hyper_parameters.get("framework", "torch")
        )
        
        # 训练参数设置
        dqn_config = dqn_config.training(
            lr=hyper_parameters.get("lr", 6.25e-5),
            gamma=hyper_parameters.get("gamma", 0.99),
            double_q=hyper_parameters.get("double_q", True),
            dueling=hyper_parameters.get("dueling", False),
            hiddens=hyper_parameters.get("hiddens", [512]),
            target_network_update_freq=hyper_parameters.get("target_network_update_freq", 8000),
            replay_buffer_config=buffer_config
        )
        
        # 探索设置
        dqn_config = dqn_config.exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": hyper_parameters.get("final_epsilon", 0.01),
                "epsilon_timesteps": hyper_parameters.get("epsilon_timesteps", 200000),
            }
        )
        
        # 资源配置
        dqn_config = dqn_config.resources(
            num_gpus=hyper_parameters.get("num_gpus", 1)
        )

        # 构建算法
        self.trainer = dqn_config.build()
        self.log(f"✓ DQN算法创建成功", "TRAIN")

        return self.trainer

    def _filter_result(self, iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """过滤训练结果并返回统计信息。"""
        # 提取统计信息
        stats = {
            "reward": result.get("episode_reward_mean", "N/A"),
            "timesteps": result.get("timesteps_total", "N/A"),
            "episodes": result.get("episodes_total", "N/A"),
            "time_s": result.get("time_total_s", "N/A")
        }
        
        # 保存到文件
        if self.log_path:
            try:
                from utils import convert_np_arrays
                
                result_with_meta = result.copy()
                result_with_meta["iteration"] = iteration
                result_with_meta["timestamp"] = time.time()
                
                if hasattr(self.trainer, 'local_replay_buffer'):
                    buffer_stats = self.trainer.local_replay_buffer.stats()
                    result_with_meta["buffer_stats"] = buffer_stats
                
                processed_data = convert_np_arrays(result_with_meta)
                
                result_file = self.log_path / f"result_{iteration}.json"
                with open(result_file, "w") as f:
                    json.dump(processed_data, f)
                    
            except Exception as e:
                self.log(f"保存指标失败: {e}", "WARNING")
        
        return stats