"""
Simple DQN Trainer example using RASPBERry (block-prioritized replay with compression).
"""

from trainers.dqn_raspberry_trainer import DQNRaspberryTrainer
from utils import env_creator


def main():
    env_in = "Pong"
    env_name = f"Atari-{env_in}NoFrameskip-v4"
    env_config = {
        "id": env_name
    }
    game = env_creator(env_config)
    trainer = DQNRaspberryTrainer(
        config="./configs/ddqn_raspberry_atari.yml",
        env_name=env_name,
        run_name=f"{env_in}_RASPBERRY",
        log_path=f"~/data/logging/New_RASPBERry/Atari/{env_in}/",
        checkpoint_path=f"~/data/checkpoints/New_RASPBERry/Atari/{env_in}/",
        obs_space=game.observation_space,
        action_space=game.action_space,
        mlflow="./configs/mlflow.yml",
    )
    # Run training
    trainer.run(
        initialize=True,
        max_iterations=1000,
        max_time=43200,
    )


if __name__ == "__main__":
    main()
