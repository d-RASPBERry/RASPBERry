"""
Simple DQN Trainer example using Ray PER.
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
        log_path=f"./log/Atari/{env_in}/",
        checkpoint_path=f"./checkpoints/Atari/{env_in}/",
        obs_space=game.observation_space,
        action_space=game.action_space,
        mlflow="./configs/mlflow.yml",
    )
    # Run training
    trainer.run(
        initialize=True,
        max_iterations=1000,
        max_time=3600,
    )


if __name__ == "__main__":
    main()
