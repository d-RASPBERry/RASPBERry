"""
Simple DQN Trainer example using Ray PER.
"""

from trainers.dqn_raspberry_trainer import DQNRaspberryTrainer
from utils import env_creator

def main():
    env_name = "Atari-PongNoFrameskip-v4"
    env_config = {
        "id": env_name
    }
    game = env_creator(env_config)
    trainer = DQNRaspberryTrainer(
        config="./configs/ddqn_raspberry_atari.yml",
        env_name=env_name,
        run_name="Pong_RASPBERRY",
        log_path=f"./log/Atari/Pong/",
        checkpoint_path="./checkpoints/Atari/Pong/",
        obs_space=game.observation_space,
        action_space=game.action_space,
    )
    trainer.setup_ray(10, 1, False)
    trainer.init_algorithm()

    # Run training
    trainer.run(
        initialize=False,
        max_iterations=1000,
        max_time=3600,
    )


if __name__ == "__main__":
    main()
