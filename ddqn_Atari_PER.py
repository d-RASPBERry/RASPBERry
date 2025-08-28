"""
Simple DQN Trainer example using Ray PER.
"""

from trainers.dqn_per_trainer import DQNTrainer


def main():
    trainer = DQNTrainer(
        config="./configs/ddqn_per.yml",
        env_name="Atari-PongNoFrameskip-v4",
        run_name="Pong_PER",
        log_path=f"./log/Atari/Pong/",
        checkpoint_path="./checkpoints/Atari/Pong/",
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
