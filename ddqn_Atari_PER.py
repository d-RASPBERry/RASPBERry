"""
Simple DQN Trainer example using Ray PER.
"""

from trainers.dqn_per_trainer import DQNTrainer


def main():
    env_in = "Pong"
    trainer = DQNTrainer(
        config="./configs/ddqn_per.yml",
        env_name=f"Atari-{env_in}NoFrameskip-v4",
        run_name=f"{env_in}_PER",
        log_path=f"~/data/logging/New_RASPBERry/Atari/{env_in}/",
        checkpoint_path=f"~/data/checkpoints/New_RASPBERry/Atari/{env_in}/"
    )
    trainer.setup_ray(5, 1, False)
    # Run training
    trainer.run(
        initialize=True,
        max_iterations=10000,
        max_time=360000,
    )


if __name__ == "__main__":
    main()
