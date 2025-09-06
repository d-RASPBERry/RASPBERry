"""
Simple DQN Trainer example using Ray PER.
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA device id(s), e.g., '0' or '0,1'",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from trainers.dqn_per_trainer import DQNTrainer

    env_in = "Pong"
    trainer = DQNTrainer(
        config="./configs/ddqn_per.yml",
        env_name=f"Atari-{env_in}NoFrameskip-v4",
        run_name=f"{env_in}_PER",
        log_path=f"~/data/logging/New_RASPBERry/Atari/{env_in}/",
        checkpoint_path=f"~/data/checkpoints/New_RASPBERry/Atari/{env_in}/",
        mlflow="./configs/mlflow.yml",
    )
    trainer.setup_ray(5, 1, False)
    # Run training
    trainer.run(
        initialize=True,
        max_iterations=10000,
        max_time=43200,
    )


if __name__ == "__main__":
    main()
