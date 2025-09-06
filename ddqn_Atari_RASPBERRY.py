"""
Simple DQN Trainer example using RASPBERry (block-prioritized replay with compression).
"""
import os
import argparse
from trainers.dqn_raspberry_trainer import DQNRaspberryTrainer
from utils import env_creator, load_paths

paths = load_paths()
os.environ["RAY_TMPDIR"] = os.path.abspath(paths['tmp_dir'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA device id(s), e.g., '0' or '0,1'",
    )
    parser.add_argument(
        "--env_in",
        type=str,
        default="Pong",
        help="Atari environment name (e.g., Pong, Breakout)",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    env_in = args.env_in
    env_name = f"Atari-{env_in}NoFrameskip-v4"
    env_config = {
        "id": env_name
    }
    game = env_creator(env_config)
    trainer = DQNRaspberryTrainer(
        config="./configs/ddqn_raspberry_atari.yml",
        env_name=env_name,
        run_name=f"{env_in}_RASPBERRY",
        log_path=f"{paths['log_base_path']}{env_in}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_in}/",
        obs_space=game.observation_space,
        action_space=game.action_space,
        mlflow="./configs/mlflow.yml",
    )
    # Run training
    trainer.run(
        initialize=True,
        max_iterations=10000,
        max_time=360000,
    )


if __name__ == "__main__":
    main()
