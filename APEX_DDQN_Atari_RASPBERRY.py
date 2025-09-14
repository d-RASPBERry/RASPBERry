"""
Simple Ape-X DQN Trainer example using RASPBERry (block-prioritized replay with compression).
"""
import os
import argparse
from trainers.apex_raspberry_trainer import ApexDQNRaspberryTrainer
from utils import env_creator, load_paths


# Top-level path and Ray temp dir setup (unified style)
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

    trainer = ApexDQNRaspberryTrainer(
        config="./configs/apex_raspberry_atari.yml",
        env_name=env_name,
        run_name=f"{env_in}_APEX_RASPBERRY",
        log_path=f"{paths['log_base_path']}{env_in}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_in}/",
        obs_space=game.observation_space,
        action_space=game.action_space,
        mlflow="./configs/mlflow.yml",
    )
    trainer.run(initialize=True, max_iterations=1000, max_time=2400)


if __name__ == "__main__":
    main()


