"""
Simple Ape-X DQN Trainer example using RASPBERry (block-prioritized replay with compression).
"""
import os
import argparse
from trainers.apex_raspberry_trainer import APEXRaspberryTrainer
from utils import env_creator, load_config


# Top-level path and Ray temp dir setup (unified style)
paths = load_config("configs/path.yml")
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
        default="Breakout",
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
    config = load_config("./configs/apex_raspberry_atari.yml")
    run_cfg = config['run_config']
    if run_cfg.get('use_mlflow', False):
        mlflow_cfg = load_config("configs/mlflow.yml")
    else:
        mlflow_cfg = None

    trainer = APEXRaspberryTrainer(
        config=config,
        env_name=env_name,
        run_name=run_cfg['run_name_template'].format(env_in=env_in),
        log_path=f"{paths['log_base_path']}{env_in}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_in}/",
        obs_space=game.observation_space,
        action_space=game.action_space,
        mlflow_cfg=mlflow_cfg,
    )
    trainer.run(
        initialize=True,
        max_iterations=run_cfg['max_iterations'],
        max_time=run_cfg['max_time_s']
    )


if __name__ == "__main__":
    main()


