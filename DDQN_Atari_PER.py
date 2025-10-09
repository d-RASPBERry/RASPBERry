"""
Simple DQN Trainer example using Ray PER.
"""
import os
import argparse
from trainers.dqn_per_trainer import DQNTrainer
from utils import load_paths, load_config

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
    config = load_config("./configs/ddqn_per.yml")

    run_cfg = config['run_config']

    if run_cfg.get('use_mlflow', False):
        mlflow_cfg = load_config("configs/mlflow.yml")
    else:
        mlflow_cfg = None

    trainer = DQNTrainer(
        config=config,
        env_name=env_name,
        run_name=run_cfg['run_name_template'].format(env_in=env_in),
        log_path=f"{paths['log_base_path']}{env_in}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_in}/",
        mlflow_cfg=mlflow_cfg,
    )
    # Run training
    trainer.run(
        initialize=True,
        max_iterations=run_cfg['max_iterations'],
        max_time=run_cfg['max_time_s']
    )



if __name__ == "__main__":
    main()
