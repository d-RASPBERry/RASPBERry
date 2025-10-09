"""
SAC Trainer example using Ray PER buffer on Box2D environment(s).
Default env: CarRacing
"""
import os
import argparse
from trainers.sac_per_trainer import SACTrainer
from utils import load_config

# Top-level path and Ray temp dir setup (unified style)
paths = load_config("configs/path.yml")
os.environ["RAY_TMPDIR"] = os.path.abspath(paths['tmp_dir'])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
                        help="CUDA device id(s), e.g., '0' or '0,1'")
    parser.add_argument("--env_in", type=str, default="CarRacing",
                        help="Box2D env name (e.g., CarRacing)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    env_in = args.env_in
    env_name = env_in if env_in.startswith("BOX2D-") else f"BOX2D-{env_in}"

    config = load_config("./configs/sac_per.yml")
    run_cfg = config['run_config']
    if run_cfg.get('use_mlflow', False):
        mlflow_cfg = load_config("configs/mlflow.yml")
    else:
        mlflow_cfg = None

    trainer = SACTrainer(
        config=config,
        env_name=env_name,
        run_name=run_cfg['run_name_template'].format(env_in=env_in),
        log_path=f"{paths['log_base_path']}{env_in}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_in}/",
        mlflow_cfg=mlflow_cfg,
    )
    trainer.run(
        initialize=True,
        max_iterations=run_cfg['max_iterations'],
        max_time=run_cfg['max_time_s']
    )


if __name__ == "__main__":
    main()

