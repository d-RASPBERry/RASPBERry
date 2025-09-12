"""
SAC Trainer example using Ray PER buffer on Box2D environment(s).
Default env: CarRacing
"""
import os
import argparse
from trainers.sac_per_trainer import SACTrainer
from utils import load_paths


def main():
    paths = load_paths()
    os.environ["RAY_TMPDIR"] = os.path.abspath(paths['tmp_dir'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
                        help="CUDA device id(s), e.g., '0' or '0,1'")
    parser.add_argument("--env_in", type=str, default="BOX2D-CarRacing",
                        help="Box2D env id (e.g., BOX2D-CarRacing)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    env_name = args.env_in

    trainer = SACTrainer(
        config="./configs/sac_per.yml",
        env_name=env_name,
        run_name=f"{env_name}_SAC_PER",
        log_path=f"{paths['log_base_path']}{env_name}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_name}/",
        mlflow="./configs/mlflow.yml",
    )
    trainer.run(initialize=True, max_iterations=1000, max_time=10800)


if __name__ == "__main__":
    main()

