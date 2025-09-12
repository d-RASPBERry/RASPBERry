"""
SAC Trainer example using RASPBERry buffer on Box2D environment(s).
Default env: CarRacing
"""
import os
import argparse
from trainers.sac_raspberry_trainer import SACRaspberryTrainer
from utils import env_creator, load_paths

# Top-level path and Ray temp dir setup (unified style)
paths = load_paths()
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

    env_config = {"id": env_name}
    game = env_creator(env_config)

    trainer = SACRaspberryTrainer(
        config="./configs/sac_raspberry.yml",
        env_name=env_name,
        run_name=f"{env_in}_RASPBERRY",
        log_path=f"{paths['log_base_path']}{env_in}/",
        checkpoint_path=f"{paths['checkpoint_base_path']}{env_in}/",
        obs_space=game.observation_space,
        action_space=game.action_space,
        mlflow="./configs/mlflow.yml",
    )
    trainer.run(initialize=True, max_iterations=1000, max_time=10800)


if __name__ == "__main__":
    main()

