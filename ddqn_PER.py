"""
Simple DQN Trainer example using Ray PER.
"""

from dynaconf import Dynaconf
from trainers.dqn_per_trainer import DQNTrainer


def main():
    """Simple DQN training example."""
    # Load config

    # Create trainer
    trainer = DQNTrainer(
        config="configs/ddqn_pong_per.yml",
        run_name="DQN_Pong_RayPER",
    )
    
    # Run training
    trainer.run(
        max_iterations=1000,
        max_time=3600,
        initialize=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import ray
        if ray.is_initialized():
            ray.shutdown()