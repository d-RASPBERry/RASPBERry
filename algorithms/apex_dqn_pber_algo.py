"""Apex DQN PBER algorithm implementation.

Uses RLlib's ApexDQN with the PBER replay buffer (no compression).
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import logging
from typing import Any, Dict

from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_TRAINED, NUM_ENV_STEPS_TRAINED
from ray.rllib.utils.typing import AlgorithmConfigDict, EnvCreator

# ====== Section: Module State ======
logger = logging.getLogger(__name__)


# ====== Section: Classes ======
class ApexDQNPberAlgo(ApexDQN):
    """Apex DQN algorithm customized for PBER replay."""

    def _init(self, config: AlgorithmConfigDict, env_creator: EnvCreator) -> None:
        super()._init(config, env_creator)

    @override(ApexDQN)
    def update_replay_sample_priority(self) -> None:
        """Update replay buffer priorities via replay actors."""
        replay_cfg = self.config.get("replay_buffer_config", {}) or {}
        num_samples_trained_this_itr = 0

        for _ in range(self.learner_thread.outqueue.qsize()):
            if self.learner_thread.is_alive():
                (
                    replay_actor_id,
                    priority_dict,
                    env_steps,
                    agent_steps,
                ) = self.learner_thread.outqueue.get(timeout=0.001)

                if replay_cfg.get("prioritized_replay_alpha", 0) > 0:
                    self._replay_actor_manager.foreach_actor(
                        func=lambda actor: actor.update_priorities(priority_dict),
                        remote_actor_ids=[replay_actor_id],
                        timeout_seconds=0,
                    )
                num_samples_trained_this_itr += env_steps
                self.update_target_networks(env_steps)
                self._counters[NUM_ENV_STEPS_TRAINED] += env_steps
                self._counters[NUM_AGENT_STEPS_TRAINED] += agent_steps
                self.workers.local_worker().set_global_vars(
                    {"timestep": self._counters[NUM_ENV_STEPS_TRAINED]}
                )
            else:
                raise RuntimeError("The learner thread died while training")

        self._timers["learner_dequeue"] = self.learner_thread.queue_timer
        self._timers["learner_grad"] = self.learner_thread.grad_timer
        self._timers["learner_overall"] = self.learner_thread.overall_timer

    @override(ApexDQN)
    def _get_shard0_replay_stats(self) -> Dict[str, Any]:
        """Get replay stats from the replay actor shard 0."""
        healthy_actor_ids = self._replay_actor_manager.healthy_actor_ids()
        if not healthy_actor_ids:
            return {}

        healthy_actor_id = healthy_actor_ids[0]
        results = list(
            self._replay_actor_manager.foreach_actor(
                func=lambda actor: actor.stats(),
                remote_actor_ids=[healthy_actor_id],
            )
        )
        if not results:
            return {}
        if not results[0].ok:
            raise results[0].get()
        return results[0].get()
