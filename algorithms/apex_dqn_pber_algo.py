"""Apex DQN PBER algorithm implementation.

Uses RLlib's ApexDQN with the PBER replay buffer (no compression).
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import logging
from typing import Any, Dict

from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict, EnvCreator

# ====== Section: Module State ======
logger = logging.getLogger(__name__)


# ====== Section: Classes ======
class ApexDQNPberAlgo(ApexDQN):
    """Apex DQN algorithm customized for PBER replay."""

    def _init(self, config: AlgorithmConfigDict, env_creator: EnvCreator) -> None:
        super()._init(config, env_creator)

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
