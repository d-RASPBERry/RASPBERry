"""Apex DQN RASPBERry algorithm implementation.

Extends RLlib's Apex DQN with distributed block-level prioritized replay buffer support.
This implementation handles block-level priority updates for distributed training.
"""

from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_TRAINED, NUM_ENV_STEPS_TRAINED
from ray.rllib.utils.typing import AlgorithmConfigDict, EnvCreator
import logging

logger = logging.getLogger(__name__)


class ApexDQNRaspberryAlgo(ApexDQN):
    """Apex DQN algorithm customized to work with the distributed RASPBERry replay buffer.
    
    This implementation overrides priority update logic to handle block-level priorities
    in a distributed setting, where replay buffers are managed by remote actors.
    """

    def _init(self, config: AlgorithmConfigDict, env_creator: EnvCreator) -> None:
        super()._init(config, env_creator)

    @override(ApexDQN)
    def update_replay_sample_priority(self) -> None:
        """Update replay buffer priorities using block-level aggregation.
        
        Processes priority updates from the learner thread and aggregates sample-level
        priorities into block-level priorities before updating distributed replay actors.
        
        Key differences from standard Apex DQN:
        - Aggregates priorities at block level (using sub_buffer_size)
        - Takes first index of each block for batch_indices
        - Averages TD errors within each block
        """
        sub_buffer_size = self.config["replay_buffer_config"]["sub_buffer_size"]
        train_batch_size = self.config.get("train_batch_size", 512)
        num_samples_trained_this_itr = 0
        
        queue_size = self.learner_thread.outqueue.qsize()
        
        # Log every 10 updates
        current_iter = self._iteration if hasattr(self, '_iteration') else 0
        should_log = (current_iter % 10 == 0) and queue_size > 0
        
        if should_log:
            logger.info(f"[APEX-RASPBERry Learner Debug] Iteration {current_iter}")
            logger.info(f"  Queue size: {queue_size}")
            logger.info(f"  Config train_batch_size: {train_batch_size}")
            logger.info(f"  Config sub_buffer_size: {sub_buffer_size}")
        
        for update_idx in range(queue_size):
            if self.learner_thread.is_alive():
                (
                    replay_actor_id,
                    priority_dict,
                    env_steps,
                    agent_steps,
                ) = self.learner_thread.outqueue.get(timeout=0.001)
                
                # Log first update details
                if should_log and update_idx == 0:
                    logger.info(f"  Update {update_idx}: env_steps={env_steps}, agent_steps={agent_steps}")
                    for policy_id in priority_dict:
                        original_indices = priority_dict[policy_id][0]
                        original_td_error = priority_dict[policy_id][1]
                        logger.info(f"    Policy {policy_id}:")
                        logger.info(f"      Original batch_indices shape: {original_indices.shape}")
                        logger.info(f"      Original td_error shape: {original_td_error.shape}")
                        logger.info(f"      Total transitions in batch: {len(original_indices)}")
                
                # Convert sample-level priorities to block-level priorities
                for policy_id in priority_dict:
                    original_indices = priority_dict[policy_id][0]
                    original_td_error = priority_dict[policy_id][1]
                    
                    batch_indices = original_indices.reshape(-1, sub_buffer_size)[:, 0]
                    td_error = original_td_error.reshape([-1, sub_buffer_size]).mean(axis=1)
                    
                    if should_log and update_idx == 0:
                        logger.info(f"      After reshape to blocks:")
                        logger.info(f"        Block indices shape: {batch_indices.shape}")
                        logger.info(f"        Block td_error shape: {td_error.shape}")
                        logger.info(f"        Num blocks: {len(batch_indices)}")
                        logger.info(f"        Expected: {len(original_indices)} transitions / {sub_buffer_size} = {len(original_indices)//sub_buffer_size} blocks")
                    
                    priority_dict[policy_id] = (batch_indices, td_error)
                
                # Update priorities in distributed replay actors
                if self.config["replay_buffer_config"].get("prioritized_replay_alpha") > 0:
                    self._replay_actor_manager.foreach_actor(
                        func=lambda actor: actor.update_priorities(priority_dict),
                        remote_actor_ids=[replay_actor_id],
                        timeout_seconds=0,  # Do not wait for results
                    )
                
                num_samples_trained_this_itr += env_steps
                self.update_target_networks(env_steps)
                self._counters[NUM_ENV_STEPS_TRAINED] += env_steps
                self._counters[NUM_AGENT_STEPS_TRAINED] += agent_steps
                self.workers.local_worker().set_global_vars(
                    {"timestep": self._counters[NUM_ENV_STEPS_TRAINED]}
                )
            else:
                raise RuntimeError("Learner thread died during training")
        
        if should_log and queue_size > 0:
            logger.info(f"  Total samples trained this iteration: {num_samples_trained_this_itr}")
            logger.info(f"  Cumulative env_steps_trained: {self._counters[NUM_ENV_STEPS_TRAINED]}")
            logger.info(f"  Cumulative agent_steps_trained: {self._counters[NUM_AGENT_STEPS_TRAINED]}")

        self._timers["learner_dequeue"] = self.learner_thread.queue_timer
        self._timers["learner_grad"] = self.learner_thread.grad_timer
        self._timers["learner_overall"] = self.learner_thread.overall_timer

    @staticmethod
    def execution_plan(workers, config, **kwargs):
        """Execution plan for Apex DQN (deprecated in newer RLlib versions)."""
        pass

