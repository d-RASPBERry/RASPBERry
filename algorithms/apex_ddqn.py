from ray.rllib.utils.typing import EnvCreator
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.utils.metrics import NUM_ENV_STEPS_TRAINED, NUM_AGENT_STEPS_TRAINED
from ray.rllib.utils.typing import AlgorithmConfigDict


class ApexDDQNWithDPBER(ApexDQN):

    def _init(self, config: AlgorithmConfigDict, env_creator: EnvCreator) -> None:
        super(ApexDDQNWithDPBER, self)._init(config, env_creator)

    def update_replay_sample_priority(self) -> None:
        """Update the priorities of the sample batches with new priorities that are
        computed by the learner thread.
        """
        sub_buffer_size = self.config["replay_buffer_config"]["sub_buffer_size"]
        num_samples_trained_this_itr = 0
        for _ in range(self.learner_thread.outqueue.qsize()):
            if self.learner_thread.is_alive():
                (
                    replay_actor_id,
                    priority_dict,
                    env_steps,
                    agent_steps,
                ) = self.learner_thread.outqueue.get(timeout=0.001)
                # The different with original ApexDQN object
                for i in priority_dict:
                    batch_indices = priority_dict[i][0].reshape(-1, sub_buffer_size)[:, 0]
                    td_error = priority_dict[i][1].reshape([-1, sub_buffer_size]).mean(axis=1)
                    priority_dict[i] = (batch_indices, td_error)
                if self.config["replay_buffer_config"].get("prioritized_replay_alpha") > 0:
                    self._replay_actor_manager.foreach_actor(
                        func=lambda actor: actor.update_priorities(priority_dict),
                        remote_actor_ids=[replay_actor_id],
                        timeout_seconds=0,  # Do not wait for results.
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

    @staticmethod
    def execution_plan(workers, config, **kwargs):
        pass
