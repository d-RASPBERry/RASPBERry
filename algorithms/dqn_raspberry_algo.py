from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ
from ray.rllib.execution.common import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED, SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
from ray.rllib.utils.typing import AlgorithmConfigDict, EnvCreator, ResultDict, SampleBatchType
from ray.util import log_once
from replay_buffer.d_raspberry_ray import MultiAgentRASPBERryReplayBuffer
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


@DeveloperAPI
def update_priorities_in_replay_buffer(
        replay_buffer: Any,
        train_batch: SampleBatchType,
        train_results: ResultDict,
    ) -> None:
    """Update replay buffer priorities using raw sample-level values.
    
    Supports any MultiAgentPrioritizedReplayBuffer-style implementation that
    exposes `update_priorities(prio_dict)`, including:
    - RASPBERry (`MultiAgentRASPBERryReplayBuffer`)
    - PBER (`MultiAgentPrioritizedBlockReplayBuffer`)
    - PER (RLlib `MultiAgentPrioritizedReplayBuffer`)
    """
    
    if replay_buffer is None or not hasattr(replay_buffer, "update_priorities"):
        return

    prio_dict: Dict = {}
    for policy_id, info in train_results.items():
        td_error = info.get("td_error", info.get("learner_stats", {}).get("td_error"))

        policy_batch = train_batch.policy_batches[policy_id]
        policy_batch.set_get_interceptor(None)
        batch_indices = policy_batch.get("batch_indexes")

        # Handle sequence batches: extract first index of each sequence
        if SampleBatch.SEQ_LENS in policy_batch:
            _batch_indices = []
            if policy_batch.zero_padded:
                seq_lens = len(td_error) * [policy_batch.max_seq_len]
            else:
                seq_lens = policy_batch[SampleBatch.SEQ_LENS][: len(td_error)]

            sequence_sum = 0
            for seq_len in seq_lens:
                _batch_indices.append(batch_indices[sequence_sum])
                sequence_sum += seq_len
            batch_indices = np.array(_batch_indices)

        if td_error is None:
            if log_once(f"no_td_error_in_train_results_from_policy_{policy_id}"):
                logger.warning("Policy %s: missing td_errors, skipping priority update", policy_id)
            continue

        if batch_indices is None:
            if log_once(f"no_batch_indices_in_train_result_for_policy_{policy_id}"):
                logger.warning("Policy %s: missing batch_indices, skipping priority update", policy_id)
            continue

        if len(batch_indices) != len(td_error):
            t = replay_buffer.replay_sequence_length
            assert len(batch_indices) > len(
                td_error) and len(batch_indices) % t == 0
            batch_indices = batch_indices.reshape([-1, t])[:, 0]
            assert len(batch_indices) == len(td_error)

        prio_dict[policy_id] = (np.asarray(
            batch_indices), np.asarray(td_error))

    if prio_dict:
        replay_buffer.update_priorities(prio_dict)


class DQNRaspberryAlgo(DQN):

    def _init(self, config: AlgorithmConfigDict, env_creator: EnvCreator) -> None:
        super()._init(config, env_creator)

    @override(SimpleQ)
    def training_step(self) -> ResultDict:
        train_results = {}

        store_weight, sample_and_train_weight = calculate_rr_weights(
            self.config)

        for _ in range(store_weight):
            new_sample_batch = synchronous_parallel_sample(
                worker_set=self.workers, concat=True
            )

            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            self.local_replay_buffer.add(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        cur_ts = self._counters[NUM_AGENT_STEPS_SAMPLED if self.config.count_steps_by ==
                                                           "agent_steps" else NUM_ENV_STEPS_SAMPLED]

        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config.train_batch_size,
                    count_by_agent_steps=self.config.count_steps_by == "agent_steps",
                )

                post_fn = self.config.get(
                    "before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    train_batch,
                    train_results,
                )
                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config.target_network_update_freq:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda p, pid: pid in to_update and p.update_target()
                    )
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        return train_results
