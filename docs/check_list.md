[ ] block 索引语义可能被误用，导致 priority 聚合错位：RLlib 的 PrioritizedReplayBuffer.sample() 会把 batch_indexes 设为存储项索引并对该索引进行重复（每个存储项的实际样本数次），update_priorities() 也明确要求这些索引就是“存储项索引”。如果我们在 block replay 里把 batch_indexes 当作 transition 索引再做 // sub_buffer_size 或依赖 reshape 聚合，会把多个 block 错误合并，priority 更新将系统性偏移，分布式下尤甚。


[ ] RASPBERry 的 block 聚合对“连续且对齐”的假设过强：论文要求 block 优先级由 block 内 TD‑error 平均值更新（见下），但如果训练管线在 learner 里对 batch 做过 shuffle/concat，reshape 依赖的“每 m 个样本恰好来自同一 block”会失效，导致 block priority 被交叉污染。需要验证 batch_indexes 的排列是否稳定、是否按 block 成段排列。

[ ] 论文明确 RASPBERry 是 “block 内 TD‑error 平均”，若 PBER 使用 max 聚合，会使对比解释不对齐：

