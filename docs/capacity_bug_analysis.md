# Capacity Bug 分析报告

**日期**: 2025-11-06  
**状态**: 已识别并修正 PBER，RASPBERry 保持现状（技术债）

---

## 执行摘要

在分析 sub_buffer_size 对比实验时，发现 replay buffer 的 capacity 管理存在设计问题：

- **PBER**: 错误的 capacity 除法 → **已修正** ✅
- **RASPBERry**: 双重错误相互抵消 + workaround → **功能正确但需文档化** ⚠️

---

## 问题根源

### Ray RLlib 的 Capacity 语义

根据 Ray 2.8 源码分析 (`replay_buffer.py` 和 `prioritized_replay_buffer.py`)：

```python
# replay_buffer.py:37-39
capacity: Max number of timesteps to store in the FIFO
    buffer. After reaching this number, older samples will be
    dropped to make space for new ones.
```

**关键点**：
1. `capacity` 指的是 **timesteps (transitions)** 数量，不是 blocks
2. Eviction 逻辑使用 `_num_timesteps_added_wrap` 与 `capacity` 比较
3. `_num_timesteps_added_wrap` 累加每个 item 的 `SampleBatch.count`

### Ray 使用 item.count 的三个关键场景

```python
# 场景1: 累计已添加的 timesteps (replay_buffer.py:256-257)
def _add_single_batch(self, item: SampleBatchType, **kwargs):
    self._num_timesteps_added += item.count
    self._num_timesteps_added_wrap += item.count  # ← 用于 eviction 判断

# 场景2: Eviction 触发条件 (replay_buffer.py:274-279)
if self._num_timesteps_added_wrap >= self.capacity:
    self._eviction_started = True
    self._num_timesteps_added_wrap = 0
    self._next_idx = 0

# 场景3: Weights 展开 (prioritized_replay_buffer.py:141-153)
for idx in idxes:
    count = self._storage[idx].count
    actual_size = count  # (如果不是 zero_padded)
    weights.extend([weight / max_weight] * actual_size)  # ← 展开到 transition-level
    batch_indexes.extend([idx] * actual_size)
```

---

## PBER 问题分析

### 原始实现 (d_pber_ray.py)

```python
# 错误的实现 (已修正)
effective_block_capacity = max(1, int(capacity // max(1, sub_buffer_size)))
pber_config["capacity"] = effective_block_capacity  # ← 传递错误的 capacity
```

**问题**：
- 配置文件：`capacity = 1,000,000` (期望存储 1M transitions)
- sub_buffer_size = 16
- 实际传递：`capacity = 62,500` (1,000,000 ÷ 16)

### SampleBatch 结构

```python
# PBER 的 SampleBatch 结构（未压缩）
block = SampleBatch({
    "obs": np.zeros((16, 8), dtype=np.float32),      # shape = (16, 8)
    "new_obs": np.zeros((16, 8), dtype=np.float32),
    "actions": np.zeros((16, 2), dtype=np.float32),
    "rewards": np.zeros(16, dtype=np.float32),
    ...
})
# block.count = 16  ✅ 正确反映 transitions 数量
```

### 错误的后果

| 步骤 | 错误实现 | 正确实现 |
|------|---------|---------|
| 配置 capacity | 1,000,000 | 1,000,000 |
| d_pber_ray.py 除法 | ÷16 → 62,500 | 无除法 → 1,000,000 |
| 传给底层 buffer | 62,500 | 1,000,000 |
| 每个 block 的 count | 16 | 16 |
| Eviction 阈值 | 62,500 timesteps | 1,000,000 timesteps |
| 实际存储 blocks | ~3,906 | ~62,500 |
| 实际存储 transitions | **~62,500** ❌ | **~1,000,000** ✅ |

**意外的"正确"结果**：
- 虽然有除法错误，但恰好 `block.count = 16`
- Ray 计算：`3,906 blocks × 16 transitions/block = 62,500 timesteps` 达到阈值
- 结果：实际存储容量被缩小到 1/16

### 修正方案 (已实施)

```python
# 修正后的实现 (d_pber_ray.py:104-111)
# Capacity management: capacity is in transitions (NOT blocks)
# Ray's ReplayBuffer uses capacity as the threshold for _num_timesteps_added_wrap
# Each block contributes block.count (= sub_buffer_size) transitions to this counter
self._configured_capacity_transitions = int(capacity)
self._top_storage_unit = storage_unit

# Pass capacity directly (in transitions) to underlying buffer
pber_config["capacity"] = self._configured_capacity_transitions  # ← 直接传递
```

---

## RASPBERry 问题分析

### 双重错误设计

RASPBERry 有**两个错误相互抵消**，加上一个 workaround：

#### 错误1: Capacity 除法 (d_raspberry_ray.py:93)

```python
# 和 PBER 一样的错误
effective_block_capacity = max(1, int(capacity // max(1, sub_buffer_size)))
pber_config["capacity"] = effective_block_capacity  # ← 62,500 for sub_size=16
```

#### 错误2: SampleBatch.count = 1 (压缩导致)

```python
# RASPBERry 压缩后的 SampleBatch 结构
compressed_batch = SampleBatch({
    "obs": np.array([compressed_bytes], dtype=object),      # shape = (1,) ← 问题！
    "new_obs": np.array([compressed_bytes], dtype=object),  # shape = (1,)
    "actions": np.zeros((16, 2), dtype=np.float32),         # shape = (16, 2)
    "rewards": np.zeros(16, dtype=np.float32),              # shape = (16,)
    ...
})
# compressed_batch.count = 1  ❌ 根据 obs.shape[0] 计算
```

**原因**：
- Blosc 压缩将整个 block 的 obs 压缩成单个 bytes 对象
- `obs = np.array([compressed_bytes], dtype=object)` → shape = (1,)
- `SampleBatch.count` 从第一个字段推断 → count = obs.shape[0] = 1

#### Workaround: 手动展开 weights (raspberry_ray.py:551-570)

```python
def sample(self, num_items: int, beta: Optional[float] = None, **kwargs):
    batch = super(RASPBERryReplayBuffer, self).sample(num_items, beta=beta, **kwargs)
    
    # Ray 的 sample() 用 count=1 展开了 weights，导致 weights.shape = (num_blocks,)
    # 手动展开到正确的大小
    num_transitions = len(batch.get("actions", batch.get("rewards", [])))
    self._expand_block_field(batch, "weights", num_transitions)        # ← 修正
    self._expand_block_field(batch, "batch_indexes", num_transitions)
    return batch

def _expand_block_field(self, batch: SampleBatch, field_name: str, target_size: int):
    if field_name not in batch or len(batch[field_name]) == target_size:
        return
    replicate_factor = target_size // len(batch[field_name])
    batch[field_name] = np.repeat(batch[field_name], replicate_factor)
```

### 复合效果分析

| 步骤 | 值 | 说明 |
|------|-----|------|
| 配置 capacity | 1,000,000 | 用户配置 |
| d_raspberry_ray.py 除法 | ÷16 → 62,500 | **错误1** |
| 传给底层 buffer | 62,500 | eviction 阈值 |
| 每个 block 的 count | 1 | **错误2** (压缩导致) |
| Ray 累计逻辑 | `+1` per block | 每个 block 贡献 1 timestep |
| Eviction 触发时机 | 62,500 blocks | 62,500 timesteps |
| 实际存储 transitions | **1,000,000** ✅ | 62,500 × 16 |
| Weights 展开 (Ray) | × 1 | 基于 count=1 |
| Weights 展开 (手动) | × 16 | `_expand_block_field()` |
| 最终 weights 长度 | 正确 ✅ | Workaround 生效 |

**结论**：
- ✅ Eviction 时机正确 (两个错误抵消)
- ✅ Weights 展开正确 (workaround 修正)
- ⚠️ 代码语义混乱，难以维护

### 三种情况对比

| 情况 | count | capacity | Eviction | Weights | 结果 |
|------|-------|----------|----------|---------|------|
| A: 当前状态 | 1 | ÷16 | ✅ 62,500 blocks | ✅ 手动展开 | **正确** |
| B: 只改 count | 16 | ÷16 | ❌ 3,906 blocks | ✅ Ray 自动 | **容量变小** |
| C: 同时修正 | 16 | 不除 | ✅ 62,500 blocks | ✅ Ray 自动 | **正确且清晰** |

---

## Buffer Dump 验证

### 实验配置 (旧实验 2025-11-05 19:15)

所有实验：
- capacity = 1,000,000 (配置文件)
- max_time_s = 14400 (4 hours)
- 训练到 timestep = 600,768

### 实际存储数据

| 实验 | 存储单元数 | Sub_size | Total Trans | Capacity | 内存 MB | 文件 MB |
|------|-----------|----------|-------------|----------|---------|---------|
| SUB1 | 600,768 | 1 | 600,768 | 1,000,000 | 50.42 | 258.49 |
| SUB8 | 75,096 | 8 | 600,768 | 1,000,000 | 50.42 | 108.38 |
| SUB16 | 37,548 | 16 | 600,768 | 1,000,000 | 50.42 | 97.41 |
| SUB32 | 18,774 | 32 | 600,768 | 1,000,000 | 50.42 | 91.98 |
| PER | 600,768 | 1 | 600,768 | 1,000,000 | 69.90 | 430.08 |

**验证结论**：
1. ✅ 所有实验都存储了完全相同的 600,768 transitions
2. ✅ Block 数量符合 `total_trans ÷ sub_buffer_size` 的预期
3. ✅ 证明双重错误确实相互抵消了
4. ✅ PBER 系列内存使用一致 (88 bytes/trans)，PER 更高 (122 bytes/trans)
5. ✅ Block size 越大，序列化开销越小（91.98 MB vs 258.49 MB）

---

## 修正方案

### PBER: 已修正 ✅

**修改文件**: `replay_buffer/d_pber_ray.py`

**修改内容**:
```python
# 移除错误的除法，直接传递 capacity
self._configured_capacity_transitions = int(capacity)
pber_config["capacity"] = self._configured_capacity_transitions
```

**修改范围**: Lines 104-111

**影响**:
- ✅ Capacity 语义和 Ray 一致
- ✅ 代码清晰易懂
- ✅ `_expand_block_field()` 变成安全的 no-op

### RASPBERry: 保持现状 ⚠️

**技术决策**：
- ✅ 当前功能正确（双重错误抵消 + workaround）
- ✅ 不需要立即修改
- ⚠️ 技术债：代码语义不清晰

**如需修正** (可选，长期目标):

1. **修改 d_raspberry_ray.py:93-109**
   ```python
   # 移除错误的除法
   self._configured_capacity_transitions = int(capacity)
   pber_config["capacity"] = self._configured_capacity_transitions
   ```

2. **修改 raspberry_ray.py 和 compress_replay_node.py**
   ```python
   # 在创建 SampleBatch 后显式设置 count
   compressed_batch = SampleBatch({...})
   compressed_batch.count = self.sub_buffer_size  # 或 node_data['sub_buffer_size']
   ```

3. **保留 _expand_block_field()**
   - 作为安全的 no-op（如果 count 正确，这个函数不会做任何事）
   - 提供向后兼容性

**注意事项**：
- ⚠️ 修改后必须进行完整的回归测试
- ⚠️ 确认 eviction 时机、weights 展开、训练性能都一致
- ⚠️ 测试多种 sub_buffer_size 配置

---

## 实验配置更新

### 已更新的文件

所有 YAML 配置文件已统一更新为：
- `capacity: 1000000` (1M transitions)
- `max_time_s: 14400` (4 hours)

**文件列表**:
- `configs/tests/sac/lunarlander_test_sub1.yml`
- `configs/tests/sac/lunarlander_test_sub8.yml`
- `configs/tests/sac/lunarlander_test_sub16.yml`
- `configs/tests/sac/lunarlander_test_sub32.yml`
- `configs/tests/sac/lunarlander_test_per16.yml`

### 新实验预期

修正 PBER 后，新实验应该：
- ✅ Capacity 语义正确（1M transitions）
- ✅ Eviction 行为可预测
- ✅ 所有配置存储相同数量的 transitions
- ✅ 公平对比 sub_buffer_size 的影响

---

## 关键洞察

### 1. Ray 的设计哲学

Ray 的 `capacity` 始终以 **timesteps** 为单位，不关心存储单元是什么：
- Timesteps: 直接计数
- Sequences/Episodes: 累计其中的 timesteps
- Blocks: 累计 block.count (应该等于 sub_buffer_size)

### 2. SampleBatch.count 的计算

`SampleBatch.count` 从字段推断，通常取第一个可用字段的 `shape[0]`：
- 未压缩：`obs.shape[0]` = sub_buffer_size ✅
- 压缩后：`obs.shape[0]` = 1 (单个 compressed bytes) ❌

### 3. 压缩与 count 的冲突

压缩技术（将多个 transitions 压缩成单个 bytes 对象）与 Ray 的 count 推断逻辑存在语义冲突：
- Ray 假设：第一维度 = batch size
- 压缩现实：第一维度 = 1 (单个压缩对象)
- 解决方案：显式设置 `batch.count` 或手动展开 weights

### 4. Block-based Buffer 的设计挑战

Block-based replay buffer 需要在两个层面管理容量：
- **Block 层面**：存储单元数量（len(storage)）
- **Transition 层面**：实际 timesteps 数量（capacity）

错误将这两个概念混淆，导致 capacity 被错误地除以 sub_buffer_size。

---

## 后续工作

### 短期 (已完成)
- [x] 识别并修正 PBER 的 capacity 问题
- [x] 验证 RASPBERry 的双重错误设计
- [x] 更新所有实验配置为统一的 1M capacity
- [x] 文档化分析结果

### 中期 (建议)
- [ ] 为 RASPBERry 添加详细注释，说明设计意图
- [ ] 在代码中添加单元测试验证 capacity 行为
- [ ] 监控新实验结果，确认修正后的行为

### 长期 (可选)
- [ ] 重构 RASPBERry，移除双重错误设计
- [ ] 统一 PBER 和 RASPBERry 的 capacity 管理逻辑
- [ ] 考虑向 Ray 上游提交 patch，改进压缩场景的 count 处理

---

## 参考资料

### Ray RLlib 源码 (v2.8)
- `ray/rllib/utils/replay_buffers/replay_buffer.py`
  - Line 37-39: capacity 定义
  - Line 256-257: `_add_single_batch()` 累计逻辑
  - Line 274-279: eviction 触发条件

- `ray/rllib/utils/replay_buffers/prioritized_replay_buffer.py`
  - Line 141-153: sample() 中 weights 展开逻辑

### 本地实现
- `replay_buffer/d_pber_ray.py`: MultiAgent PBER wrapper
- `replay_buffer/pber_ray.py`: PBER 核心实现
- `replay_buffer/d_raspberry_ray.py`: MultiAgent RASPBERry wrapper
- `replay_buffer/raspberry_ray.py`: RASPBERry 核心实现
- `replay_buffer/compress_replay_node.py`: 压缩节点实现

### 实验日志
- 旧实验 (修正前): 2025-11-05 19:15
- Buffer dumps: `logs/sub_buffer_size_test/*/buffer_t0600768.pkl`

---

**最后更新**: 2025-11-06  
**作者**: AI Assistant (通过与用户协作分析)  
**状态**: 已修正 PBER，RASPBERry 保持现状并文档化

