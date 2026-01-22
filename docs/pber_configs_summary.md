# PBER 配置文件创建完成

**日期**: 2025-10-26  
**目标**: 为独立 PBER runner 创建配置文件

---

## ✅ 创建的配置文件

### SAC
- ✅ `sac_pber_image.yml` - SAC + PBER (图像观测)
- ✅ `sac_pber_vector.yml` - SAC + PBER (向量观测)

### DDQN
- ✅ `ddqn_pber_atari.yml` - DDQN + PBER (Atari)

### APEX-DQN
- ✅ `apex_pber_atari.yml` - APEX-DQN + PBER (Atari)

---

## 📊 配置对比

### SAC Image 配置对比

| 参数 | PER | PBER | RASPBERry |
|------|-----|------|-----------|
| **Buffer Type** | `MultiAgentPrioritizedReplayBuffer` | `MultiAgentPrioritizedBlockReplayBuffer` | `MultiAgentPrioritizedBlockReplayBuffer` |
| **Capacity** | 100000 | 100000 | 100000 |
| **Block Size** | - | 16 | 16 |
| **Priority α** | (default) | 0.5 | 0.5 |
| **Priority β** | (default) | 1.0 | 1.0 |
| **Compression** | ❌ | ❌ | ✅ lz4 |
| **Compression Mode** | - | - | D (async) |
| **Chunk Size** | - | - | 20 |

---

### DDQN Atari 配置对比

| 参数 | PER | PBER | RASPBERry |
|------|-----|------|-----------|
| **Buffer Type** | `MultiAgentPrioritizedReplayBuffer` | `MultiAgentPrioritizedBlockReplayBuffer` | `MultiAgentPrioritizedBlockReplayBuffer` |
| **Capacity** | 100000 | 100000 | 100000 |
| **Block Size** | - | 8 | 8 |
| **Priority α** | 0.5 | 0.5 | 0.5 |
| **Priority β** | 1.0 | 1.0 | 1.0 |
| **Compression** | ❌ | ❌ | ✅ lz4 |
| **Compression Mode** | - | - | D (async) |

---

### APEX-DQN Atari 配置对比

| 参数 | PER | PBER | RASPBERry |
|------|-----|------|-----------|
| **Buffer Type** | `MultiAgentPrioritizedReplayBuffer` | `MultiAgentPrioritizedBlockReplayBuffer` | `MultiAgentPrioritizedBlockReplayBuffer` |
| **Capacity** | 500000 | 500000 | 500000 |
| **Block Size** | - | 16 | 16 |
| **Train Batch** | (default) | 32 | 32 |
| **Priority α** | 0.5 | 0.5 | 0.5 |
| **Priority β** | 1.0 | 1.0 | 1.0 |
| **Compression** | ❌ | ❌ | ✅ lz4 |
| **Worker Priority** | true | true | true |
| **Shards Colocated** | true | true | true |

---

## 📝 PBER 配置特点

### 核心原则
✅ **纯分块** - 使用 block-level 存储  
❌ **无压缩** - 不包含任何压缩参数  
✅ **简洁** - 只有必要的参数

### 与 PER 的区别
- ✅ 添加 `sub_buffer_size` (block size)
- ✅ 使用 `MultiAgentPrioritizedBlockReplayBuffer`
- ❌ 无其他额外参数

### 与 RASPBERry 的区别
- ❌ 无 `compress_base`
- ❌ 无 `compress_pool_size`
- ❌ 无 `compression_algorithm`
- ❌ 无 `compression_level`
- ❌ 无 `compression_nthreads`
- ❌ 无 `compression_mode`
- ❌ 无 `chunk_size`

---

## 📖 使用示例

### SAC + PBER (CarRacing)
```bash
python runner/run_sac_pber_algo.py \
  --config configs/sac_pber_image.yml \
  --env CarRacing-v2 \
  --gpu 0
```

### DDQN + PBER (Atari Breakout)
```bash
python runner/run_ddqn_pber_algo.py \
  --config configs/ddqn_pber_atari.yml \
  --env Atari-Breakout \
  --gpu 0
```

### APEX + PBER (Atari Pong)
```bash
python runner/run_apex_pber_algo.py \
  --config configs/apex_pber_atari.yml \
  --env Atari-Pong \
  --gpu 0
```

---

## 🗂️ 完整配置文件列表

```
configs/
├── sac_per_image.yml           # PER baseline
├── sac_pber_image.yml          # ✨ NEW: PBER
├── sac_raspberry_image.yml     # RASPBERry
│
├── sac_per_vector.yml
├── sac_pber_vector.yml         # ✨ NEW: PBER
├── sac_raspberry_vector.yml
│
├── ddqn_per.yml
├── ddqn_pber_atari.yml         # ✨ NEW: PBER
├── ddqn_raspberry_atari.yml
│
├── apex_per.yml
├── apex_pber_atari.yml         # ✨ NEW: PBER
└── apex_raspberry_atari.yml
```

---

## 🎯 实验对比建议

### 标准对比实验
```bash
# 1. PER baseline
python runner/run_sac_per_algo.py --config configs/sac_per_image.yml --gpu 0

# 2. PBER (block-level)
python runner/run_sac_pber_algo.py --config configs/sac_pber_image.yml --gpu 0

# 3. RASPBERry (block-level + compression)
python runner/run_sac_raspberry_algo.py --config configs/sac_raspberry_image.yml --gpu 0
```

### 对比维度
1. **时间效率**: 每轮迭代时间
2. **内存占用**: Buffer 内存使用
3. **学习性能**: Episode reward
4. **吞吐量**: Timesteps/second

---

## 📋 配置检查清单

### 必需参数（PBER）
- [x] `capacity` - buffer 容量
- [x] `type: MultiAgentPrioritizedBlockReplayBuffer`
- [x] `sub_buffer_size` - block 大小
- [x] `prioritized_replay_alpha` - PER α
- [x] `prioritized_replay_beta` - PER β
- [x] `prioritized_replay_eps` - 数值稳定性

### 禁止参数（PBER应避免）
- [ ] `compress_base`
- [ ] `compress_pool_size`
- [ ] `compression_algorithm`
- [ ] `compression_level`
- [ ] `compression_nthreads`
- [ ] `compression_mode`
- [ ] `chunk_size`

---

## 📈 预期内存占用

### SAC CarRacing (capacity=100000, sub_buffer_size=16)

| Buffer | Blocks | Transitions | Memory (Est.) |
|--------|--------|-------------|---------------|
| PER | - | 100000 | ~55 GB |
| PBER | 6250 | 100000 | ~55 GB |
| RASPBERry | 6250 | 100000 | ~2-10 GB (60-95% 减少) |

### DDQN Atari (capacity=100000, sub_buffer_size=8)

| Buffer | Blocks | Transitions | Memory (Est.) |
|--------|--------|-------------|---------------|
| PER | - | 100000 | ~57 GB |
| PBER | 12500 | 100000 | ~57 GB |
| RASPBERry | 12500 | 100000 | ~2-10 GB (60-95% 减少) |

**注意**: PBER 和 PER 内存占用相似，主要优势在操作效率（O(M/m) vs O(M)）

---

## ✅ 验证步骤

1. ✅ 配置文件语法检查
2. ⏭️ Runner 启动测试
3. ⏭️ Buffer 初始化验证
4. ⏭️ 短时间训练测试
5. ⏭️ 内存占用监控

---

**创建时间**: 2025-10-26  
**状态**: ✅ 所有配置文件创建完成  
**下一步**: 运行验证测试

