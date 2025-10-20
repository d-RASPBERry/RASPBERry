# 实验脚本说明

## 📋 脚本列表

### 完整实验脚本 (4小时)
每个脚本运行一个环境的 **RASPBERry + PER** 对比实验：

| 脚本 | 环境 | GPU | 类型 | 说明 |
|------|------|-----|------|------|
| `run_sac_pendulum_4h.sh` | Pendulum | 0 | 图像 (84×84×3) | CNN, 简单环境 |
| `run_sac_lunarlander_4h.sh` | LunarLander | 1 | 向量 (8维) | MLP, 中等难度 |
| `run_sac_bipedalwalker_4h.sh` | BipedalWalker | 2 | 向量 (24维) | MLP, 困难环境 |
| `run_sac_carracing_4h.sh` | CarRacing | 3 | 图像 (96×96×3) | CNN, 复杂图像 |

### 统一启动脚本
| 脚本 | 功能 |
|------|------|
| `run_all_sac_4gpu_4h.sh` | 一键启动所有4个环境的实验（推荐） |

### 快速测试脚本 (30分钟)
用于快速验证配置：

| 脚本 | 内容 | 说明 |
|------|------|------|
| `run_quick_test_batch1.sh` | Pendulum + LunarLander | 测试图像和向量环境 |
| `run_quick_test_batch2.sh` | BipedalWalker + CarRacing | 测试复杂环境 |

### 工具脚本

| 脚本 | 功能 |
|------|------|
| `monitor_experiments.sh` | 实时监控运行中的实验 |

---

## 🚀 使用方法

### 1. 一键启动所有实验（推荐⭐）

```bash
# 自动在4个GPU上并行运行所有环境
bash scripts/run_all_sac_4gpu_4h.sh
```

**GPU分配**:
- GPU 0: Pendulum (图像)
- GPU 1: LunarLander (向量)
- GPU 2: BipedalWalker (向量)
- GPU 3: CarRacing (图像)

**预计时间**: 4小时（并行）  
**总实验数**: 8个（4环境 × 2算法）  
**GPU显存**: 每个GPU约3-8GB

### 2. 单个环境实验

```bash
# GPU 0: Pendulum (4小时)
bash scripts/run_sac_pendulum_4h.sh

# GPU 1: LunarLander (4小时)
bash scripts/run_sac_lunarlander_4h.sh

# GPU 2: BipedalWalker (4小时)
bash scripts/run_sac_bipedalwalker_4h.sh

# GPU 3: CarRacing (4小时)
bash scripts/run_sac_carracing_4h.sh
```

**每个脚本会运行 2 个实验**：
- SAC-RASPBERry (压缩buffer)
- SAC-PER (基线)

**注意**: 每个脚本的GPU已固定，无需指定GPU参数

### 4. 快速测试 (验证配置)

```bash
# 测试批次1 (Pendulum + LunarLander, 30分钟)
bash scripts/run_quick_test_batch1.sh

# 测试批次2 (BipedalWalker + CarRacing, 30分钟)
bash scripts/run_quick_test_batch2.sh
```

---

## 📊 监控实验

### 查看实时日志

```bash
# 监控Pendulum实验
LOG_DIR=$(ls -td logs/sac_pendulum_2h_*/ | head -1)
tail -f ${LOG_DIR}/pendulum_raspberry.log

# 同时监控两个算法
tail -f ${LOG_DIR}/*.log
```

### 快速查看进度

```bash
# 查看最新迭代次数和奖励
watch -n 10 'grep "Iter" logs/sac_*_2h_*/*.log | tail -8'
```

### 使用监控脚本

```bash
# 自动监控所有运行中的实验
bash scripts/monitor_experiments.sh
```

---

## 📁 日志结构

每次运行会创建一个时间戳命名的日志目录：

```
logs/
├── sac_pendulum_4h_20251019_193000/
│   ├── pendulum_raspberry.log     # RASPBERry实验日志
│   └── pendulum_per.log            # PER基线日志
├── sac_lunarlander_4h_20251019_193001/
│   ├── lunarlander_raspberry.log
│   └── lunarlander_per.log
├── sac_bipedalwalker_4h_20251019_193002/
│   ├── bipedalwalker_raspberry.log
│   └── bipedalwalker_per.log
└── sac_carracing_4h_20251019_193003/
    ├── carracing_raspberry.log
    └── carracing_per.log
```

---

## 🔍 故障排查

### 检查进程状态

```bash
# 查看所有SAC实验进程
ps aux | grep "run_sac.*algo.py" | grep -v grep
```

### 检查GPU使用

```bash
# 查看GPU状态
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi
```

### 常见问题

**Q: 实验启动后立即退出**  
A: 检查日志文件开头的错误信息，通常是配置或环境问题。

**Q: GPU内存不足**  
A: 减少并行实验数量，或调整配置中的 `num_workers` 和 `num_envs_per_worker`。

**Q: CarRacing启动慢**  
A: 正常现象，图像环境初始化需要1-2分钟。

---

## 📈 预期结果

基于30分钟快速测试的结果：

| 环境 | 迭代数 (30min) | 预计迭代数 (4h) | 最终Reward范围 |
|------|----------------|-----------------|----------------|
| Pendulum | ~200 | ~1600 | -200 ~ -100 |
| LunarLander | ~400 | ~3200 | 200 ~ 250 |
| BipedalWalker | ~450 | ~3600 | 100 ~ 300 |
| CarRacing | ~180 | ~1440 | -20 ~ 800+ |

**性能对比**:
- RASPBERry ≈ PER (相近的最终性能)
- RASPBERry 内存占用 < 10% PER
- RASPBERry 采样效率 ≥ PER

---

## 🗂️ 已归档脚本

旧版脚本已移至 `scripts/archive/`:
- `run_sac_4gpu_parallel_LunarLander.sh` - 旧的4GPU并行脚本

---

## 🎯 推荐工作流

```bash
# 1. 一键启动所有实验 (4小时)
bash scripts/run_all_sac_4gpu_4h.sh

# 2. 在另一个终端监控进度
watch -n 10 'grep "Iter" logs/sac_*_4h_*/*.log 2>/dev/null | tail -16'

# 3. 查看GPU使用情况
watch -n 2 nvidia-smi

# 4. 实验完成后，查看结果汇总
ls -lh logs/sac_*_4h_*/
```

---

**最后更新**: 2025-10-19  
**版本**: v3.0 - 4小时完整实验，4GPU并行

