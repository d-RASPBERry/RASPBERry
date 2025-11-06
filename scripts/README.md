# RASPBERry 实验脚本

本目录包含 RASPBERry 项目的所有实验、测试和验证脚本。

---

## 📁 目录结构

### 根目录脚本

- **`validate_pber_configs.sh`** - 验证所有PBER配置文件的完整性和正确性

### `ablation/` - 消融实验脚本

完整的 **PER vs PBER vs RASPBERry** 对比实验，用于正式的消融实验研究。

**可用脚本**：
- `run_sac_ablation_CarRacing.sh` - CarRacing 环境（图像输入）
- `run_sac_ablation_LunarLander.sh` - LunarLander 环境（向量输入）
- `run_sac_ablation_BipedalWalker.sh` - BipedalWalker 环境（向量输入）
- `run_sac_ablation_Pendulum.sh` - Pendulum 环境（向量输入）
- `run_sac_ablation_Walker2d.sh` - Walker2d 环境（向量输入）

**使用方法**：
```bash
# 使用4个GPU进行完整消融实验
bash scripts/ablation/run_sac_ablation_CarRacing.sh -n 4

# 使用1个GPU进行测试
bash scripts/ablation/run_sac_ablation_CarRacing.sh -n 1

# 查看帮助
bash scripts/ablation/run_sac_ablation_CarRacing.sh -h
```

**实验设计**：
每个脚本会在指定的 GPU 上运行三个对比实验：
1. **SAC-PER** - 基线算法（Prioritized Experience Replay）
2. **SAC-PBER** - 分块经验回放（无压缩）
3. **SAC-RASPBERry** - 分块+压缩经验回放

---

### `test/` - 测试和验证脚本

用于快速验证、端到端测试和实验监控。

**主要脚本**：
- `quick_test_modes_e2e.sh` - 快速验证所有压缩模式（10分钟/模式）
- `test_all_modes_e2e.sh` - 完整端到端测试（可配置时长）
- `validate_test_scripts.sh` - 验证脚本和环境完整性
- `monitor_apex.sh` - APEX分布式训练监控（支持 -q/-v 模式）

**快速开始**：
```bash
# 快速验证所有模式（~40分钟）
bash scripts/test/quick_test_modes_e2e.sh

# 完整测试（1小时）
bash scripts/test/test_all_modes_e2e.sh -n 1 -t 3600

# 监控APEX实验
bash scripts/test/monitor_apex.sh -v
```

---

## 🚀 使用场景

### 场景1：验证配置文件
在运行实验前检查配置文件的完整性。

```bash
# 验证PBER配置
bash scripts/validate_pber_configs.sh

# 验证测试脚本
bash scripts/test/validate_test_scripts.sh
```

### 场景2：正式消融实验
使用 `ablation/` 中的脚本进行完整的 PER vs PBER vs RASPBERry 对比实验。

```bash
# 在4个GPU上并行运行CarRacing消融实验
bash scripts/ablation/run_sac_ablation_CarRacing.sh -n 4

# 监控实验进度
watch -n 2 'nvidia-smi; echo; ps aux | grep "run_sac"'
```

### 场景3：快速功能验证
使用 `test/` 中的快速测试脚本验证代码改动。

```bash
# 验证所有压缩模式是否正常工作（40分钟）
bash scripts/test/quick_test_modes_e2e.sh

# 监控实验
bash scripts/test/monitor_apex.sh -q
```

---

## 📊 实验对比

| 类型 | 位置 | 用途 | 时长 | GPU需求 |
|------|------|------|------|---------|
| **配置验证** | `validate_*.sh` | 检查配置 | <1分钟 | 0 |
| **快速测试** | `test/quick_*` | 功能验证 | 10-40分钟 | 1 |
| **完整测试** | `test/test_all_*` | 全面验证 | 可配置 | 1-2 |
| **消融实验** | `ablation/` | 论文实验数据 | 数小时 | 1-4+ |

---

## 📝 日志位置

- **消融实验日志**: `logs/scripts/sac_{per,pber,raspberry}_*.log`
- **测试日志**: `logs/test/`
- **实验数据**: `/home/seventheli/data/logging/New_RASPBERry/`

---

## 🎯 脚本特点

### 统一标准
所有脚本遵循统一的代码风格：
- ✅ 使用 `set -euo pipefail` 确保错误处理
- ✅ 标准化的 header 注释（功能、用法）
- ✅ 一致的分隔线样式（80字符）
- ✅ 清晰的帮助信息（`-h` 选项）
- ✅ 详细的进度和状态反馈
- ✅ 彩色输出提升可读性

### 错误处理
- 参数验证
- 文件存在性检查
- GPU资源检测
- 进程状态监控

---

## 🔍 常见问题

**Q: ablation/ 和 test/ 的区别？**  
A: `ablation/` 用于正式的完整消融实验（论文数据），`test/` 用于快速验证和功能测试。

**Q: 应该使用哪个脚本？**  
A: 
- 日常开发和验证 → `test/`
- 正式实验和论文数据 → `ablation/`
- 配置检查 → `validate_*.sh`

**Q: 如何监控实验进度？**  
A: 
- APEX实验: `bash scripts/test/monitor_apex.sh -v`
- GPU监控: `watch -n 2 nvidia-smi`
- 日志查看: `tail -f logs/scripts/sac_*.log`

**Q: 如何清理测试环境？**  
A: 
```bash
# 杀死所有运行中的实验
pkill -f "run_sac.*algo.py"
pkill -f "run_apex.*algo.py"

# 清理临时配置
rm -rf /tmp/raspberry_*
```

---

**最后更新**: 2025-10-27  
**维护者**: RASPBERry Team

---

## 📦 已归档/移除的脚本

以下脚本已被移除（一次性使用或已过时）：
- ~~`monitor_pber_10min.sh`~~ - 包含硬编码路径的临时监控脚本
- ~~`clean_pber_configs.py`~~ - 一次性配置清理脚本（已完成任务）

