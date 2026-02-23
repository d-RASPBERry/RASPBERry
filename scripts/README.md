# RASPBERry 实验脚本

当前 `scripts/` 目录仅保留 **消融实验（5+3）** 相关脚本与说明。

- 权威说明：`scripts/ablation/ENVIRONMENTS.md`

## 目录结构

- `ablation/`: 消融实验脚本（PER(ti=4) vs PBER vs RASPBERry）
  - **DDQN（5 envs）**：`scripts/ablation/run_ddqn_abalation_*.sh`
  - **APEX（5 envs）**：`scripts/ablation/run_apex_abalation_*.sh`
  - **SAC（3 envs）**：`scripts/ablation/run_sac_ablation_{CarRacing,LunarLander,HalfCheetah}.sh`

## 使用方法

所有脚本都支持：

- `-n <gpu_ids>`：逗号分隔 GPU 列表（默认 `0`，例如 `0,1,2`）
- `-m shared|exclusive`：共享/独占模式（默认 `shared`）
- `-h`：帮助

示例：

```bash
# DDQN Breakout：在 GPU 0,1 上分别提交 PER/PBER/RASPBERry 任务
bash scripts/ablation/run_ddqn_abalation_breakout.sh -n 0,1

# APEX Pong：独占模式（需要 GPU 数量为 3 的倍数）
bash scripts/ablation/run_apex_abalation_pong.sh -n 0,1,2 -m exclusive

# SAC HalfCheetah：单卡运行
bash scripts/ablation/run_sac_ablation_HalfCheetah.sh -n 0
```

## 约定

- PER 仅保留 `ti=4`（`training_intensity=4`）

## 日志

- 默认输出到 `logs/scripts/`（脚本会自动创建目录）

