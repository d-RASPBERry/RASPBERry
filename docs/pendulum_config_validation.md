# Pendulum 配置验证策略

目标：确认 `sac/pendulum` 三套 YAML（RASPBERry / PBER / PER）与启动脚本、Runner 逻辑一致，确保实验参数落地。

## 1. YAML 层面
- 打开 `configs/experiments/sac/{raspberry,pber,per}/pendulum.yml`
- 对照模板字段：
  - `hyper_parameters.train_batch_size == 64`
  - `hyper_parameters.training_intensity == 1.0`
  - `replay_buffer_config.capacity == 600000`
  - `replay_buffer_config.sub_buffer_size == 8`（仅 RASPBERry / PBER）
  - `num_workers == 3`
- 确认 `max_time_s == 360000`（100h），`env_alias`、`mlflow` 标签正确。

## 2. Bash 启动脚本
- 打开 `scripts/ablation/run_sac_ablation_Pendulum.sh`
- 检查：
  - `PER_CONFIG`, `PBER_CONFIG`, `RASP_CONFIG` 指向上述 YAML
  - 三个 `python runner/... --config ${X_CONFIG}` 按 PER → PBER → RASP 顺序执行
  - 记录日志目录、GPU 传参、延迟启动设置无误

## 3. Runner 读取逻辑
### 共通检查
- `ConfigLoader` 注入 `runtime.yml` → `config['runtime']` 可用
- `run_cfg = config["run_config"]` → `max_time_s` 落地
- `hyper = config["hyper_parameters"]` → 后续构造算法

### 定制回放
- `run_sac_raspberry_algo.py`
  - 构造 `MultiAgentRASPBERryReplayBuffer`
  - 写入 `sub_buffer_size`, `capacity`, 压缩相关键
- `run_sac_pber_algo.py`
  - 构造 `MultiAgentPrioritizedBlockReplayBuffer`
  - 日志记录器同步复用
- `run_sac_per_algo.py`
  - 使用 `MultiAgentPrioritizedReplayBuffer`

### 训练控制
- `max_time_s` 用于 while 循环超时判断
- `num_workers` / `num_envs_per_worker` 传给 Ray 初始化
- `train_batch_size`、`training_intensity` 由 `hyper` 直接返回给算法

## 4. 交叉验证步骤
1. `yaml.safe_load` 合并模板 → 核对关键字段（可使用短脚本或手动）
2. 搜索 runner 中 `.get("max_time_s", ...)`、`replay_buffer_config` 赋值，确认键名一致
3. 在 Bash 中执行 `echo ${PER_CONFIG}` 等，确认路径无拼写
4. 实际 dry-run（`--config` 指向 YAML，`--env` 默认即可），观察日志输出：
   - `Log dir: ... Pendulum-Pendulum`
   - `[Iter ...]` 日志中 `buffer.est_size_gb` 正常
   - 若启用 MLflow，run name 匹配 `env_alias-GPU-时间戳`

## 5. 结果记录
- 验证完成后，在实验文档中注明：
  - 批次、容量、时间限制、worker 数保持一致
  - Runner / Bash 路径一致性
  - 需要持续关注的点：压缩率、Ray object store 内存、水位监控

