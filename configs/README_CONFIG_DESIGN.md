# 配置文件设计说明

## 📋 配置文件层次

```
configs/
├── path.yml          ← 全局路径配置（所有算法共享）
├── mlflow.yml        ← 全局 mlflow 配置（所有算法共享）
├── templates/
│   ├── ddqn_base.yml ← DDQN 基础配置
│   └── sac_base.yml  ← SAC 基础配置
├── ddqn_per.yml      ← DDQN+PER 专用配置
├── ddqn_raspberry_atari.yml ← DDQN+RASPBERry 专用配置
├── sac_per.yml       ← SAC+PER 专用配置
└── sac_raspberry.yml ← SAC+RASPBERry 专用配置
```

## 🎯 配置加载顺序

参考 `DDQN_Atari_RASPBERRY.py` 的设计：

```python
# 1. 加载三个配置文件
config = load_config("configs/sac_per.yml")      # 算法配置（自动继承 templates/）
paths = load_paths("configs/path.yml")           # 全局路径
mlflow_base = load_config("configs/mlflow.yml") # 全局 mlflow

# 2. 合并 mlflow 配置
mlflow_cfg = mlflow_base.copy()
if "mlflow" in config:
    mlflow_cfg.update(config["mlflow"])  # 算法特定的 mlflow 设置覆盖全局设置
```

## 📝 各配置文件职责

### 1. `path.yml` - 全局路径配置

**职责**：定义所有算法共享的基础路径

```yaml
tmp_dir: "/tmp/"
log_base_path: "/home/seventheli/data/logging/New_RASPBERry/Atari/"
checkpoint_base_path: "/home/seventheli/data/checkpoints/New_RASPBERry/Atari/"
ray_temp_dir: "/home/seventheli/ray/"
```

**使用方式**：
```python
paths = load_paths("configs/path.yml")
log_root = Path(paths["log_base_path"]) / config["logging"]["log_subdir"]
# 例如：/home/seventheli/data/logging/New_RASPBERry/Atari/Breakout/
```

### 2. `mlflow.yml` - 全局 mlflow 配置

**职责**：定义 mlflow 的通用设置（tracking URI、全局 tags）

```yaml
tracking_uri: "http://seventheli-mlflow-home.vip.cpolar.cn"
run_tags:
  user: "researcher"
  project: "RASPBERry"
  version: "v1.0"
log_artifacts_every: 1000
```

**覆盖方式**：算法配置文件可以覆盖部分字段

```yaml
# sac_per.yml 中
mlflow:
  experiment: "SAC-PER"  # 覆盖 experiment
  tags:                   # 与全局 run_tags 合并
    algorithm: "SAC"
    buffer: "PER"
```

### 3. 算法配置文件 - 算法专用设置

**职责**：定义算法特定的配置（环境、硬件、日志、超参数）

```yaml
# sac_per.yml
extends: templates/sac_base.yml  # 继承基础配置

env_config:
  env_name: "BOX2D-CarRacing-v2"
  env_alias: "SAC-PER-CarRacing"

hardware:
  cuda_devices: "0"

logging:
  log_subdir: "Box2D/CarRacing"  # 相对于 path.yml 的 log_base_path
  log_every: 10

mlflow:  # 覆盖 mlflow.yml 的部分设置
  experiment: "SAC-PER"
  tags:
    algorithm: "SAC"
    buffer: "PER"

run_config:
  run_name_template: "{env_alias}"
  max_iterations: 100000
  max_time_s: 7200

hyper_parameters:
  # ... 算法超参数
```

## 🔧 参数优先级

```
命令行参数 > 算法配置 > 模板配置 > 全局配置
```

### 示例：`max_time_s` 的解析

1. **YAML 默认值**：`sac_per.yml` 中 `run_config.max_time_s: 7200`
2. **命令行覆盖**：`python run_sac_per_algo.py --max-time 3600`
3. **最终值**：3600（命令行优先）

### 示例：mlflow tags 的合并

```python
# mlflow.yml
run_tags:
  user: "researcher"
  project: "RASPBERry"

# sac_per.yml
mlflow:
  tags:
    algorithm: "SAC"
    buffer: "PER"

# 合并后
mlflow_cfg["tags"] = {
    "user": "researcher",
    "project": "RASPBERry",
    "algorithm": "SAC",
    "buffer": "PER"
}
```

## 🎨 设计原则

### 1. **分层配置**
- **全局配置** (`path.yml`, `mlflow.yml`) - 跨算法共享
- **模板配置** (`templates/*.yml`) - 同算法系列共享
- **算法配置** (`*_per.yml`, `*_raspberry.yml`) - 算法专用

### 2. **路径拼接**
- 基础路径在 `path.yml` 中定义
- 算法配置只需指定相对子目录 `log_subdir`
- Python 代码拼接：`log_root = log_base_path / log_subdir`

**优点**：
- 修改基础路径只需改 `path.yml`
- 算法配置保持简洁
- 便于不同环境部署

### 3. **配置覆盖**
- mlflow 配置支持部分覆盖
- `tags` 字段会合并而不是替换
- `experiment`、`tracking_uri` 可以单独覆盖

### 4. **最小硬编码**
- Python 脚本只保留配置文件路径
- 所有可调参数都在 YAML 中
- 命令行参数用于快速覆盖

## 📌 最佳实践

### 修改实验配置时

**❌ 不推荐**：修改 Python 代码中的硬编码值
```python
# 不要这样做
LOG_ROOT = "/path/to/logs"
MAX_TIME_S = 7200
```

**✅ 推荐**：修改对应的 YAML 配置
```yaml
# sac_per.yml
logging:
  log_subdir: "Box2D/CarRacing"

run_config:
  max_time_s: 7200
```

**⚡ 快速调试**：使用命令行参数
```bash
# 快速测试 30 分钟
python tests/run_sac_per_algo.py --max-time 1800
```

### 添加新算法时

1. 选择合适的模板：`templates/ddqn_base.yml` 或 `templates/sac_base.yml`
2. 创建新配置文件：`configs/my_algorithm.yml`
3. 使用 `extends` 继承模板
4. 只覆盖需要修改的参数

```yaml
# configs/my_algorithm.yml
extends: templates/sac_base.yml

env_config:
  env_name: "MyEnv-v0"
  env_alias: "MyAlgo-MyEnv"

mlflow:
  experiment: "MyAlgorithm"
  tags:
    algorithm: "MyAlgo"

# 只覆盖不同的超参数
hyper_parameters:
  lr: 1e-3
```

## 🔍 配置验证

运行脚本时会打印关键配置：

```
[CONFIG] max_time=7200s max_iterations=100000 log_every=10
LOG_BASE_DIR=/home/seventheli/data/logging/New_RASPBERry/Box2D/CarRacing/SAC-PER-CarRacing_20251002_120000
[MLFLOW] 记录已开启 -> 实验: SAC-PER
```

确认配置是否正确加载。


