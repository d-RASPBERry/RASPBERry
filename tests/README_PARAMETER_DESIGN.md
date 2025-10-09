# 参数传递设计说明

## 设计原则

**Bash 脚本控制参数，通过命令行传递给 Python 脚本**，避免过度工程化。

## 参数流程

```
run_compare_xxx_2h.sh  →  Python 脚本  →  训练算法
    (定义参数)         (接收参数)      (执行训练)
```

### 1. Bash 脚本（控制层）

在 `run_compare_sac_2h.sh` 和 `run_compare_ddqn_2h.sh` 中：

```bash
# === 运行参数：bash 脚本统一控制 === #
MAX_TIME_S=7200          # 2 小时
MAX_ITERATIONS=100000    # 最大迭代次数

# 启动训练
python run_sac_per_algo.py --max-time "$MAX_TIME_S" --max-iter "$MAX_ITERATIONS"
```

**优点**：
- 脚本名称（`2h`）与实际行为（7200秒）一致
- 参数集中管理，修改方便
- 符合 Shell 脚本最佳实践

### 2. Python 脚本（执行层）

所有训练脚本（`run_xxx_algo.py`）均支持命令行参数：

```python
# === 默认参数（可通过命令行覆盖）=== #
DEFAULT_MAX_TIME_S = 7200
DEFAULT_MAX_ITERATIONS = 100000

parser = argparse.ArgumentParser()
parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME_S)
parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITERATIONS)
args = parser.parse_args()

# 使用命令行参数，忽略 YAML 中的配置
max_time_s = args.max_time
max_iterations = args.max_iter
```

**优点**：
- 命令行参数优先级最高（覆盖 YAML 配置）
- 可独立运行脚本进行测试
- 参数来源清晰明确

### 3. 对比旧设计

#### ❌ 旧设计（过度工程化）

```python
# Python 脚本硬编码
MAX_TIME_S = 3600  # 但 bash 脚本叫 run_compare_sac_2h.sh（2小时）

# 从 YAML 读取
max_time_s = run_cfg.get("max_time_s", MAX_TIME_S)

# 再取最小值
max_time_s = min(max_time_s, MAX_TIME_S)
```

**问题**：
1. Bash 脚本名称与实际行为不符
2. 参数来源混乱（硬编码、YAML、脚本名称都不一致）
3. 修改运行时长需要改 Python 代码
4. `min()` 逻辑增加复杂度

#### ✅ 新设计（简洁清晰）

```bash
# Bash 脚本定义参数
MAX_TIME_S=7200  # 与脚本名称一致

# 传递给 Python
python script.py --max-time "$MAX_TIME_S"
```

```python
# Python 接收并使用
args = parser.parse_args()
max_time_s = args.max_time  # 直接使用，优先级最高
```

**优点**：
1. 参数定义与脚本名称一致
2. 单一数据源（Bash 脚本）
3. 修改灵活（只改 Bash 脚本）
4. 代码简洁，易于理解

## 使用方法

### 运行 2 小时对比实验

```bash
# SAC 算法
./run_compare_sac_2h.sh

# DDQN 算法
./run_compare_ddqn_2h.sh
```

### 自定义运行时长

#### 方法1：修改 Bash 脚本（推荐）

```bash
# 编辑 run_compare_sac_2h.sh
MAX_TIME_S=3600  # 改为 1 小时
```

#### 方法2：手动运行 Python 脚本

```bash
# 运行 30 分钟
python tests/run_sac_per_algo.py --max-time 1800

# 运行 100 次迭代
python tests/run_sac_per_algo.py --max-iter 100
```

## 参数优先级

```
命令行参数 > Python 默认值 > YAML 配置（已忽略）
```

**注意**：当前设计中，命令行参数会完全覆盖 YAML 配置中的 `max_time_s` 和 `max_iterations`。

## 文件清单

### Bash 脚本
- `run_compare_sac_2h.sh` - SAC 算法对比（PER vs RASPBERry）
- `run_compare_ddqn_2h.sh` - DDQN 算法对比（PER vs RASPBERry）

### Python 脚本
- `tests/run_sac_per_algo.py` - SAC + PER
- `tests/run_sac_raspberry_algo.py` - SAC + RASPBERry
- `tests/run_ddqn_per_algo.py` - DDQN + PER
- `tests/run_ddqn_raspberry_algo.py` - DDQN + RASPBERry

## 日志输出

所有脚本会在启动时打印配置参数：

```
[CONFIG] max_time=7200s max_iterations=100000
```

便于验证参数是否正确传递。

