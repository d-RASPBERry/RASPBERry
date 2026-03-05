# RASPBERry

**R**eplay with **A**synchronous compre**S**sed **P**rioritized **B**lock **E**xperience **R**epla**y** — a memory-efficient prioritized experience replay buffer that combines block-level storage with asynchronous compression for deep reinforcement learning.

## Overview

RASPBERry extends standard Prioritized Experience Replay (PER) by organizing transitions into fixed-size blocks and compressing them via [blosc](https://www.blosc.org/), significantly reducing memory footprint while preserving sample quality. Built on top of [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html), RASPBERry supports DDQN, SAC, and distributed APEX-DQN, and has been evaluated across Atari, MiniGrid, Box2D, and MuJoCo environments.

### Key Ideas

- **Block-level storage** — transitions are accumulated into fixed-size blocks before insertion, enabling batch compression and coarser priority management.
- **Asynchronous compression** — blosc compression is offloaded to Ray remote tasks with backpressure control, keeping the training loop non-blocking.
- **Drop-in replacement** — integrates as a custom replay buffer within RLlib's algorithm classes, requiring minimal changes to existing training pipelines.

### Buffer Variants

| Buffer | Class | Description |
|--------|-------|-------------|
| **PER** | RLlib built-in | Standard per-transition prioritized replay |
| **PBER** | `MultiAgentPrioritizedBlockReplayBuffer` | Block-based storage, no compression (ablation baseline) |
| **RASPBERry** | `MultiAgentRASPBERryReplayBuffer` | Block-based storage + blosc compression |

## Project Structure

```
RASPBERry/
├── algorithms/          # Custom RLlib algorithm classes (DDQN, SAC, APEX variants)
├── configs/
│   ├── runtime.yml      # Global runtime paths, Ray, MLflow settings
│   ├── templates/       # Base configs (ddqn_base, sac_image_base, etc.)
│   └── experiments/     # Per-algorithm, per-buffer, per-environment configs
│       ├── ddqn/{per,pber,raspberry}/
│       ├── sac/{per,pber,raspberry}/
│       └── apex/{per,pber,raspberry}/
├── metrics/             # Logging, MLflow integration, iteration result helpers
├── models/              # Custom network architectures (e.g. SACLightweightCNN)
├── replay_buffer/       # Core buffer implementations
│   ├── raspberry_ray.py          # Single-agent RASPBERry buffer
│   ├── d_raspberry_ray.py        # Multi-agent wrapper
│   ├── pber_ray.py               # Single-agent PBER buffer
│   ├── d_pber_ray.py             # Multi-agent wrapper
│   ├── block_accumulator.py      # Transition → block accumulation logic
│   └── compress_replay_node.py   # Ray-based async compression workers
├── runner/              # Training entry-point scripts
├── scripts/
│   ├── ablation/        # Cross-buffer comparison scripts (PER vs PBER vs RASPBERry)
│   └── sub/             # Sub-experiment scripts (block-size sweeps, seeds)
├── utils/               # Config loader, environment creator, dump helpers
└── environment.yml      # Conda environment specification
```

## Getting Started

### Prerequisites

- Linux with CUDA 12.1+
- Conda (Miniconda or Anaconda)

### Installation

```bash
git clone <repo-url> RASPBERry
cd RASPBERry
conda env create -f environment.yml
conda activate RASPBERRY
```

### Configuration

Experiment configs use a YAML inheritance system:

1. **`configs/runtime.yml`** — global settings (paths, Ray cluster, MLflow URI)
2. **`configs/templates/*.yml`** — algorithm-level defaults
3. **`configs/experiments/{algo}/{buffer}/{env}.yml`** — per-experiment overrides via `extends`

Example experiment config:

```yaml
# configs/experiments/ddqn/raspberry/breakout.yml
extends: ../../../templates/ddqn_base.yml

env_config:
  id: "Atari-BreakoutNoFrameskip-v4"
  env_alias: "DDQN-RASPBERry-Breakout"

hyper_parameters:
  replay_buffer_config:
    type: MultiAgentRASPBERryReplayBuffer
    capacity: 1000000
    sub_buffer_size: 8
    compression_mode: "C"    # A=sync, B=batch-sync, C=async (recommended)
```

## Running Experiments

### Single Run

Each `runner/run_<algo>_<buffer>_algo.py` script trains one algorithm–buffer combination:

```bash
# DDQN + RASPBERry on Breakout
python runner/run_ddqn_raspberry_algo.py \
  --config configs/experiments/ddqn/raspberry/breakout.yml --gpu 0

# SAC + PER on CarRacing
python runner/run_sac_per_algo.py \
  --config configs/experiments/sac/per/carracing.yml --gpu 0

# APEX + PBER on Pong
python runner/run_apex_pber_algo.py \
  --config configs/experiments/apex/pber/pong.yml --gpu 0
```

### Ablation Studies

Scripts under `scripts/ablation/` launch PER, PBER, and RASPBERry runs in parallel for a given environment:

```bash
# DDQN ablation on Breakout (GPUs 0,1,2)
./scripts/ablation/run_ddqn_abalation_breakout.sh -n 0,1,2

# SAC ablation on CarRacing
./scripts/ablation/run_sac_ablation_CarRacing.sh -n 0

# APEX ablation on Pong
./scripts/ablation/run_apex_abalation_pong.sh -n 0,1,2
```

Options:
- `-n <gpu_ids>` — comma-separated GPU indices
- `-m shared|exclusive` — GPU allocation mode (`shared` = all runs share the same GPUs; `exclusive` = one GPU per run)

### Sub-experiments

Scripts under `scripts/sub/` sweep over block sizes and seeds:

```bash
# Freeway with sub_buffer_size ∈ {16, 32, 64}
./scripts/sub/group4_freeway_s1.sh -n 0
```

## Supported Environments

| Category | Env ID Prefix | Examples |
|----------|--------------|----------|
| Atari | `Atari-` | BreakoutNoFrameskip-v4, PongNoFrameskip-v4, FreewayNoFrameskip-v4, AtlantisNoFrameskip-v4 |
| MiniGrid | `MiniGrid-` | LavaCrossingS9N1, DoorKey-8x8 |
| Box2D (image) | `BOX2DI-` | CarRacing-v2 |
| Box2D (vector) | `BOX2DV-` | LunarLander-v2 |
| MuJoCo (vector) | `MUJOCOV-` | Walker2d-v4 |

## Compression Modes

RASPBERry supports three compression strategies via the `compression_mode` parameter:

| Mode | Strategy | Description |
|------|----------|-------------|
| **A** | Synchronous | Compress inline on the training thread |
| **B** | Batch synchronous | Compress via `ray.get()` in batch |
| **C** | Asynchronous | Offload to Ray remote workers with backpressure (recommended) |

## Tracking

Experiment metrics are logged to [MLflow](https://mlflow.org/). Set `run_config.use_mlflow: true` in the experiment config and configure the tracking URI in `configs/runtime.yml`.

## License

See [LICENSE](LICENSE) for details.
