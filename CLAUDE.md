# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RASPBERry is a research project implementing **RAM-Saving Prioritized Block Experience Replay** for continuous off-policy deep reinforcement learning. The project focuses on enabling large-scale RL on resource-constrained environments by dramatically reducing memory usage (from ~56GB to ~2-4GB for Atari games) while maintaining performance.

## Environment Setup

Use the conda environment configuration:

```bash
conda env create -f environment.yml
conda activate RASPBERRY
```

The environment includes PyTorch 2.5.1+cu121, Ray 2.8.0, and specialized RL libraries.

## Development Commands

### Training Commands

Run SAC training on Box2D environments:
```bash
python SAC_Box2D_RASPBERRY.py --gpu 0 --env_in CarRacing
```

Run DDQN training on Atari environments:
```bash
python DDQN_Atari_RASPBERRY.py --gpu 0 --env_in Pong
```

Run baseline PER versions:
```bash
python SAC_Box2D_PER.py --gpu 0 --env_in CarRacing
python DDQN_Atari_PER.py --gpu 0 --env_in Pong
```

### Testing Commands

Run performance benchmarks:
```bash
python -m pytest tests/test_raspberry_performance.py -v
```

Run compression tests:
```bash
python -m pytest tests/test_compress_node.py -v
```

Run environment compatibility tests:
```bash
python -m pytest tests/test_env_obs_act.py -v
```

### Ray Configuration

The project uses Ray for distributed computing. Set the Ray temp directory:
```bash
export RAY_TMPDIR=/tmp/
```

## Architecture Overview

### Core Components

1. **Replay Buffer System** (`replay_buffer/`)
   - `d_raspberry.py`: Multi-agent distributed replay buffer wrapper
   - `raspberry.py`: Core block-based prioritized replay buffer with compression
   - `compress_replay_node.py`: Individual compression units
   - Key innovation: Block-level compression reducing memory by ~28x

2. **Training System** (`trainers/`)
   - `base_trainer.py`: Abstract trainer with MLflow integration
   - `sac_raspberry_trainer.py`: SAC with RASPBERry buffer
   - `dqn_raspberry_trainer.py`: DQN with RASPBERry buffer
   - `*_per_trainer.py`: Baseline PER implementations

3. **Configuration System** (`configs/`)
   - YAML-based hierarchical configuration with template inheritance
   - `templates/`: Base configurations for algorithms
   - `path.yml`: Environment-specific paths
   - `mlflow.yml`: Experiment tracking configuration

### Key Parameters

**Buffer Configuration:**
- `sub_buffer_size`: Block size for compression (typically 8-32)
- `compress_base`: Compression axis (-1 for automatic)
- `compression_algorithm`: "lz4" (default), "zstd", "lz4hc", "blosclz"
- `compression_mode`: "D" (batch_async), "B" (batch_pool), "A" (sync)

**Memory Efficiency:**
- Standard PER: ~56GB for Atari Pong
- RASPBERry: ~2GB for same setup
- Performance impact: <5%

### Data Flow

1. **Training Loop**: Environment → BaseBuffer (node) → Block compression → Storage
2. **Sampling**: Compressed retrieval → Decompression → Training batch
3. **Prioritization**: Block-level priorities with individual sample expansion

### Configuration Inheritance

Configuration files use `extends:` for template inheritance:
```yaml
# configs/sac_raspberry.yml
extends: templates/sac_base.yml
hyper_parameters:
  replay_buffer_config:
    type: MultiAgentPrioritizedBlockReplayBuffer
```

### Compression Strategy

The system uses Blosc compression with configurable algorithms:
- **LZ4**: Fast compression/decompression (default)
- **ZSTD**: Better compression ratio
- **LZ4HC**: High compression variant
- Block-level compression with intelligent axis transposition

### MLflow Integration

Experiments are tracked using MLflow:
- Automatic metric logging every training iteration
- Artifact uploading every 1000 iterations
- Checkpoint management with Ray Tune integration

### Path Configuration

Set up paths in `configs/path.yml`:
- `log_base_path`: Training logs and metrics
- `checkpoint_base_path`: Model checkpoints
- `tmp_dir`: Ray temporary directory

## Common Development Tasks

### Adding New Algorithms

1. Create trainer in `trainers/` inheriting from `BaseTrainer`
2. Add configuration template in `configs/templates/`
3. Create environment-specific config extending template
4. Add entry script following naming pattern

### Debugging Memory Usage

Use the performance tests to analyze compression efficiency:
```bash
python tests/test_raspberry_performance.py
```

Monitor Ray memory usage with Ray Dashboard:
```bash
ray dashboard
```

### Configuration Management

All hyperparameters are externalized to YAML files. Use template inheritance to share common settings and override specific parameters per environment.

The trainer automatically handles:
- Configuration loading and validation
- MLflow experiment setup
- Ray initialization and cleanup
- Path resolution and directory creation