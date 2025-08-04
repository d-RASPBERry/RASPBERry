# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RASPBERry is a research project implementing **RAM-Saving Prioritized Block Experience Replay** for continuous off-policy deep reinforcement learning. The project focuses on enabling large-scale RL on resource-constrained environments by dramatically reducing memory usage (from ~56GB to ~2-4GB for Atari games) while maintaining performance.

## Environment Setup

```bash
# Create conda environment from specification
conda env create -f environment.yml
conda activate gymnasium

# Alternative environment file available
conda env create -f ber_environment.yml
```

## Common Commands

### Training Commands

**DDQN with RASPBERry:**
```bash
python atari_RASPBERry.py --env PongNoFrameskip-v4 --replay-capacity 1000000 --block-size 8
```

**Distributed Ape-X DDQN:**
```bash
python atari_d_RASPBERry.py --env BreakoutNoFrameskip-v4 --num-workers 2 --block-size 16 --batch-size 512
```

**Generic Training Launcher:**
```bash
python run_trainer.py  # Uses YAML configs from settings/
```

### Environment-Specific Entry Points

- `atari_*.py` - Atari game environments
- `minigrid_*.py` - MiniGrid environments  
- `*_per.py` - Prioritized Experience Replay variants
- `*_dper.py` - Distributed PER variants

## Architecture Overview

### Core Components

**Replay Buffer System** (`replay_buffer/`):
- `mpber_ram_saver_v7.py` - Main RAM-saving prioritized block replay buffer
- `mpber_ram_saver_v8.py` - Newer version with optimizations
- `replay_node.py` - Base buffer abstraction
- `ber.py`, `mpber.py` - Standard buffer implementations

**Algorithm Implementations** (`algorithms/`):
- `apex_ddqn.py` - Ape-X DDQN with custom replay buffer integration
- `ddqn_pber.py` - DDQN with Prioritized Block Experience Replay

**Statistics Variants** (`algorithms_with_statistics/`):
- Same algorithms with additional logging and metrics collection

### Key Technical Details

**Block-Based Compression:**
- Uses `blosc` compression for efficient storage
- Block sizes typically 8-16 for optimal memory/performance tradeoff
- Supports arbitrary observation dimensions beyond images

**Distributed Training:**
- Ray-based distributed execution
- Multi-worker sampling with centralized replay buffer
- Custom priority update mechanisms for block-based sampling

**Configuration System:**
- YAML-based configuration in `settings/`
- Uses `dynaconf` for dynamic configuration management
- Separate configs for different environments and algorithms

### Memory Management

The core innovation is block-based experience storage with compression:
- Experiences grouped into blocks before compression
- Significantly reduces memory footprint vs standard replay buffers
- Priority updates operate on block-level rather than individual transitions

## Development Workflow

**Checkpoint Management:**
- Results saved to `checkpoints/{run_name}/`
- Automatic checkpoint saving every N iterations (configurable)
- JSON logs for training metrics

**Logging:**
- Training logs in `logs/{run_name}/`
- Structured JSON format for easy analysis
- Buffer statistics tracking (size, compression ratio, etc.)

**Ray Configuration:**
- Distributed training uses Ray with custom system configs
- Typical setup: 20 CPUs, 1 GPU
- Dashboard disabled by default for resource efficiency

## Important Notes

- No formal test suite detected - validation through training runs
- Uses PyTorch with CUDA 11.3 support
- Requires significant computational resources for distributed training
- Research codebase - some experimental features may be unstable
- Block size tuning is critical for memory/performance balance

## File Naming Conventions

- `*_RASPBERry.py` - Main implementations using the novel replay buffer
- `*_d_*.py` - Distributed/multi-worker versions
- `*_example.py` - Usage examples and tutorials
- `*_test_*.py` - Testing/validation scripts
- `v7`, `v8` suffixes indicate version evolution of replay buffer implementations