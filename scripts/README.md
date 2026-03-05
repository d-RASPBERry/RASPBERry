# RASPBERry Experiment Scripts

The current `scripts/` directory only contains scripts and documentation for **ablation experiments (5+3)**.

- Canonical reference: `scripts/ablation/ENVIRONMENTS.md`

## Directory Structure

- `ablation/`: Ablation experiment scripts (PER(ti=4) vs PBER vs RASPBERry)
  - **DDQN (5 envs)**: `scripts/ablation/run_ddqn_abalation_*.sh`
  - **APEX (5 envs)**: `scripts/ablation/run_apex_abalation_*.sh`
  - **SAC (3 envs)**: `scripts/ablation/run_sac_ablation_{CarRacing,LunarLander,HalfCheetah}.sh`

## Usage

All scripts support:

- `-n <gpu_ids>`: Comma-separated GPU list (default `0`, e.g. `0,1,2`)
- `-m shared|exclusive`: Shared/exclusive mode (default `shared`)
- `-h`: Help

Examples:

```bash
# DDQN Breakout: submit PER/PBER/RASPBERry jobs on GPU 0,1
bash scripts/ablation/run_ddqn_abalation_breakout.sh -n 0,1

# APEX Pong: exclusive mode (requires GPU count to be a multiple of 3)
bash scripts/ablation/run_apex_abalation_pong.sh -n 0,1,2 -m exclusive

# SAC HalfCheetah: single GPU run
bash scripts/ablation/run_sac_ablation_HalfCheetah.sh -n 0
```

## Conventions

- PER only uses `ti=4` (`training_intensity=4`)

## Logs

- Default output to `logs/scripts/` (scripts auto-create the directory)

