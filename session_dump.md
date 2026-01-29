# Session Dump (2026-01-29)
Repo: /home/seventheli/research/RASPBERry

## Scope
- Update Pong test launcher to run APEX/DDQN PBER and RASPBERry variants.
- Fix runtime config discovery for runner scripts.
- Stabilize environment dependencies (PyTorch + NumPy/OpenCV).

## Changes Applied
File: `scripts/ablation/run_test_pong.sh`
- Launch 4 tasks per GPU: APEX-PBER, APEX-RASPBERry, DDQN-PBER, DDQN-RASPBERry.
- Mode handling: `both` -> 4 tasks, `pber/raspberry` -> 2 tasks.
- Distinct log names for APEX vs DDQN tasks.

File: `configs/runtime.yml` (new, local)
- Added by copying from `configs/runtimes.yml`.
- Updated `paths.log_base_path` to `/home/seventheli/data/logging/New_RASPBERry/`.

## Environment Fixes
- Installed PyTorch stack: `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`,
  `torchaudio 2.5.1+cu121` (CUDA available).
- Re-pinned `numpy==1.24.3` and `opencv-python==4.6.0.66` to fix ABI mismatch
  (`numpy 2.x` breaks cv2).

## Run Attempts / Errors
- `FileNotFoundError: configs/runtime.yml` in runner scripts.
  Resolved by adding `configs/runtime.yml`.
- `RuntimeError: module compiled against ABI version 0x1000009...`
  Resolved by reinstalling NumPy/OpenCV pinned versions.
- RLlib warning: ApexDQN moved to `rllib_contrib` (non-blocking).

## Current Status
- `configs/runtime.yml` present and ignored by git.
- PyTorch 2.5.1+cu121 installed; NumPy/OpenCV pinned and import OK.
- Next step: rerun `./scripts/ablation/run_test_pong.sh -n 0` and inspect logs.
# Session Dump (2026-01-27)
Repo: /home/seventheli/research/RASPBERry

## Scope
- Compare PBER vs RASPBERry replay buffer behavior.
- Simplify `replay_buffer/d_pber_ray.py` sampling logic.
- Run an APEX-PBER end-to-end test (target 1800s).

## Analysis Notes
- `PrioritizedBlockReplayBuffer.sample()` (PBER) returns `None` on empty buffer.
- `RASPBERryReplayBuffer.sample()` also returns `None` on empty buffer.
- `d_pber_ray.py` had additional guard logic vs `d_raspberry_ray.py` (type checks, `__all__` skip).

## Changes Applied
File: `replay_buffer/d_pber_ray.py`
- Removed `SampleBatch` import (no longer used in `sample()`).
- Simplified `sample()`:
  - Removed `replay_buffer is None` guard in the policy-specific branch.
  - Removed `__all__` skip in the multi-policy branch.
  - Removed `SampleBatch` type checks (no wrapping or raising).

## E2E Run
Command:
```
python runner/run_apex_pber_algo.py --config configs/apex_pber_atari.yml --env Atari-PongNoFrameskip-v4 --gpu 0
```

Observed output:
- Training started and Ray initialized.
- Error occurred shortly after start:
  - `AttributeError: 'dict' object has no attribute 'zero_padded'`
  - Source: `ray/rllib/policy/torch_policy.py` during `compute_gradients`.

Log directory:
```
logs/experiments/Atari/Atari-PongNoFrameskip-v4/APEX-PBER-Atari-PongNoFrameskip-v4-apex_pber_atari-0-20260127_183210
```

Process status:
- Training terminated before reaching the 1800s limit.
