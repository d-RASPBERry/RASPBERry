# Ablation Experiments (5+3) — 3×5×2 + 3×3 = 39

## Scope locked (5+3 split)

- Preferred split: **5 + 3** (value/policy emphasis), not `4 + 4`
- Value-based（DDQN+APEX 共用 5）:
  - `Atlantis`, `Breakout`, `Pong`, `MiniGrid-DoorKey-8x8`, `MiniGrid-KeyCorridorS6R3`
- SAC（3）:
  - `CarRacing`, `LunarLanderContinuous`, `HalfCheetah`

## Re-run and reuse policy

- Value-based track: always full re-run
- SAC track: partial reuse is allowed **only if strict alignment passes**
  - Same git commit / code fingerprint
  - Same base config fingerprint
  - Same runner fingerprint
  - Same seed and environment

## Reporting policy (fixed)

- Seeds fixed to `{7, 42, 84}`
- Main statistic: **median + IQR** (or bootstrap CI)
- Must report:
  - Planned run count and budget basis (`max_time_s` / `max_iterations`)
  - Completed run count
  - Failed / skipped count and reasons
- Reuse cases (SAC only): clearly marked as `reused`, with fingerprint evidence

## Figure layout hints

- Value-based panels: `3 + 2`, reserve one slot for legend/summary
- SAC panels: `1 x 3` row

## Common Settings

| Parameter | Value |
| --- | --- |
| `run_config.max_time_s` | `360000` (100h) |
| `run_config.use_mlflow` | `true` |
| `replay_buffer_config.capacity` | `1000000` |

---

## DDQN -- 5 envs x 3 variants = 15 configs

Envs: Atlantis, Breakout, Pong (Atari) + MiniGrid-DoorKey-8x8, MiniGrid-KeyCorridorS6R3

Shared across all DDQN variants: `train_batch_size=32`, `num_workers=2`, `num_envs_per_worker=8`, `max_iterations=100000`

### DDQN variant parameter diff

| Parameter | PER (ti=4) | PBER | RASPBERry |
| --- | --- | --- | --- |
| `training_intensity` | **4.0** | 1.0 | 1.0 |
| `buffer type` | PrioritizedReplay | **PrioritizedBlockReplay** | **RASPBERryReplay** |
| `sub_buffer_size` | 8 | 8 | 8 |
| `compress_pool_size` | - | - | **100** |
| `compression_algorithm` | - | - | **lz4** |
| `compression_mode` | - | - | **C** |
| `chunk_size` | - | - | **20** |

### DDQN config files

| Env | PER (ti=4) | PBER | RASPBERry |
| --- | --- | --- | --- |
| Atlantis | `ddqn/per/atlantis.yml` | `ddqn/pber/atlantis.yml` | `ddqn/raspberry/atlantis.yml` |
| Breakout | `ddqn/per/breakout.yml` | `ddqn/pber/breakout.yml` | `ddqn/raspberry/breakout.yml` |
| Pong | `ddqn/per/pong.yml` | `ddqn/pber/pong.yml` | `ddqn/raspberry/pong.yml` |
| DoorKey-8x8 | `ddqn/per/minigrid_doorkey_8x8.yml` | `ddqn/pber/minigrid_doorkey_8x8.yml` | `ddqn/raspberry/minigrid_doorkey_8x8.yml` |
| KeyCorridorS6R3 | `ddqn/per/minigrid_keycorridor_s6r3.yml` | `ddqn/pber/minigrid_keycorridor_s6r3.yml` | `ddqn/raspberry/minigrid_keycorridor_s6r3.yml` |

---

## APEX -- 5 envs x 3 variants = 15 configs

Envs: same 5 as DDQN

Shared across all APEX variants: `num_workers=8`, `num_envs_per_worker=8`, `max_iterations=500`, `no_local_replay_buffer=true`

### APEX variant parameter diff

| Parameter | PER (ti=4) | PBER | RASPBERry |
| --- | --- | --- | --- |
| `training_intensity` | **4.0** | 1.0 | 1.0 |
| `train_batch_size` | 512 | **32** | **32** |
| `buffer type` | PrioritizedReplay | **PrioritizedBlockReplay** | **RASPBERryReplay** |
| `sub_buffer_size` | 16 | 16 | 16 |
| `compress_pool_size` | - | - | **200** |
| `compression_algorithm` | - | - | **lz4** |
| `compression_mode` | - | - | **C** |
| `chunk_size` | - | - | **20** |

### APEX config files

| Env | PER (ti=4) | PBER | RASPBERry |
| --- | --- | --- | --- |
| Atlantis | `apex/per/atlantis.yml` | `apex/pber/atlantis.yml` | `apex/raspberry/atlantis.yml` |
| Breakout | `apex/per/breakout.yml` | `apex/pber/breakout.yml` | `apex/raspberry/breakout.yml` |
| Pong | `apex/per/pong.yml` | `apex/pber/pong.yml` | `apex/raspberry/pong.yml` |
| DoorKey-8x8 | `apex/per/minigrid_doorkey_8x8.yml` | `apex/pber/minigrid_doorkey_8x8.yml` | `apex/raspberry/minigrid_doorkey_8x8.yml` |
| KeyCorridorS6R3 | `apex/per/minigrid_keycorridor_s6r3.yml` | `apex/pber/minigrid_keycorridor_s6r3.yml` | `apex/raspberry/minigrid_keycorridor_s6r3.yml` |

---

## SAC -- 3 envs x 3 variants = 9 configs (robustness / supporting evidence)

SAC does not vary `training_intensity`; only buffer type differs.

### SAC variant parameter diff

| Parameter | PER | PBER | RASPBERry |
| --- | --- | --- | --- |
| `buffer type` | PrioritizedReplay | **PrioritizedBlockReplay** | **RASPBERryReplay** |
| `sub_buffer_size` | - | 16 | 16 |
| `compress_pool_size` | - | - | **400** |
| `compression_algorithm` | - | - | **lz4** |
| `compression_mode` | - | - | **C** |
| `chunk_size` | - | - | **20** |

### SAC per-env shared params

| Env | Base | `train_batch_size` | `num_envs_per_worker` |
| --- | --- | --- | --- |
| CarRacing | sac_image_base | 256 | 8 |
| LunarLanderContinuous | sac_vector_base | 512 | 8 |
| HalfCheetah | sac_image_base | 256 | 4 |

### SAC config files

| Env | PER | PBER | RASPBERry |
| --- | --- | --- | --- |
| CarRacing | `sac/per/carracing.yml` | `sac/pber/carracing.yml` | `sac/raspberry/carracing.yml` |
| LunarLanderContinuous | `sac/per/lunarlander.yml` | `sac/pber/lunarlander.yml` | `sac/raspberry/lunarlander.yml` |
| HalfCheetah | `sac/per/halfcheetah_image.yml` | `sac/pber/halfcheetah_image.yml` | `sac/raspberry/halfcheetah_image.yml` |

All config paths relative to `configs/experiments/`.

---

## Script Entry Points

| Algorithm | Scripts |
| --- | --- |
| DDQN | `scripts/ablation/run_ddqn_abalation_*.sh` |
| APEX | `scripts/ablation/run_apex_abalation_*.sh` |
| SAC | `scripts/ablation/run_sac_ablation_{CarRacing,LunarLander,HalfCheetah}.sh` |

---

## Execution TODO

- [ ] Run dry-run plan generation for `all` tracks
- [ ] Confirm total job counts and GPU assignment
- [ ] Launch value-based full re-run
- [ ] Decide SAC reuse (`--allow-sac-reuse` only with strict manifest)
- [ ] Export result table with completion and budget fields
- [ ] Draw figures with the locked layouts

---

## File Count

| Scope | Count |
| --- | --- |
| DDQN | 15 (5 envs x 3 variants) |
| APEX | 15 (5 envs x 3 variants) |
| SAC | 9 (3 envs x 3 variants) |
| **Total** | **39** |
