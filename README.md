# RASPBERry 🍓  
**RAM-Saving Prioritized Block Experience Replay**  
*Enabling continuous off-policy DRL on edge devices & large clusters*

---

## 🗂 Repository layout
```
algorithms/                # Ape‑X/DDQN nets, loss, schedulers
algorithms_with_statistics # same as above + logging hooks
replay_buffer/             # Sum‑tree, block builder, compressors
settings/                  # YAML configs for paper experiments
atari_*                    # Quick‑start Atari entry points
minigrid_*                 # MiniGrid entry points
run_trainer.py             # Generic launch helper (local ↔ HPC)
environment.yml            # Full Conda spec
utils.py                   # Misc helpers (logging, seeding, etc.)
```

---

## 🚀 Quick start

### DDQN

```bash
python atari_RASPBERry.py  --env PongNoFrameskip-v4                            --replay-capacity 1000000                            --block-size 8
```

### Ape‑X DDQN

```bash
python atari_d_RASPBERry.py        --env BreakoutNoFrameskip-v4        --num-workers 2        --block-size 16        --batch-size 512
```


## 📈 Expected results

| Env | Method | Peak RAM |
|-----|--------|----------|
| Pong | **RASPBERry** | **2 GB** |
| Pong | DPER | 56 GB    |
| Breakout | **RASPBERry** | **4 GB** |
| Breakout | PER | 57 GB    |

---

## 📝 Citation
```bibtex

```

---

## 🔒 License
Released under the MIT License; see `LICENSE`.  
*Note:* This repo is anonymised for peer review—please avoid adding personally identifying info until the review period ends.*


---

Enjoy faster, slimmer replay buffers! 🍓
