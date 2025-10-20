# 支持的环境

## 环境列表

| 环境前缀                    | 观察空间       | 动作空间 | 推荐算法     | 状态 |
|-------------------------|------------|------|----------|----|
| `Pendulum-*`            | 84×84×3 图像 | 连续   | SAC      | ✅  |
| `BOX2DI-CarRacing`      | 96×96×3 图像 | 连续   | SAC      | ✅  |
| `BOX2DV-LunarLander*`   | 8 维向量      | 连续   | SAC      | ✅  |
| `BOX2DV-BipedalWalker*` | 24 维向量     | 连续   | SAC      | ✅  |
| `GYM-*`                 | 向量         | 连续   | SAC      | ✅  |
| `Atari-*`               | 84×84×4 灰度 | 离散   | DQN/APEX | ✅  |
| `MiniGrid-*`            | 80×80×3 图像 | 离散   | DQN      | ✅  |

**命名规则**:

- `BOX2DV-*`: 向量观察空间 (Vector) - 用于 MLP
- `BOX2DI-*`: 图像观察空间 (Image) - 用于 CNN

---

## 快速使用

### SAC 算法

```bash
# Pendulum (图像环境, CNN)
python runner/run_sac_raspberry_algo.py \
  --config configs/experiments/sac/raspberry/pendulum.yml --gpu 0

# LunarLander (向量环境, MLP)
python runner/run_sac_raspberry_algo.py \
  --config configs/experiments/sac/raspberry/lunarlander.yml --gpu 0

# CarRacing (图像环境, CNN)
python runner/run_sac_raspberry_algo.py \
  --config configs/experiments/sac/raspberry/carracing.yml --gpu 0
```

### DDQN 算法

```bash
# Atari Pong
python runner/run_ddqn_raspberry_algo.py \
  --config configs/experiments/ddqn/raspberry/pong.yml --gpu 0

# MiniGrid
python runner/run_ddqn_raspberry_algo.py \
  --env MiniGrid-Empty-8x8-v0 --gpu 0
```

---

## 环境详情

### Pendulum

- **环境 ID**: `Pendulum-Pendulum`
- **观察空间**: 84×84×3 RGB 图像
- **动作空间**: 1 维连续 [-2, 2]
- **模型**: SACLightweightCNN
- **特点**: 简单的图像环境，训练快速

### CarRacing

- **环境 ID**: `BOX2DI-CarRacing-v2`
- **观察空间**: 96×96×3 RGB 图像
- **动作空间**: 3 维连续 (转向,油门,刹车)
- **模型**: SACLightweightCNN
- **特点**: 复杂的图像环境，需要长期策略

### LunarLander

- **环境 ID**: `BOX2DV-LunarLanderContinuous`
- **观察空间**: 8 维向量
- **动作空间**: 2 维连续
- **模型**: MLP (256x256)
- **特点**: 中等难度，清晰的奖励信号

### BipedalWalker

- **环境 ID**: `BOX2DV-BipedalWalker-v3`
- **观察空间**: 24 维向量
- **动作空间**: 4 维连续
- **模型**: MLP (256x256)
- **特点**: 困难环境，需要精细的控制

### Atari

- **环境 ID**: `Atari-{GameName}NoFrameskip-v4`
- **观察空间**: 84×84×4 灰度帧栈
- **动作空间**: 离散 (游戏相关)
- **模型**: NatureDQN
- **常用游戏**: Pong, Breakout, Seaquest, SpaceInvaders

### MiniGrid

- **环境 ID**: `MiniGrid-*-v0`
- **观察空间**: 80×80×3 RGB
- **动作空间**: 离散 (7 种)
- **模型**: LightweightCNN
- **自动配置**: tile_size=10, img_size=80, max_steps=100

---

## 环境包装器

### ClipObservationWrapper

用于 `BOX2DV-*` 环境，裁剪观察值到合法范围，防止 LunarLander 等环境在极端情况下返回超出边界的值。

### PixelObservationWrapper

用于 `Pendulum-*` 环境，将状态转换为像素观察。

### ResizeObservation

统一调整图像大小：

- Pendulum: 84×84
- MiniGrid: 80×80
- CarRacing: 保持原始 96×96

---

## 常见问题

**Q: BOX2DV 和 BOX2DI 有什么区别？**  
A: BOX2DV 是向量观察 (Vector)，用 MLP；BOX2DI 是图像观察 (Image)，用 CNN。

**Q: CarRacing 的图像是 96×96，会有问题吗？**  
A: 不会。SACLightweightCNN 可以处理不同尺寸的输入。

**Q: MiniGrid 的默认参数是什么？**  
A: tile_size=10, img_size=80, max_steps=100 (基于 2024 年实验优化)。

**Q: 如何自定义环境配置？**  
A: 复制对应的 YAML 配置文件，修改 `env_config` 部分即可。

---

**版本**: v2.1 - 更新 CarRacing 支持，移除 MountainCar  
**最后更新**: 2025-10-19
