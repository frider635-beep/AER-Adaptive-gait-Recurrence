# Go2 Sim-to-Sim: MuJoCo Deployment

将 Isaac Lab 训练的 Go2 策略部署到 MuJoCo 仿真环境中。

## 依赖

```bash
pip install mujoco
```

## 使用

```bash
# 使用默认配置运行
python deploy_mujoco/deploy_mujoco_go2.py

# 指定配置文件
python deploy_mujoco/deploy_mujoco_go2.py --config deploy_mujoco/config/go2_sim2sim_config.yaml

# 指定策略文件
python deploy_mujoco/deploy_mujoco_go2.py --policy deploy_mujoco/pre_train/policy_AER.pt

# 不带 RL 策略运行（仅 PD 控制）
python deploy_mujoco/deploy_mujoco_go2.py --no-rl
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| `W/S` | 前后移动 (vx) |
| `A/D` | 左右移动 (vy) |
| `Q/E` | 旋转 (yaw) |
| `Space` | 急停/置零指令 |
| `R` | 重置仿真 |
| `B` | 切换 RL / 纯 PD 模式 |
| `ESC` | 退出 |

## 文件结构

```
deploy_mujoco/
├── readme.md                              # 本文件
├── config/
│   └── go2_sim2sim_config.yaml            # 仿真和策略配置
├── pre_train/
│   └── policy_AER.pt                      # 导出的策略文件 (JIT TorchScript)
├── resources/
│   └── robots/
│       └── go2_description/
│           ├── go2.xml                    # MuJoCo 模型文件
│           ├── meshes/                    # 碰撞/视觉网格
│           └── urdf/                      # URDF 描述文件
├── deploy_mujoco_go2.py                  # 主部署脚本
└── math_lab.py                           # 工具函数（PD控制、重力投影）
```
