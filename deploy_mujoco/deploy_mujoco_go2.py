"""
Go2 Sim-to-Sim: MuJoCo 部署
=======================
基于 LeggedLab G1 参考代码 (deploy_mujoco_lab_g1.py) 的结构和逻辑，适配 Go2 机器人。

核心思路：
- PD 站立模式（RL OFF）：使用训练默认姿态 default_angles_lab 作为目标，kp/kd 增益
- RL 模式（RL ON）：从策略网络推理动作，target = 当前位置 + 动作偏移
- 初始姿态使用训练默认姿态（降低基座至 0.305m 使脚着地）

按键控制（与 G1 参考代码一致）：
  6/7: 前后 (vx)  8/9: 左右 (vy)  4/5: 旋转 (yaw)
  Space: 指令置零  R: 重置  B: RL ON/OFF
"""

import argparse
import os
import sys
import time

import mujoco        # MuJoCo 物理引擎
from mujoco import viewer as mj_viewer  # 需显式导入，否则 mujoco.viewer 不可用
import numpy as np
import torch
import yaml
from pynput import keyboard  # 全局键盘监听（不依赖窗口焦点）

# 将项目根目录加入 Python 路径，以便导入 deploy_mujoco.math_lab
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy_mujoco.math_lab import project_grav, pd_control, quat_rotate, quat_conjugate

# ======== 全局变量（供键盘回调修改）========
x_vel = 0.0       # 线速度 x 指令 (m/s)
y_vel = 0.0       # 线速度 y 指令 (m/s)
yaw_vel = 0.0     # 角速度 z 指令 (rad/s)
use_rl = False     # 默认 RL 关闭，仿真启动时机器人保持站立
reset_requested = False
rl_toggled = False  # RL 切换标志（主循环中处理）  # 复位标志


def on_press(key):
    """pynput 键盘回调——全局热键，不依赖 MuJoCo 窗口焦点。"""
    global x_vel, y_vel, yaw_vel, use_rl, reset_requested, rl_toggled
    try:
        k = key.char
    except AttributeError:
        return

    # 速度指令步进 0.2，上限 ±1.0（与 G1 参考代码一致）
    if k == '6':
        x_vel = min(x_vel + 0.2, 1.0)
    elif k == '7':
        x_vel = max(x_vel - 0.2, -1.0)
    elif k == '8':
        y_vel = min(y_vel + 0.2, 1.0)
    elif k == '9':
        y_vel = max(y_vel - 0.2, -1.0)
    elif k == '4':
        yaw_vel = min(yaw_vel + 0.2, 1.0)
    elif k == '5':
        yaw_vel = max(yaw_vel - 0.2, -1.0)
    elif k == ' ':
        x_vel = 0.0
        y_vel = 0.0
        yaw_vel = 0.0
    elif k == 'r':
        reset_requested = True
    elif k == 'b':
        use_rl = not use_rl
        rl_toggled = True
        print(f"RL: {'ON' if use_rl else 'OFF (PD stand)'}")
    print(f"cmd: [{x_vel:.1f}, {y_vel:.1f}, {yaw_vel:.1f}]")


if __name__ == "__main__":
    # ======== 命令行参数解析 ========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（默认: config/go2_sim2sim_config.yaml）")
    parser.add_argument("--policy", type=str, default=None,
                        help="策略文件路径 (.pt)，覆盖配置中的路径")
    parser.add_argument("--no-rl", action="store_true",
                        help="不使用 RL 策略，仅 PD 控制")
    args = parser.parse_args()

    # ======== 加载配置 ========
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config", "go2_sim2sim_config.yaml"
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 解析路径（支持相对路径和绝对路径）
    policy_path = cfg["policy_path"]
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(project_root, policy_path)
    xml_path = cfg["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(project_root, xml_path)

    # ======== 解析配置参数 ========
    sim_dt = cfg["simulation_dt"]               # MuJoCo 仿真步长 (s)
    decimation = cfg["control_decimation"]       # 策略控制间隔（每 N 步推理一次）
    sim_duration = cfg["simulation_duration"]    # 仿真总时长 (s)
    num_obs = cfg["num_obs"]                     # 观测维度
    num_actions = cfg["num_actions"]             # 动作维度
    policy_input_dim = cfg.get("policy_input_dim", num_obs)  # 策略实际输入维度（可能含零填充）

    # 观测缩放因子（与训练配置一致）
    ang_vel_scale = cfg["ang_vel_scale"]         # 角速度缩放
    dof_pos_scale = cfg["dof_pos_scale"]         # 关节位置缩放
    dof_vel_scale = cfg["dof_vel_scale"]         # 关节速度缩放
    dof_effort_scale = cfg["dof_effort_scale"]   # 关节力矩缩放
    action_scale = cfg["action_scale"]           # 动作缩放（默认）
    action_scale_hip = cfg.get("action_scale_hip", action_scale)  # hip 动作缩放（通常更小）
    clip_actions = cfg["clip_actions"]           # 动作裁剪阈值
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float64)  # 指令缩放

    # PD 增益（分 PD 站立模式和 RL 模式，与 G1 参考代码一致）
    kp = np.array(cfg["kp"], dtype=np.float64)   # PD 站立模式位置增益
    kd = np.array(cfg["kd"], dtype=np.float64)   # PD 站立模式速度增益
    kps = np.array(cfg["kps"], dtype=np.float64)  # RL 模式位置增益（可设更硬）
    kds = np.array(cfg["kds"], dtype=np.float64)  # RL 模式速度增益

    default_angles_lab = np.array(cfg["default_angles_lab"], dtype=np.float64)  # 训练默认姿态

    # 关节名称映射（仅当训练顺序与 MuJoCo 顺序不同时需要）
    joint_names_lab = cfg["joint_names_lab"]        # 策略训练时的关节顺序
    joint_names_mujoco = cfg["joint_names_mujoco"]  # MuJoCo 树序（data.qpos[7:] 的顺序）
    torque_limits = cfg.get("torque_limits", 23.5)  # 关节力矩限制 (Nm)

    # ======== 加载策略网络 ========
    if args.no_rl:
        # --no-rl 模式：不加载策略
        policy = None
        use_rl = False
        prev_action = np.zeros(num_actions, dtype=np.float32)
    else:
        print(f"Loading policy: {policy_path}")
        policy = torch.jit.load(policy_path)   # 加载 TorchScript JIT 模型
        policy.eval()
        if hasattr(policy, "reset"):           # 学生 GRU 模型需要重置隐藏状态
            policy.reset()
        print(f"  input_dim={policy_input_dim}, output_dim={num_actions}")
        prev_action = np.zeros(num_actions, dtype=np.float32)

    # ======== 初始化 MuJoCo 仿真 ========
    m = mujoco.MjModel.from_xml_path(xml_path)  # 加载模型（含地面）
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt                      # 设置仿真步长

    # 设置初始姿态：使用训练默认姿态 + 降低基座使脚着地
    # 训练默认姿态 thigh=0.8/calf=-1.5 在 MuJoCo 中脚离地约 9cm，
    # 需降低基座至 0.305m 使脚刚好接触地面
    d.qpos[2] = 0.305
    for i in range(num_actions):
        d.qpos[7 + i] = default_angles_lab[i]    # 关节角度（训练默认姿态）
    mujoco.mj_forward(m, d)                      # 前向运动学计算
    initial_qpos = d.qpos.copy()                 # 保存初始状态（供复位用）
    initial_qvel = d.qvel.copy()

    # PD 目标位置 = 训练默认姿态（与观测偏移基准一致）
    target_dof_pos = default_angles_lab.copy()
    target_dof_pos = default_angles_lab.copy()
    action = np.zeros(num_actions, dtype=np.float32)

    # ======== 启动键盘监听 ========
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # ======== 仿真主循环 ========
    with mj_viewer.launch_passive(m, d) as viewer:
        print("[按键] 6/7:vx 8/9:vy 4/5:yaw Space:零 R:重置 B:RL开关")
        start = time.time()
        counter = 0                               # 步计数器

        while viewer.is_running() and time.time() - start < sim_duration:
            sim_start = time.time()

            # ---- 复位处理 ----
            if reset_requested:
                d.qpos[:] = initial_qpos           # 恢复初始位置
                d.qvel[:] = 0.0                    # 速度置零
                mujoco.mj_forward(m, d)
                viewer.sync()
                action[:] = 0.0
                prev_action[:] = 0.0
                target_dof_pos = default_angles_lab.copy()  # PD 目标恢复站立姿态
                reset_requested = False
                counter = 0
                print("已复位")
                continue

            # ---- RL 切换处理 ----
            if rl_toggled:
                rl_toggled = False
                if use_rl:
                    # RL 开启：重置 GRU 隐藏状态 + 动作缓存
                    if policy is not None and hasattr(policy, "reset"):
                        policy.reset()
                    action[:] = 0.0
                    prev_action[:] = 0.0
                else:
                    # RL 关闭：目标恢复训练默认姿态（站立）
                    target_dof_pos = default_angles_lab.copy()
                    action[:] = 0.0
                    prev_action[:] = 0.0

            # ---- PD 控制：根据模式选择增益和目标 ----
            if use_rl:
                # RL 模式：使用 RL 专用 PD 增益 kps/kds，
                # target_dof_pos 由策略推理更新
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], kps,
                    np.zeros_like(kds), d.qvel[6:], kds
                )
            else:
                # PD 站立模式：使用 default_angles_lab 作为固定目标，
                # 使用 PD 站立专用增益 kp/kd
                tau = pd_control(
                    default_angles_lab, d.qpos[7:], kp,
                    np.zeros_like(kd), d.qvel[6:], kd
                )

            # 力矩裁剪并施加
            d.ctrl[:] = np.clip(tau, -torque_limits, torque_limits)
            mujoco.mj_step(m, d)                   # MuJoCo 仿真步进
            counter += 1

            # ---- 策略推理（每 decimation 步执行一次）----
            if use_rl and counter % decimation == 0:
                # 读取 MuJoCo 状态
                qj = d.qpos[7:7 + num_actions].copy()       # 关节位置
                dqj = d.qvel[6:6 + num_actions].copy()      # 关节速度
                quat = d.qpos[3:7].copy()                   # 基座四元数 [w,x,y,z]
                omega_w = d.qvel[3:6].copy()                # 全局系角速度
                # 转换到机体坐标系（训练时策略使用机体系角速度）
                omega_body = quat_rotate(quat_conjugate(quat), omega_w)
                gravity = project_grav(quat)                # 重力方向投影
                cmd = np.array([x_vel, y_vel, yaw_vel], dtype=np.float64)

                # 关节数据重映射到 Lab 顺序（如果训练顺序与 MuJoCo 不同）
                qj_lab = np.array(
                    [qj[joint_names_mujoco.index(j)] for j in joint_names_lab]
                )
                dqj_lab = np.array(
                    [dqj[joint_names_mujoco.index(j)] for j in joint_names_lab]
                )

                # 观测值计算（与训练 StudentCfg 的顺序和缩放一致）
                qj_offset = (qj_lab - default_angles_lab) * dof_pos_scale   # 相对默认位置的偏移
                dqj_scaled = dqj_lab * dof_vel_scale                        # 关节速度 × 缩放
                effort_scaled = d.qfrc_actuator[6:6 + num_actions].copy() * dof_effort_scale  # 力矩

                # 拼接 57 维观测向量：ang_vel(3) + gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + effort(12) + last_action(12)
                obs = np.zeros(num_obs, dtype=np.float32)
                obs[0:3] = omega_body * ang_vel_scale
                obs[3:6] = gravity
                obs[6:9] = cmd * cmd_scale
                obs[9:21] = qj_offset
                obs[21:33] = dqj_scaled
                obs[33:45] = effort_scaled
                obs[45:57] = prev_action

                # 零填充到策略输入维度（导出策略 GRU 输入为 247 而非 57）
                # 前 57 维为学生观测，后 190 维对应 base_lin_vel(3) + height_scan(187)
                # base_lin_vel 填 0（无速度传感器），height_scan 填地面高度
                policy_in = np.zeros(policy_input_dim, dtype=np.float32)
                policy_in[:num_obs] = obs
                # height_scan: 基座到地面距离 ≈ d.qpos[2] - 地面高度(0)
                base_height = max(0.0, d.qpos[2])  # 基座离地高度
                policy_in[60:247] = base_height    # 填充 height_scan 区域

                # 策略推理
                action[:] = policy(
                    torch.from_numpy(policy_in).unsqueeze(0)
                )[0].detach().numpy()
                action = np.clip(action, -clip_actions, clip_actions)

                # 目标位置 = 当前位置 + 动作偏移 × 缩放
                # （注意：此处不从 default_angles_lab 起始，因为训练默认姿态在 MuJoCo 中不可达）
                target_from_current = np.zeros(num_actions, dtype=np.float64)
                for j, name in enumerate(joint_names_mujoco):
                    scale = action_scale_hip if "hip" in name else action_scale
                    target_from_current[j] = d.qpos[7 + j] + action[j] * scale
                target_dof_pos = np.clip(
                    target_from_current, -clip_actions, clip_actions
                )
                prev_action = action.copy()

            # viewer 同步（每 5 步同步一次，减少开销，与 G1 一致）
            if counter % 5 == 0:
                viewer.sync()

            # 保持实时仿真速度
            step_elapsed = time.time() - sim_start
            time.sleep(max(0, m.opt.timestep - step_elapsed))

    # 清理
    listener.stop()
