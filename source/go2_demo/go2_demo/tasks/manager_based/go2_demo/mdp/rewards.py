# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


## 机器人关节奖励
# 关节位置奖励
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

# 惩罚关节位置误差
def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

def joint_effort(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """观测：关节力矩（applied torque）"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


## 能量奖励 
# 基于实际能量消耗，考虑线速度和角速度的加权和来调整奖励，鼓励机器人在保持运动性能的同时降低能耗
def energy_new_actual(env, asset_cfg=SceneEntityCfg("robot"), sigma_lin=1000.0, sigma_ang=500.0, clip_lin=0.2, clip_ang=0.2):
    """Penalize the actual energy consumption."""
    asset = env.scene[asset_cfg.name]
    # 关节速度和关节力矩
    joint_vel = asset.data.joint_vel
    joint_torque = asset.data.applied_torque
    # 分子：能量消耗
    energy = torch.sum(torch.abs(joint_torque * joint_vel), dim=1)
    base_lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    base_ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    # 分母：线速度和角速度的加权和
    denom = (
        sigma_lin * torch.clamp(torch.abs(base_lin_vel_x), min=clip_lin) 
        + sigma_ang * torch.clamp(torch.abs(base_ang_vel_z), min=clip_ang)
    )
    # 返回能量奖励
    return torch.exp(-energy / denom)


## 足部奖励
# 足部滑动惩罚
def feet_slip(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(contacts * torch.sum(torch.square(feet_vel_xy), dim=-1), dim=1)

# 足部间隔距离惩罚（防止碰撞）
def foot_distance(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize when feet get too close to each other.
    
    足部索引: FL=0, FR=1, RL=2, RR=3
    计算所有足部对之间的水平距离，低于阈值时产生惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # [num_envs, 4, 2] (x, y only)
    
    # 计算所有足部对的欧氏距离
    diff = foot_pos[:, :, None, :] - foot_pos[:, None, :, :]  # [num_envs, 4, 4, 2]
    dist = torch.norm(diff, dim=-1)  # [num_envs, 4, 4]
    
    # 取上三角矩阵（不含对角线），得到所有6个足部对的距离
    n = foot_pos.shape[1]
    triu_indices = torch.triu_indices(n, n, offset=1, device=dist.device)
    pair_dists = dist[:, triu_indices[0], triu_indices[1]]  # [num_envs, 6]
    
    # 低于阈值的距离产生惩罚
    penalty = torch.clamp(threshold - pair_dists, min=0.0)
    return penalty.sum(dim=1)

# 足部在空中时间差异惩罚
def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


## 步态奖励
# 对角步态奖励（鼓励trot，抑制pace）
def gait_trot_phase(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward diagonal leg pairs being in the same contact state (trot gait).
    
    足部索引: FL=0, FR=1, RL=2, RR=3
    对角对: (FL, RR) 和 (FR, RL) 应有相同触地状态 → 鼓励 trot
    同侧对: (FL, RL) 和 (FR, RR) 应有不同触地状态 → 抑制 pace
    
    Returns: [0, 1] 之间的值，越高表示越接近理想 trot 步态
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 使用触地传感器判断足部是否触地
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    # contacts: [num_envs, 4] - [FL, FR, RL, RR]
    
    # 对角对应状态相同
    diag_match = (contacts[:, 0] == contacts[:, 3]).float() + (contacts[:, 1] == contacts[:, 2]).float()
    
    # 同侧对应状态不同
    side_diff = (contacts[:, 0] != contacts[:, 2]).float() + (contacts[:, 1] != contacts[:, 3]).float()
    
    # 归一化到 [0, 1]
    return (diag_match + side_diff) / 4.0



