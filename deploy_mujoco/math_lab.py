"""
MuJoCo Sim2Sim 工具函数

参考 LeggedLab: deploy_mujoco/math_lab.py
"""

import numpy as np


def project_grav(quat):
    """
    将重力向量 [0,0,-1] 通过四元数旋转到机体坐标系。
    
    参数:
        quat: [w, x, y, z] 四元数
    返回:
        gravity_vec: [3] 机体坐标系下的重力方向
    """
    v = np.array([0, 0, -1], dtype=np.float64)
    q_w = quat[0]
    q_vec = quat[1:4]

    # 四元数旋转: R(v) = v * (2*q_w^2 - 1) - 2*q_w*(q_vec × v) + 2*(q_vec·v)*q_vec
    a = v * (2 * q_w**2 - 1)
    cross_prod = np.cross(q_vec, v)
    b = 2 * q_w * cross_prod
    dot_prod = np.dot(q_vec, v)
    c = 2 * q_vec * dot_prod

    rotated_v = a - b + c
    return rotated_v


def pd_control(target_q, q, kp, target_dq=None, dq=None, kd=None):
    """
    PD 控制器：从目标位置计算控制力矩。
    
    参数:
        target_q: [N] 目标关节位置
        q: [N] 当前关节位置
        kp: float 或 [N] 位置增益
        target_dq: [N] 目标关节速度（默认为 0）
        dq: [N] 当前关节速度
        kd: float 或 [N] 速度增益
    返回:
        tau: [N] 控制力矩
    """
    if target_dq is None:
        target_dq = np.zeros_like(q)
    if dq is None:
        dq = np.zeros_like(q)
    if kd is None:
        kd = 0.0

    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_multiply(q1, q2):
    """
    四元数乘法 q = q1 * q2。
    
    参数:
        q1: [w, x, y, z]
        q2: [w, x, y, z]
    返回:
        q: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    """四元数共轭。"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    """用四元数旋转向量 v。"""
    q_v = np.array([0, v[0], v[1], v[2]])
    q_result = quat_multiply(quat_multiply(q, q_v), quat_conjugate(q))
    return q_result[1:]
