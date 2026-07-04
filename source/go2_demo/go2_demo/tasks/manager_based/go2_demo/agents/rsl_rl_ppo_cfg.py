# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.distillation_cfg import RslRlDistillationStudentTeacherRecurrentCfg, RslRlDistillationAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_demo"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


#注意继承的类
# 命令：python scripts/rsl_rl/train.py --task Go2-Velocity-Distill-v0 --num_envs 400 --load_run 时间戳文件夹 --checkpoint model_xx.pt
@configclass
class DistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Distill the full observation teacher into a recurrent deployable student."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "go2_demo"
    obs_groups = {
        # 学生网络是student，之前的policy变成了教师网络
        "policy": ["student"],
        "teacher": ["policy"],
    }
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        # GRU读取一段历史观测
        rnn_type="gru",
        # 这个hidden_dim一定要和教师网络对应一致
        rnn_hidden_dim=247,
        rnn_num_layers=2,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=24,
    )