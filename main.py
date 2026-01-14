# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime

from config import DQNConfig, EnvConfig
from src.environment.amta_env import AMTAEnv
from src.agent.podrl_agent import PODRLAgent


def train():
    env = AMTAEnv()
    # 这里的维度根据你的实际情况，之前代码是7
    agent = PODRLAgent(state_dim=7, action_dim=7)

    rewards_history = []

    # [新增监控] 1. 初始化一个记录本，用来存各项指标的历史数据
    stats_recorder = {
        "state_max": [],  # 记录输入状态的最大值
        "action_max": [],  # 记录输入动作矩阵的最大值
        "loss": [],  # 记录 Loss
        "q_mean": [],  # 记录平均 Q 值
        "grad_norm": []  # 记录梯度模长（判断是否爆炸）
    }

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    print(f"Start Training for {DQNConfig.NUM_EPISODES} episodes...")

    for episode in range(DQNConfig.NUM_EPISODES):
        # -------------------------------------------------------
        # Episode Start
        # -------------------------------------------------------
        state, action_matrix = env.reset(n_targets=9, rho=3)

        # [新增监控] 2. 探针 A：检查输入数据的尺度
        # 如果这里打印出来经常 > 10，说明必须做归一化
        stats_recorder["state_max"].append(np.max(state))
        stats_recorder["action_max"].append(np.max(action_matrix))

        episode_buffer = []

        done = False
        while not done:
            action_idx = agent.select_action(state, action_matrix)
            selected_target_type = env.targets[action_idx].t_type
            selected_action_vec = action_matrix[action_idx]

            next_obs, r_local, done, info = env.step(action_idx)

            if not done:
                next_state, next_action_matrix = next_obs
            else:
                next_state = None
                next_action_matrix = None

            episode_buffer.append({
                's': state,
                'a': selected_action_vec,
                'r_local': r_local,
                'ns': next_state,
                'na_mat': next_action_matrix,
                'done': done,
                't_type': selected_target_type
            })

            state = next_state
            action_matrix = next_action_matrix

        # -------------------------------------------------------
        # Episode End: Reward Backfilling
        # -------------------------------------------------------
        r_global = info.get('global_reward', 0.0)
        final_effectiveness = info.get('total_effectiveness', 0.0)
        alpha = EnvConfig.REWARD_ALPHA

        for step_data in episode_buffer:
            mixed_reward = alpha * step_data['r_local'] + (1.0 - alpha) * r_global

            # [注意] 如果之后发现 Q 值过大，可以在这里除以 10.0 进行缩放
            # mixed_reward = mixed_reward / 10.0

            agent.memory.push(
                state=step_data['s'],
                action=step_data['a'],
                reward=mixed_reward,
                next_state=step_data['ns'],
                next_action_matrix=step_data['na_mat'],
                done=step_data['done'],
                target_type=step_data['t_type']
            )

        # -------------------------------------------------------
        # Agent Update & [新增监控]
        # -------------------------------------------------------
        # 注意：这里假设你已经修改了 podrl_agent.py 的 update() 方法，让它返回一个字典
        diagnostics = agent.update()
        agent.update_epsilon()

        # [新增监控] 3. 探针 B：记录网络健康状况
        # 如果 diagnostics 不为空（说明发生了更新），就把数据记下来
        if diagnostics:
            # 兼容旧代码：如果未修改 agent.py，diagnostics 可能只是一个 float loss
            if isinstance(diagnostics, dict):
                stats_recorder["loss"].append(diagnostics["loss"])
                stats_recorder["q_mean"].append(diagnostics["q_mean"])
                stats_recorder["grad_norm"].append(diagnostics["grad_norm"])
            else:
                # 如果你还没改 agent.py，暂时只记 loss，但这样就看不到梯度了
                stats_recorder["loss"].append(diagnostics)

        # -------------------------------------------------------
        # Logging & [新增监控报告]
        # -------------------------------------------------------
        rewards_history.append(final_effectiveness)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])

            # [新增监控] 4. 打印体检报告
            # 计算最近 100 轮的平均值
            avg_s_max = np.mean(stats_recorder["state_max"][-100:])
            avg_a_max = np.mean(stats_recorder["action_max"][-100:])

            # 安全获取 update 相关的指标（防止初期还没开始 update 为空）
            avg_q = np.mean(stats_recorder["q_mean"][-100:]) if stats_recorder["q_mean"] else 0
            avg_grad = np.mean(stats_recorder["grad_norm"][-100:]) if stats_recorder["grad_norm"] else 0
            avg_loss = np.mean(stats_recorder["loss"][-100:]) if stats_recorder["loss"] else 0

            print("-" * 60)
            print(f"Episode {episode + 1} | Avg Effect: {avg_reward:.2f}")
            print(f"--- 诊断报告 (Diagnostics) ---")
            print(f"1. 输入数据最大值 (Input Scale):")
            print(f"   State Max: {avg_s_max:.2f} (理想值应接近 1.0)")
            print(f"   Action Max: {avg_a_max:.2f} (如果 > 10，必须归一化！)")
            print(f"2. 网络状态 (Network Health):")
            print(f"   Avg Q-Value: {avg_q:.2f} (如果 > 200，说明奖励可能需要缩放)")
            print(f"   Avg Grad Norm: {avg_grad:.4f} (如果 > 10 或被截断，说明梯度爆炸)")
            print("-" * 60)

        if (episode + 1) % 500 == 0:
            path = f"checkpoints/podrl_ep{episode + 1}.pth"
            torch.save(agent.eval_net.state_dict(), path)
            print(f"Model saved to {path}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Training Curve")
    plt.savefig("training_curve.png")


if __name__ == "__main__":
    train()