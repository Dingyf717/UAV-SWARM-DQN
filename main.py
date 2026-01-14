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
    # 1. 初始化环境与智能体
    env = AMTAEnv()
    # 状态维度=7, 动作维度=7 (根据之前的测试结果)
    # 你也可以动态获取: s_dim = env.reset()[0].shape[0]
    agent = PODRLAgent(state_dim=7, action_dim=7)

    # 记录训练曲线
    rewards_history = []
    loss_history = []

    # 创建模型保存目录
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    print(f"Start Training for {DQNConfig.NUM_EPISODES} episodes...")
    print(f"Device: {DQNConfig.DEVICE}")

    for episode in range(DQNConfig.NUM_EPISODES):
        # -------------------------------------------------------
        # Episode Start
        # -------------------------------------------------------
        # 论文使用 N=9, rho=3 进行训练 [cite: 589-590]
        state, action_matrix = env.reset(n_targets=9, rho=3)

        episode_buffer = []  # 暂存区：(s, a, r_local, ns, na_mat, done, t_type)
        total_local_reward = 0

        done = False
        while not done:
            # A. 选择动作
            action_idx = agent.select_action(state, action_matrix)

            # 获取被选目标的类型（用于公平采样）[cite: 425-427]
            selected_target_type = env.targets[action_idx].t_type

            # 记录当前动作向量 (用于存入Buffer)
            # action_matrix 是所有目标的矩阵，我们只需要存被选中的那个向量
            selected_action_vec = action_matrix[action_idx]

            # B. 执行环境交互
            next_obs, r_local, done, info = env.step(action_idx)

            # 解析 Next State
            if not done:
                next_state, next_action_matrix = next_obs
            else:
                next_state = None
                next_action_matrix = None

            # C. 暂存 Transition (注意：这里只存了 r_local)
            episode_buffer.append({
                's': state,
                'a': selected_action_vec,
                'r_local': r_local,
                'ns': next_state,
                'na_mat': next_action_matrix,
                'done': done,
                't_type': selected_target_type
            })

            total_local_reward += r_local

            # 状态滚动
            state = next_state
            action_matrix = next_action_matrix

        # -------------------------------------------------------
        # Episode End: Reward Backfilling
        # -------------------------------------------------------
        # 获取全局奖励 (Global Reward)
        # 在 amta_env.py 的 step 中，Done 时 info 里包含了 global_reward
        r_global = info.get('global_reward', 0.0)
        final_effectiveness = info.get('total_effectiveness', 0.0)

        alpha = EnvConfig.REWARD_ALPHA

        # 回填奖励并推入 Replay Buffer
        for step_data in episode_buffer:
            # 混合奖励公式: r = alpha * r_l + (1 - alpha) * r_g
            mixed_reward = alpha * step_data['r_local'] + (1.0 - alpha) * r_global

            agent.memory.push(
                state=step_data['s'],
                action=step_data['a'],
                reward=mixed_reward,
                next_state=step_data['ns'],
                next_action_matrix=step_data['na_mat'],
                done=step_data['done'],
                target_type=step_data['t_type']  # 关键：存入对应的子池子
            )

        # -------------------------------------------------------
        # Agent Update (Training)
        # -------------------------------------------------------
        # 可以在每个 Episode 结束时训练若干次，或者每步训练
        # 论文 Algorithm 1 是在 Episode 结束后采样训练 [cite: 343-344]
        # "At the end of an episode, a mini-batch ... is sampled"
        loss = agent.update()
        agent.update_epsilon()

        # -------------------------------------------------------
        # Logging
        # -------------------------------------------------------
        rewards_history.append(final_effectiveness)
        if loss != 0:
            loss_history.append(loss)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Ep {episode + 1}/{DQNConfig.NUM_EPISODES} | "
                  f"Avg Effect: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.2f} | "
                  f"Loss: {loss:.4f}")

        # 定期保存模型
        if (episode + 1) % 500 == 0:
            path = f"checkpoints/podrl_ep{episode + 1}.pth"
            torch.save(agent.eval_net.state_dict(), path)
            print(f"Model saved to {path}")

    # -------------------------------------------------------
    # Plotting Results
    # -------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Training Curve: Combat Effectiveness")
    plt.xlabel("Episode")
    plt.ylabel("Total Effectiveness")
    plt.savefig("training_curve.png")
    print("Training finished. Curve saved to training_curve.png")


if __name__ == "__main__":
    train()