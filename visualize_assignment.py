import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config import EnvConfig
from src.environment.amta_env import AMTAEnv
from src.agent.podrl_agent import PODRLAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualize_one_episode():
    # 1. 准备
    env = AMTAEnv()
    agent = PODRLAgent(7, 7)
    agent.eval_net.load_state_dict(torch.load('checkpoints/podrl_ep2000.pth', weights_only=True))
    agent.eval_net.eval()

    # 2. 运行一个 Episode
    s, a_mat = env.reset(n_targets=9, rho=3)

    # 记录分配结果: target_id -> list of missile_types
    assignment_log = {t.t_id: [] for t in env.targets}
    target_info = {t.t_id: f"{t.t_type}\n(V={t.value})" for t in env.targets}

    done = False
    step = 0
    while not done:
        with torch.no_grad():
            s_tensor = torch.FloatTensor(s).unsqueeze(0).repeat(9, 1)
            a_tensor = torch.FloatTensor(a_mat)
            action = agent.eval_net(s_tensor, a_tensor).argmax().item()

        # 记录是谁发射的
        current_missile = env.missiles[step]
        assignment_log[env.targets[action].t_id].append(current_missile.m_type)

        next_obs, _, done, _ = env.step(action)
        if not done:
            s, a_mat = next_obs
        step += 1

    # 3. 绘图
    t_ids = list(assignment_log.keys())
    m_counts = [len(assignment_log[tid]) for tid in t_ids]
    labels = [target_info[tid] for tid in t_ids]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(t_ids)), m_counts, color='skyblue', edgecolor='black')

    plt.xticks(range(len(t_ids)), labels)
    plt.xlabel('Target (Type & Value)')
    plt.ylabel('Number of Missiles Assigned')
    plt.title('PODRL Assignment Strategy (N=9, rho=3)')

    # 在柱状图上标出导弹类型详情
    for i, bar in enumerate(bars):
        tid = t_ids[i]
        m_list = assignment_log[tid]
        m1_count = m_list.count('M1')
        m2_count = m_list.count('M2')
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"M1:{m1_count}\nM2:{m2_count}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_one_episode()