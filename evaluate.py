# evaluate.py
import torch
import numpy as np
import os
from config import DQNConfig
from src.environment.amta_env import AMTAEnv
from src.agent.podrl_agent import PODRLAgent

# 修复 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate(n_episodes=20):
    # 1. 准备环境
    env = AMTAEnv()
    state_dim = 7
    action_dim = 7

    # 2. 加载 PODRL 智能体
    agent = PODRLAgent(state_dim, action_dim)
    model_path = 'checkpoints/podrl_ep2000.pth'

    if os.path.exists(model_path):
        agent.eval_net.load_state_dict(torch.load(model_path, weights_only=True))
        agent.eval_net.eval()  # 设置为评估模式
        print(f"Successfully loaded model: {model_path}")
    else:
        print(f"Error: Model {model_path} not found!")
        return

    # 3. 开始对比测试
    podrl_scores = []
    random_scores = []

    print(f"\nRunning evaluation for {n_episodes} episodes (N=9, rho=3)...")
    print("-" * 60)
    print(f"{'Episode':<10} | {'PODRL Effect':<15} | {'Random Effect':<15} | {'Improvement':<12}")
    print("-" * 60)

    for i in range(n_episodes):
        # 保证两个算法面对的是同一个初始环境，设置相同的随机种子
        seed = np.random.randint(0, 10000)

        # --- 测试 PODRL ---
        env.generator = env.generator  # Reset generator state if needed
        # 注意：要在 amta_env.py 里支持 seed 稍微麻烦，这里我们简单通过 reset 参数控制
        # 或者直接生成两套一样的环境数据，但在 current implementation 里，
        # 我们让它们跑不同的随机环境，通过多次平均来比较。

        # PODRL Run
        s, a_mat = env.reset(n_targets=9, rho=3)
        done = False
        while not done:
            # 评估时 epsilon 设为 1.0 (全贪婪) 或者直接跳过 epsilon 逻辑
            # 这里我们在 agent.select_action 里传 training=False
            # 修改 agent.select_action 逻辑: if not training -> 直接 argmax

            # 为了简单，直接手动调用网络
            with torch.no_grad():
                s_tensor = torch.FloatTensor(s).to(agent.device).unsqueeze(0).repeat(9, 1)
                a_tensor = torch.FloatTensor(a_mat).to(agent.device)
                q_vals = agent.eval_net(s_tensor, a_tensor)
                action = q_vals.argmax().item()

            next_obs, _, done, info = env.step(action)
            if not done:
                s, a_mat = next_obs
        podrl_score = info['total_effectiveness']
        podrl_scores.append(podrl_score)

        # Random Run
        s, a_mat = env.reset(n_targets=9, rho=3)
        done = False
        while not done:
            action = np.random.randint(0, 9)  # 随机选择
            _, _, done, info = env.step(action)
        random_score = info['total_effectiveness']
        random_scores.append(random_score)

        # 打印单局结果
        imp = ((podrl_score - random_score) / random_score) * 100 if random_score > 0 else 0
        print(f"{i + 1:<10} | {podrl_score:<15.2f} | {random_score:<15.2f} | {imp:>10.1f}%")

    # 4. 总结
    print("-" * 60)
    avg_podrl = np.mean(podrl_scores)
    avg_random = np.mean(random_scores)
    total_lift = ((avg_podrl - avg_random) / avg_random) * 100

    print(f"Average PODRL Effect : {avg_podrl:.4f}")
    print(f"Average Random Effect: {avg_random:.4f}")
    print(f"Overall Improvement  : {total_lift:.2f}%")


if __name__ == "__main__":
    evaluate()