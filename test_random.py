import numpy as np
import random
from src.environment.amta_env import AMTAEnv
from config import DQNConfig, EnvConfig


def evaluate_random_policy(n_episodes=1000):
    """
    运行随机策略评估
    :param n_episodes: 测试的回合数，默认1000以获得稳定的统计结果
    """
    # 初始化环境
    env = AMTAEnv()

    # 用于存储每个 Episode 的最终效能
    effectiveness_history = []

    print(f"Starting Random Policy Evaluation for {n_episodes} episodes...")
    print(f"Scenario: N=9 targets, rho=3 (Total 27 missiles)")

    for episode in range(n_episodes):
        # 1. 重置环境 (与训练时相同的配置 N=9, rho=3)
        # state, action_matrix 都是初始观测值
        state, action_matrix = env.reset(n_targets=9, rho=3)

        done = False
        while not done:
            # 2. 随机动作选择 (Random Selection)
            # action_matrix 的行数就是当前可选目标的数量
            n_targets_current = action_matrix.shape[0]
            action_idx = random.randint(0, n_targets_current - 1)

            # 3. 执行动作
            next_obs, r_local, done, info = env.step(action_idx)

            # state 更新 (虽然随机策略不需要 state，但保持逻辑完整)
            if not done:
                state, action_matrix = next_obs

            # 4. 记录结果
            if done:
                # 从 info 中提取最终的 Combat Effectiveness
                final_effectiveness = info.get('total_effectiveness', 0.0)
                effectiveness_history.append(final_effectiveness)

    # 计算统计数据
    avg_effect = np.mean(effectiveness_history)
    std_effect = np.std(effectiveness_history)
    min_effect = np.min(effectiveness_history)
    max_effect = np.max(effectiveness_history)

    print("\n" + "=" * 50)
    print(f"📊 RANDOM POLICY RESULTS (Baseline)")
    print("=" * 50)
    print(f"Total Episodes : {n_episodes}")
    print(f"Mean Effect    : {avg_effect:.4f}  <-- 您的模型应该显著高于此值")
    print(f"Std Deviation  : {std_effect:.4f}")
    print(f"Min / Max      : {min_effect:.2f} / {max_effect:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # 设置随机种子以便复现（可选）
    random.seed(42)
    np.random.seed(42)

    evaluate_random_policy()