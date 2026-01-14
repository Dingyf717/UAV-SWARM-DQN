# test_agent.py
import numpy as np
from src.agent.podrl_agent import PODRLAgent
from config import DQNConfig


def test_agent_logic():
    print("\n--- Testing PODRL Agent ---")

    # 1. 初始化
    state_dim = 7
    action_dim = 7
    agent = PODRLAgent(state_dim, action_dim)
    print("Agent Initialized.")

    # 2. 测试 Select Action
    dummy_s = np.random.randn(state_dim)
    dummy_a_mat = np.random.randn(9, action_dim)  # 假设有9个目标

    action = agent.select_action(dummy_s, dummy_a_mat, training=True)
    print(f"Selected Action Index: {action}")

    # 3. 填充 Buffer 以测试 Update
    print("Filling buffer for update test...")
    # 我们需要填够一个 Batch Size (120) / 4 = 30 per type
    # 稍微多填点确保够用
    for t_type in ['T1', 'T2', 'T3', 'T4']:
        for _ in range(35):
            s = np.random.randn(state_dim)
            a = np.random.randn(action_dim)
            r = 1.0
            ns = np.random.randn(state_dim)
            na_mat = np.random.randn(9, action_dim)  # 下一时刻也有9个目标
            d = False

            agent.memory.push(s, a, r, ns, na_mat, d, t_type)

    print(f"Buffer Size: {len(agent.memory)}")

    # 4. 测试 Update
    try:
        loss = agent.update()
        print(f"Update successful. Loss: {loss:.6f}")
    except Exception as e:
        print(f"Update failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent_logic()