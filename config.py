# config.py

"""
Global Configuration for AMTA-PODRL Project.
Based on Luo et al. (2022) IEEE Transactions on Systems, Man, and Cybernetics: Systems.
"""

import torch


# ==============================================================================
# 1. DQN Hyperparameters (Table I & Section V-B-2)
# ==============================================================================
class DQNConfig:
    # 基础训练参数
    LEARNING_RATE = 1e-4  #
    GAMMA = 0.95  # Discount factor
    BATCH_SIZE = 120  #
    NUM_EPISODES = 2000  #

    # 探索策略 (Epsilon-Greedy)
    EPSILON_START = 0.0  # 初始探索率
    EPSILON_INCREMENT = 0.005  # 每回合增加值
    EPSILON_MAX = 0.9  # 最大探索率 (意味着此时只有10%随机)

    # 网络更新参数
    SOFT_UPDATE_FACTOR = 0.99  # Beta in paper (Table I), target_net update
    # 注意：论文写 beta=0.99，实际代码通常用 tau = 1-beta = 0.01

    # 网络结构 (Table I & Fig. 3b)
    HIDDEN_LAYERS = [64, 128, 64]  #
    USE_SKIP_CONNECTION = True  # Skip connection from hidden layer 1 to 3
    ACTIVATION_SLOPE = 0.1  # Leaky ReLU slope

    # 经验回放池 (Table I)
    # 论文中为每种目标类型设置了不同大小的子存储池，单位是 10^4
    # T1, T2, T3, T4 对应的容量
    SUB_MEMORY_CAPACITIES = {
        'T1': 30000,
        'T2': 20000,
        'T3': 20000,
        'T4': 10000
    }  #



    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# 2. Environment & Entity Parameters (Section V-B-1)
# ==============================================================================
class EnvConfig:
    # 奖励函数平衡因子
    # r = alpha * r_local + (1 - alpha) * r_global
    REWARD_ALPHA = 0.5  #

    # 导弹属性定义 (Section V-B-1: Generation of Missiles)
    MISSILE_TYPES = {
        'M1': {
            'cost': 1.0,
            'dp': 6.0,  # Destructive Payload
            'penetration_prob': 0.35  # 普通导弹突防概率低
        },
        'M2': {
            'cost': 1.25,
            'dp': 6.0,
            'penetration_prob': 0.70  # 特殊导弹突防概率高
        }
    }

    # 目标属性定义 (Section V-B-1: Generation of Targets)
    TARGET_TYPES = {
        'T1': {'value': 4.0, 'health': 1.0},
        'T2': {'value': 6.0, 'health': 2.0},
        'T3': {'value': 8.0, 'health': 4.0},
        'T4': {'value': 16.0, 'health': 8.0}
    }

    # 目标生成规则索引 (用于One-hot编码顺序)
    TARGET_TYPE_ORDER = ['T1', 'T2', 'T3', 'T4']
    MISSILE_TYPE_ORDER = ['M1', 'M2']

    # [新增] 归一化常数 (Normalization Constants)
    # 估算依据：
    # MAX_MISSILE_COST_SUM: 假设一个目标最多被20枚导弹攻击 (20 * 1.0)
    # MAX_TARGET_VALUE: T4 目标的最大价值是 16.0
    MAX_MISSILE_COST_SUM = 25.0
    MAX_TARGET_VALUE = 16.0