# src/environment/amta_env.py
import numpy as np
from config import EnvConfig, DQNConfig
from src.environment.scenario_gen import ScenarioGenerator
from src.utils.reward_calc import RewardCalculator


class AMTAEnv:
    """
    Missile Allocation Environment.
    Simulates the process shown in Fig. 2.
    """

    def __init__(self):
        self.generator = ScenarioGenerator()
        self.missiles = []
        self.targets = []
        self.missile_map = {}  # ID -> Object

        self.current_step_idx = 0  # 当前轮到哪一枚导弹决策
        self.total_missile_cost = 0.0

        # 为了 One-hot 编码预存一些索引
        self.m_type_to_idx = {t: i for i, t in enumerate(EnvConfig.MISSILE_TYPE_ORDER)}
        self.t_type_to_idx = {t: i for i, t in enumerate(EnvConfig.TARGET_TYPE_ORDER)}

    def reset(self, n_targets=None, rho=None):
        """初始化一个新的 Episode"""
        # 默认使用 N9 rho3 (论文常用训练规模)
        N = n_targets if n_targets else 9
        R = rho if rho else 3.0

        self.missiles, self.targets = self.generator.generate_instance(N, R)
        self.missile_map = {m.m_id: m for m in self.missiles}

        # 重置动态状态
        for t in self.targets:
            t.reset()

        self.current_step_idx = 0
        self.total_missile_cost = sum(m.cost for m in self.missiles)

        return self._get_observation()

    def step(self, target_idx):
        """
        执行动作：将当前导弹分配给 target_idx
        :param target_idx: int, 目标在 self.targets 中的索引
        """
        if self.current_step_idx >= len(self.missiles):
            raise Exception("Episode already finished")

        current_missile = self.missiles[self.current_step_idx]
        selected_target = self.targets[target_idx]

        # 1. 计算 Local Reward (Marginal Return) [cite: 282, 388]
        # 必须在更新状态之前计算
        r_local = RewardCalculator.calculate_marginal_return(
            current_missile, selected_target, self.targets, self.missile_map
        )

        # 2. 更新状态 (Execute Assignment)
        selected_target.assigned_missile_ids.append(current_missile.m_id)
        selected_target.current_missile_cost += current_missile.cost
        current_missile.target_id = selected_target.t_id

        # 3. 移动到下一枚导弹
        self.current_step_idx += 1
        done = (self.current_step_idx >= len(self.missiles))

        # 4. 如果结束，计算 Global Reward (用于稍后回填，这里先返回 None 或 0)
        info = {}
        if done:
            total_effectiveness = RewardCalculator.calculate_total_effectiveness(
                self.targets, self.missile_map
            )
            # Global reward is the final effectiveness averaged by missile count (Eq 14) [cite: 298]
            # 注意：论文公式是 E(X)/M。
            r_global = total_effectiveness / len(self.missiles)
            info['global_reward'] = r_global
            info['total_effectiveness'] = total_effectiveness

        next_obs = self._get_observation() if not done else None

        return next_obs, r_local, done, info

    def _get_observation(self):
        """
        Construct State 's' and Action 'a' vectors.
        Reference: Section III-B [cite: 227-257]
        """
        current_missile = self.missiles[self.current_step_idx]

        # --- 1. Construct Internal State 's' [cite: 227-251] ---
        # s = (o_i, d_i, g_i^m)

        # o_i: Remaining missile resources (ratio)
        # 计算剩余未分配导弹的 Cost 总和
        remaining_cost = sum(m.cost for m in self.missiles[self.current_step_idx:])
        o_i = [remaining_cost / self.total_missile_cost]

        # d_i: Current assignment situation (normalized vector per target type)
        # 论文示例：如果有2种目标，T1被分配了Cost 1，T2被分配了Cost 2，总Cost 5
        # d_i = [1/5, 2/5]
        # 这里我们需要按 Target Type 聚合当前已分配的 Cost
        type_cost_map = {t: 0.0 for t in EnvConfig.TARGET_TYPE_ORDER}
        for t in self.targets:
            type_cost_map[t.t_type] += t.current_missile_cost

        d_i = []
        for t_type in EnvConfig.TARGET_TYPE_ORDER:
            d_i.append(type_cost_map[t_type] / self.total_missile_cost)

        # g_i^m: Current missile type (one-hot)
        g_i_m = [0] * len(EnvConfig.MISSILE_TYPES)
        g_i_m[self.m_type_to_idx[current_missile.m_type]] = 1

        state_vec = np.array(o_i + d_i + g_i_m, dtype=np.float32)

        # --- 2. Construct Action Vectors 'a' for all targets [cite: 252-257] ---
        # For each target w_j, a_j = (n_j, v_j, g_j^w)
        # n_j: cost of missiles targeting w_j (论文中似乎是按导弹类型区分的向量？)
        # 论文原文[cite: 254]: "n_j \in R^I_m represents the cost of missiles that have targeted w_j"
        # 举例: n_j=(1, 4) 表示由1个M1(cost1)和2个M2(cost2*2=4)组成。所以这是一个向量，维度等于导弹类型数。

        action_vecs = []
        for t in self.targets:
            # n_j calculation
            n_j = [0.0] * len(EnvConfig.MISSILE_TYPES)
            for m_id in t.assigned_missile_ids:
                m = self.missile_map[m_id]
                idx = self.m_type_to_idx[m.m_type]
                n_j[idx] += m.cost

            # # [修改] 使用 EnvConfig.MAX_MISSILE_COST_SUM
            # n_j = [x / EnvConfig.MAX_MISSILE_COST_SUM for x in n_j]

            # # [修改] 使用 EnvConfig.MAX_TARGET_VALUE
            # v_j = [t.value / EnvConfig.MAX_TARGET_VALUE]
            v_j = [t.value]

            # g_j^w: target type (one-hot)
            g_j_w = [0] * len(EnvConfig.TARGET_TYPES)
            g_j_w[self.t_type_to_idx[t.t_type]] = 1

            # Concatenate for this target
            a_vec = np.array(n_j + v_j + g_j_w, dtype=np.float32)
            action_vecs.append(a_vec)

        return state_vec, np.array(action_vecs)