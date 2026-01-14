# src/utils/reward_calc.py
from config import EnvConfig


class RewardCalculator:
    """
    Handles calculation of Kill Probabilities, Combat Effectiveness, and Rewards.
    Formulas based on Section II-A and IV-A.
    """

    @staticmethod
    def calculate_single_kill_prob(missile, target):
        """
        Formula (2): p_ij = dp_i / (dp_i + h_j)
        """
        return missile.dp / (missile.dp + target.health)

    @staticmethod
    def calculate_joint_kill_prob(target, assigned_missiles):
        """
        Formula (3): p_bar_j = 1 - product(1 - p_ij)
        """
        if not assigned_missiles:
            return 0.0

        survival_prob = 1.0
        for m in assigned_missiles:
            p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
            survival_prob *= (1.0 - p_ij)

        return 1.0 - survival_prob

    @staticmethod
    def calculate_total_effectiveness(targets, all_missiles_map):
        """
        Formula (4): E(X) = sum(p_bar_j * v_j)

        :param targets: List of Target objects
        :param all_missiles_map: Dict {missile_id: Missile_Object} 方便查找
        """
        total_e = 0.0
        for t in targets:
            # 获取分配给该目标的所有导弹对象
            assigned_m_objs = [all_missiles_map[m_id] for m_id in t.assigned_missile_ids]
            p_bar = RewardCalculator.calculate_joint_kill_prob(t, assigned_m_objs)
            total_e += p_bar * t.value
        return total_e

    @staticmethod
    def calculate_marginal_return(current_missile, target, targets, all_missiles_map):
        """
        Formula (13): MR_ij = E(X_new) - E(X_old)
        计算将当前导弹分配给指定目标带来的效能增量。
        """
        # 1. 计算旧的联合毁伤概率 (Before assignment)
        assigned_m_objs = [all_missiles_map[m_id] for m_id in target.assigned_missile_ids]
        p_bar_old = RewardCalculator.calculate_joint_kill_prob(target, assigned_m_objs)
        e_old_part = p_bar_old * target.value

        # 2. 计算新的联合毁伤概率 (After assignment)
        # 临时加入当前导弹进行计算
        p_bar_new = RewardCalculator.calculate_joint_kill_prob(target, assigned_m_objs + [current_missile])
        e_new_part = p_bar_new * target.value

        # 3. 边际收益 = 新部分效能 - 旧部分效能
        # 因为其他目标的效能没变，所以只需要计算这一个目标的差值
        mr = e_new_part - e_old_part
        return mr