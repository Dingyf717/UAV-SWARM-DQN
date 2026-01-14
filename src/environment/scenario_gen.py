# src/environment/scenario_gen.py
import random
import math
import numpy as np
from config import EnvConfig
from src.environment.entities import Missile, Target


class ScenarioGenerator:
    """
    Generates AMTA instances based on problem scale (N, rho).
    Reference: Section V-B-1 "Data Generator"
    """

    def __init__(self):
        self.m_types = EnvConfig.MISSILE_TYPES
        self.t_types = EnvConfig.TARGET_TYPES

    def generate_instance(self, n_targets, rho):
        """
        生成一个包含 missiles 和 targets 的实例
        :param n_targets: 目标数量 (N)
        :param rho: 导弹-目标比例 (Missile-Target Ratio)
        :return: (missile_list, target_list)
        """
        # -------------------------------------------------
        # 1. 生成目标 (Generation of Targets)
        # -------------------------------------------------
        # T1: N/2
        # T4: 1
        # T2: Random number eta between [0, N/2 - 1]
        # T3: Remainder (N/2 - eta - 1)

        n_t1 = int(n_targets / 2)
        n_t4 = 1

        # 确保剩余空间足够分配 T2 和 T3
        # Upper bound for T2 is N/2 - 1
        upper_bound_t2 = max(0, int(n_targets / 2) - 1)
        eta = random.randint(0, upper_bound_t2)
        n_t2 = eta

        n_t3 = n_targets - n_t1 - n_t4 - n_t2

        # 防止计算误差导致的负数 (虽然逻辑上不应该出现)
        if n_t3 < 0: n_t3 = 0

        target_list = []
        t_id_counter = 0

        # 辅助函数：批量创建目标
        def add_targets(count, t_type_key):
            nonlocal t_id_counter
            params = self.t_types[t_type_key]
            for _ in range(count):
                target_list.append(Target(
                    t_id=t_id_counter,
                    t_type=t_type_key,
                    value=params['value'],
                    health=params['health']
                ))
                t_id_counter += 1

        # 按 T1, T2, T3, T4 顺序添加 (顺序不影响逻辑，但方便调试)
        add_targets(n_t1, 'T1')
        add_targets(n_t2, 'T2')
        add_targets(n_t3, 'T3')
        add_targets(n_t4, 'T4')

        # -------------------------------------------------
        # 2. 生成导弹 (Generation of Missiles)
        # -------------------------------------------------
        # Total M = N * rho
        # M1: 2/3 * M
        # M2: 1/3 * M

        n_missiles = int(n_targets * rho)
        n_m1 = int(n_missiles * (2 / 3))
        n_m2 = n_missiles - n_m1  # 剩余的给 M2，确保总数匹配

        missile_list = []
        m_id_counter = 0

        def add_missiles(count, m_type_key):
            nonlocal m_id_counter
            params = self.m_types[m_type_key]
            for _ in range(count):
                missile_list.append(Missile(
                    m_id=m_id_counter,
                    m_type=m_type_key,
                    cost=params['cost'],
                    dp=params['dp'],
                    penetration_prob=params['penetration_prob']
                ))
                m_id_counter += 1

        add_missiles(n_m1, 'M1')
        add_missiles(n_m2, 'M2')

        # 打乱导弹顺序，模拟真实队列
        random.shuffle(missile_list)

        return missile_list, target_list