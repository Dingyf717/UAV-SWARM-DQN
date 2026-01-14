# src/utils/reward_calc.py
import random
import numpy as np
from config import EnvConfig


class RewardCalculator:
    """
    Handles calculation of Kill Probabilities, Combat Effectiveness, and Rewards.
    Updated to include Adversarial Environment logic (Penetration Probability).
    """

    @staticmethod
    def calculate_single_kill_prob(missile, target):
        """
        Formula (2): p_ij = dp_i / (dp_i + h_j)
        Basic kill probability assuming the missile successfully hits (penetrates).
        """
        return missile.dp / (missile.dp + target.health)

    @staticmethod
    def calculate_joint_kill_prob(target, assigned_missiles):
        """
        Formula (3): p_bar_j = 1 - product(1 - p_ij)
        Standard joint kill probability for a set of missiles (assuming they all hit).
        Used by:
          1. Monte Carlo simulation (after filtering intercepted missiles).
          2. Legacy/Visualization calls.
        """
        if not assigned_missiles:
            return 0.0

        survival_prob = 1.0
        for m in assigned_missiles:
            p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
            survival_prob *= (1.0 - p_ij)

        return 1.0 - survival_prob

    @staticmethod
    def _calculate_expected_joint_prob(target, assigned_missiles):
        """
        Helper: Calculates Expected Joint Kill Probability.
        Uses E[p_effective] = p_ij * penetration_prob.
        Used for Local Reward to provide stable gradients favoring high-penetration missiles.
        """
        if not assigned_missiles:
            return 0.0

        survival_prob = 1.0
        for m in assigned_missiles:
            p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
            # Expectation: Probability of kill = P(Penetrate) * P(Kill | Penetrate)
            p_effective = p_ij * m.penetration_prob
            survival_prob *= (1.0 - p_effective)

        return 1.0 - survival_prob

    @staticmethod
    def calculate_total_effectiveness(targets, all_missiles_map):
        """
        Formula (8) & (14): Global Reward in Adversarial Environment.

        [CRITICAL FIX]: Incorporates 'penetration_prob' via Monte Carlo simulation.
        The global reward reflects the ACTUAL outcome in the stochastic environment.
        """
        total_e = 0.0
        for t in targets:
            # 1. Get all assigned missiles
            assigned_m_objs = [all_missiles_map[m_id] for m_id in t.assigned_missile_ids]

            # 2. Simulate Interception (Monte Carlo)
            # Only surviving missiles contribute to damage.
            # Roll random die against each missile's penetration probability.
            surviving_missiles = []
            for m in assigned_m_objs:
                # Sample random number xi ~ U[0,1]
                # If xi <= penetration_prob, missile penetrates.
                if random.random() <= m.penetration_prob:
                    surviving_missiles.append(m)

            # 3. Calculate Joint Kill Prob using ONLY survivors
            # Note: We use the standard formula here because we've already
            # determined which missiles physically hit the target.
            p_bar = RewardCalculator.calculate_joint_kill_prob(t, surviving_missiles)
            total_e += p_bar * t.value

        return total_e

    @staticmethod
    def calculate_marginal_return(current_missile, target, targets, all_missiles_map):
        """
        Formula (13): MR_ij = E(X_new) - E(X_old)

        [CRITICAL FIX]: Incorporates 'penetration_prob' via Expectation.
        Uses expected effectiveness (p_ij * penetration_prob) to distinguish
        between missiles with different penetration capabilities (e.g., M1 vs M2).
        """
        # Get currently assigned missiles
        assigned_m_objs = [all_missiles_map[m_id] for m_id in target.assigned_missile_ids]

        # 1. Calculate Old Expected Effectiveness (Before assignment)
        p_bar_old = RewardCalculator._calculate_expected_joint_prob(target, assigned_m_objs)
        e_old_part = p_bar_old * target.value

        # 2. Calculate New Expected Effectiveness (After assignment)
        # Temporarily add current missile
        p_bar_new = RewardCalculator._calculate_expected_joint_prob(target, assigned_m_objs + [current_missile])
        e_new_part = p_bar_new * target.value

        # 3. Marginal Return
        mr = e_new_part - e_old_part
        return mr