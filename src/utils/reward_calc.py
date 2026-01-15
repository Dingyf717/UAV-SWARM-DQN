# # src/utils/reward_calc.py
# import random
# import numpy as np
# from config import EnvConfig
#
#
# class RewardCalculator:
#     """
#     Handles calculation of Kill Probabilities, Combat Effectiveness, and Rewards.
#     Updated to include Adversarial Environment logic (Penetration Probability).
#     """
#
#     @staticmethod
#     def calculate_single_kill_prob(missile, target):
#         """
#         Formula (2): p_ij = dp_i / (dp_i + h_j)
#         Basic kill probability assuming the missile successfully hits (penetrates).
#         """
#         return missile.dp / (missile.dp + target.health)
#
#     @staticmethod
#     def calculate_joint_kill_prob(target, assigned_missiles):
#         """
#         Formula (3): p_bar_j = 1 - product(1 - p_ij)
#         Standard joint kill probability for a set of missiles (assuming they all hit).
#         Used by:
#           1. Monte Carlo simulation (after filtering intercepted missiles).
#           2. Legacy/Visualization calls.
#         """
#         if not assigned_missiles:
#             return 0.0
#
#         survival_prob = 1.0
#         for m in assigned_missiles:
#             p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
#             survival_prob *= (1.0 - p_ij)
#
#         return 1.0 - survival_prob
#
#     @staticmethod
#     def _calculate_expected_joint_prob(target, assigned_missiles):
#         """
#         Helper: Calculates Expected Joint Kill Probability.
#         Uses E[p_effective] = p_ij * penetration_prob.
#         Used for Local Reward to provide stable gradients favoring high-penetration missiles.
#         """
#         if not assigned_missiles:
#             return 0.0
#
#         survival_prob = 1.0
#         for m in assigned_missiles:
#             p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
#             # Expectation: Probability of kill = P(Penetrate) * P(Kill | Penetrate)
#             p_effective = p_ij * m.penetration_prob
#             survival_prob *= (1.0 - p_effective)
#
#         return 1.0 - survival_prob
#
#     @staticmethod
#     def calculate_total_effectiveness(targets, all_missiles_map):
#         """
#         Formula (8) & (14): Global Reward in Adversarial Environment.
#
#         [CRITICAL FIX]: Incorporates 'penetration_prob' via Monte Carlo simulation.
#         The global reward reflects the ACTUAL outcome in the stochastic environment.
#         """
#         total_e = 0.0
#         for t in targets:
#             # 1. Get all assigned missiles
#             assigned_m_objs = [all_missiles_map[m_id] for m_id in t.assigned_missile_ids]
#
#             # 2. Simulate Interception (Monte Carlo)
#             # Only surviving missiles contribute to damage.
#             # Roll random die against each missile's penetration probability.
#             surviving_missiles = []
#             for m in assigned_m_objs:
#                 # Sample random number xi ~ U[0,1]
#                 # If xi <= penetration_prob, missile penetrates.
#                 if random.random() <= m.penetration_prob:
#                     surviving_missiles.append(m)
#
#             # 3. Calculate Joint Kill Prob using ONLY survivors
#             # Note: We use the standard formula here because we've already
#             # determined which missiles physically hit the target.
#             p_bar = RewardCalculator.calculate_joint_kill_prob(t, surviving_missiles)
#             total_e += p_bar * t.value
#
#         return total_e
#
#     @staticmethod
#     def calculate_marginal_return(current_missile, target, targets, all_missiles_map):
#         """
#         Formula (13): MR_ij = E(X_new) - E(X_old)
#
#         [CRITICAL FIX]: Incorporates 'penetration_prob' via Expectation.
#         Uses expected effectiveness (p_ij * penetration_prob) to distinguish
#         between missiles with different penetration capabilities (e.g., M1 vs M2).
#         """
#         # Get currently assigned missiles
#         assigned_m_objs = [all_missiles_map[m_id] for m_id in target.assigned_missile_ids]
#
#         # 1. Calculate Old Expected Effectiveness (Before assignment)
#         p_bar_old = RewardCalculator._calculate_expected_joint_prob(target, assigned_m_objs)
#         e_old_part = p_bar_old * target.value
#
#         # 2. Calculate New Expected Effectiveness (After assignment)
#         # Temporarily add current missile
#         p_bar_new = RewardCalculator._calculate_expected_joint_prob(target, assigned_m_objs + [current_missile])
#         e_new_part = p_bar_new * target.value
#
#         # 3. Marginal Return
#         mr = e_new_part - e_old_part
#         return mr

# src/utils/reward_calc.py
import random
import numpy as np
from config import EnvConfig


class RewardCalculator:
    """
    Handles calculation of Kill Probabilities, Combat Effectiveness, and Rewards.
    Updated to correctly implement the Adversarial Environment logic (Joint Penetration Probability).
    """

    @staticmethod
    def calculate_single_kill_prob(missile, target):
        """
        Formula (2): p_ij = dp_i / (dp_i + h_j)
        Basic kill probability assuming the missile successfully hits (penetrates).
        """
        return missile.dp / (missile.dp + target.health)

    @staticmethod
    def calculate_joint_penetration_prob(missiles):
        """
        Formula (7): tau_bar_j = 1 - product(1 - tau_ij)
        Calculates the joint penetration probability for a group of missiles.
        This shared probability is used for ALL missiles in the group[cite: 158].
        """
        if not missiles:
            return 0.0

        intercept_prob_product = 1.0
        for m in missiles:
            # 1 - tau is the probability of being intercepted individually
            intercept_prob_product *= (1.0 - m.penetration_prob)

        # The probability that AT LEAST ONE helps the group penetrate (Joint Threshold)
        return 1.0 - intercept_prob_product

    @staticmethod
    def calculate_joint_kill_prob(target, surviving_missiles):
        """
        Formula (3): p_bar_j = 1 - product(1 - p_ij)
        Standard joint kill probability for a set of missiles (assuming they all hit).
        """
        if not surviving_missiles:
            return 0.0

        survival_prob = 1.0
        for m in surviving_missiles:
            p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
            survival_prob *= (1.0 - p_ij)

        return 1.0 - survival_prob

    @staticmethod
    def _calculate_expected_effectiveness_with_saturation(target, missiles):
        """
        Calculates the Expected Combat Effectiveness (Value * E[Kill Probability])
        considering the SATURATION ATTACK mechanism.

        Mechanism Update:
        1. Calculate Joint Penetration Prob (tau_bar) for the group.
        2. In expectation, each missile now has an effective penetration prob of tau_bar.
        3. E[Damage] = Value * (1 - product(1 - p_ij * tau_bar))
        """
        if not missiles:
            return 0.0

        # 1. Calculate Joint Penetration Probability (Shared by all)
        tau_bar = RewardCalculator.calculate_joint_penetration_prob(missiles)

        # 2. Calculate Expected Target Survival Probability
        # Because survival events are Bernoulli trials with p=tau_bar,
        # the expected survival term for each missile is (1 - p_kill * tau_bar).
        target_survival_prob = 1.0
        for m in missiles:
            p_ij = RewardCalculator.calculate_single_kill_prob(m, target)
            # Effective kill probability includes the BOOSTED penetration probability
            p_effective = p_ij * tau_bar
            target_survival_prob *= (1.0 - p_effective)

        expected_kill_prob = 1.0 - target_survival_prob
        return expected_kill_prob * target.value

    @staticmethod
    def calculate_total_effectiveness(targets, all_missiles_map):
        """
        Formula (8) & (14): Global Reward in Adversarial Environment.

        [CORRECTED]: Implements the "Shared Penetration Probability" mechanism.
        """
        total_e = 0.0
        for t in targets:
            # 1. Get all assigned missiles
            assigned_m_objs = [all_missiles_map[m_id] for m_id in t.assigned_missile_ids]

            if not assigned_m_objs:
                continue

            # 2. Calculate Joint Penetration Probability (Shared Threshold)
            # [cite: 158] tau_bar_j is shared by all missiles in K_j
            joint_tau = RewardCalculator.calculate_joint_penetration_prob(assigned_m_objs)

            # 3. Simulate Interception (Monte Carlo)
            # [cite: 160] "sample a random number xi... if xi > tau_bar_j [intercepted]"
            surviving_missiles = []
            for m in assigned_m_objs:
                # IMPORTANT: Use joint_tau, not m.penetration_prob
                # This ensures the saturation attack benefit is realized.
                if random.random() <= joint_tau:
                    surviving_missiles.append(m)

            # 4. Calculate Joint Kill Prob using survivors
            p_bar = RewardCalculator.calculate_joint_kill_prob(t, surviving_missiles)
            total_e += p_bar * t.value

        return total_e

    @staticmethod
    def calculate_marginal_return(current_missile, target, targets, all_missiles_map):
        """
        Formula (13): MR_ij = E(X_new) - E(X_old)

        [CORRECTED]: Calculates expected marginal return considering that adding a missile
        boosts the penetration probability of ALL missiles assigned to that target.
        """
        # Get currently assigned missiles (Old Set)
        assigned_m_objs = [all_missiles_map[m_id] for m_id in target.assigned_missile_ids]

        # 1. Calculate Old Expected Effectiveness
        # Uses the joint penetration probability of the OLD group
        e_old = RewardCalculator._calculate_expected_effectiveness_with_saturation(target, assigned_m_objs)

        # 2. Calculate New Expected Effectiveness
        # Uses the joint penetration probability of the NEW group (Old + Current)
        # The new tau_bar will be higher, increasing the effectiveness of OLD missiles too.
        e_new = RewardCalculator._calculate_expected_effectiveness_with_saturation(target,
                                                                                   assigned_m_objs + [current_missile])

        # 3. Marginal Return
        mr = e_new - e_old
        return mr