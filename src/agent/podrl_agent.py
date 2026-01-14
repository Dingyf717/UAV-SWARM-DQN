# src/agent/podrl_agent.py
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import DQNConfig
from src.agent.networks import PODRLNetwork
from src.agent.replay_buffer import FairReplayBuffer


class PODRLAgent:
    """
    Agent implementing the PODRL algorithm.
    """

    def __init__(self, state_dim, action_dim):
        self.device = DQNConfig.DEVICE
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim

        # 初始化网络 (Eval & Target)
        self.eval_net = PODRLNetwork(self.input_dim).to(self.device)
        self.target_net = PODRLNetwork(self.input_dim).to(self.device)

        # 复制参数
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=DQNConfig.LEARNING_RATE)
        self.memory = FairReplayBuffer()

        # 探索参数
        self.epsilon = DQNConfig.EPSILON_START
        self.epsilon_max = DQNConfig.EPSILON_MAX
        self.epsilon_inc = DQNConfig.EPSILON_INCREMENT

    def select_action(self, state_vec, action_matrix, training=True):
        """
        Input:
            state_vec: (State_Dim,)
            action_matrix: (N_Targets, Action_Dim)
        Output:
            action_index (int)
        """
        n_targets = action_matrix.shape[0]

        # Epsilon-Greedy: epsilon 是 exploitation rate (利用率)
        # 如果 random > epsilon，则随机探索
        if training and random.random() > self.epsilon:
            return random.randint(0, n_targets - 1)

        # 否则贪婪选择 argmax Q(s, a)
        with torch.no_grad():
            # 将 s 重复 N 次，拼成 (N, State_Dim)
            s_tensor = torch.FloatTensor(state_vec).to(self.device).unsqueeze(0).repeat(n_targets, 1)
            a_tensor = torch.FloatTensor(action_matrix).to(self.device)

            # 计算所有目标的 Q 值
            q_values = self.eval_net(s_tensor, a_tensor)  # (N, 1)
            action_idx = q_values.argmax().item()

        return action_idx

    def update(self):
        """执行一步梯度下降"""
        if not self.memory.is_ready(DQNConfig.BATCH_SIZE):
            return None

        # 1. 采样
        states, actions, rewards, next_states, next_action_matrices, dones = self.memory.sample(DQNConfig.BATCH_SIZE)

        # 2. 计算 Target Q
        with torch.no_grad():
            next_q_values = []

            # 由于 next_action_matrices 是变长列表，必须循环处理
            for i in range(DQNConfig.BATCH_SIZE):
                if dones[i].item() > 0.5:
                    next_q_values.append(0.0)
                else:
                    # 取出第 i 个样本的 next_state (1, S_dim)
                    ns = next_states[i].unsqueeze(0)
                    # 取出第 i 个样本的 next_action_matrix (N, A_dim) -> 转 Tensor
                    na_mat = torch.FloatTensor(next_action_matrices[i]).to(self.device)

                    # 扩展 ns 以匹配目标数量
                    ns_expanded = ns.repeat(na_mat.shape[0], 1)

                    # --- Double DQN 逻辑 ---
                    # A. 用 Eval Net 选动作: argmax Q_eval(s', a')
                    q_eval_next = self.eval_net(ns_expanded, na_mat)
                    best_action_idx = q_eval_next.argmax()

                    # B. 用 Target Net 估值: Q_target(s', best_a')
                    # 取出最佳动作向量 (1, A_dim)
                    best_a_vec = na_mat[best_action_idx].unsqueeze(0)
                    q_target_next = self.target_net(ns, best_a_vec)

                    next_q_values.append(q_target_next.item())

            # 拼回 Tensor (Batch, 1)
            next_q_values = torch.FloatTensor(next_q_values).unsqueeze(1).to(self.device)
            target_q = rewards + DQNConfig.GAMMA * next_q_values

        # 3. 计算 Current Q
        current_q = self.eval_net(states, actions)

        # 4. 反向传播
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # [诊断点 1]: 计算梯度模长 (Gradient Norm)
        # 这能告诉我们梯度是否过大（导致震荡）或为0（导致不学习）
        total_norm = 0.0
        for p in self.eval_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()

        # 5. 软更新 Target Net
        self._soft_update(self.target_net, self.eval_net)

        # [诊断点 2]: 返回详细的统计字典
        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),  # 平均 Q 值
            "q_max": current_q.max().item(),  # 最大 Q 值
            "q_target_mean": target_q.mean().item(),  # 目标 Q 值均值
            "grad_norm": total_norm  # 梯度模长
        }

    def _soft_update(self, target_net, source_net):
        # beta 接近 1 (0.99)，意味着保留 99% 的旧参数，只更新 1%
        beta = DQNConfig.SOFT_UPDATE_FACTOR
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(beta * target_param.data + (1.0 - beta) * source_param.data)

    def update_epsilon(self):
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_inc