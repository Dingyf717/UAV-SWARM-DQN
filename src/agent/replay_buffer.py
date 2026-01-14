# src/agent/replay_buffer.py
import numpy as np
import torch
from collections import deque
from config import DQNConfig, EnvConfig


class FairReplayBuffer:
    """
    Replay Memory with Fair Sample Strategy.
    Now supports storing 'next_action_matrix' for correct Q-target calculation.
    """

    def __init__(self):
        self.device = DQNConfig.DEVICE
        self.sub_memories = {}
        for t_type, cap in DQNConfig.SUB_MEMORY_CAPACITIES.items():
            self.sub_memories[t_type] = deque(maxlen=cap)

        self.t_types = EnvConfig.TARGET_TYPE_ORDER

    def push(self, state, action, reward, next_state, next_action_matrix, done, target_type):
        """
        Store transition including next_action_matrix.
        :param next_action_matrix: Numpy array (N, Action_Dim) or None if done
        """
        # 存储 6 元组
        transition = (state, action, reward, next_state, next_action_matrix, done)

        if target_type in self.sub_memories:
            self.sub_memories[target_type].append(transition)

    def sample(self, batch_size):
        num_types = len(self.t_types)
        samples_per_type = batch_size // num_types

        batch_s, batch_a, batch_r, batch_ns, batch_na, batch_d = [], [], [], [], [], []

        for t_type in self.t_types:
            memory = self.sub_memories[t_type]
            current_len = len(memory)

            if current_len < samples_per_type:
                indices = np.random.choice(current_len, samples_per_type, replace=True)
            else:
                indices = np.random.choice(current_len, samples_per_type, replace=False)

            for idx in indices:
                s, a, r, ns, na, d = memory[idx]
                batch_s.append(s)
                batch_a.append(a)
                batch_r.append(r)

                if ns is None:
                    # 如果 Episode 结束，补零以保持维度一致
                    batch_ns.append(np.zeros_like(s))
                    # 补一个假的动作矩阵 (1, Action_Dim) 防止报错
                    batch_na.append(np.zeros((1, a.shape[0])))
                else:
                    batch_ns.append(ns)
                    batch_na.append(na)

                batch_d.append(float(d))

        # 注意：batch_na 是 List[np.array]，因为每个样本的 Action Matrix 可能不一样（虽然本实验固定）
        return (
            torch.FloatTensor(np.array(batch_s)).to(self.device),
            torch.FloatTensor(np.array(batch_a)).to(self.device),
            torch.FloatTensor(np.array(batch_r)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(batch_ns)).to(self.device),
            batch_na,  # 返回 List，不在 GPU 上
            torch.FloatTensor(np.array(batch_d)).unsqueeze(1).to(self.device)
        )

    def is_ready(self, batch_size):
        samples_per_type = batch_size // len(self.t_types)
        for t_type in self.t_types:
            if len(self.sub_memories[t_type]) < samples_per_type:
                return False
        return True

    def __len__(self):
        return sum(len(m) for m in self.sub_memories.values())