# src/agent/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DQNConfig


class PODRLNetwork(nn.Module):
    """
    DQN Architecture for AMTA.
    Reference: Fig. 3(b) & Table I.
    Input: Concatenated vector of State (s) and Action (a).
    Output: Single Q-value Q(s, a).
    Structure: Input(14) -> FC(64) -> FC(128) -> FC(64) -> Output(1)
    Feature: Skip connection from Hidden Layer 1 output to Hidden Layer 3 input.
    """

    def __init__(self, input_dim, output_dim=1):
        super(PODRLNetwork, self).__init__()

        hidden_dims = DQNConfig.HIDDEN_LAYERS  # [64, 128, 64]

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])

        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.fc3 = nn.Linear(hidden_dims[1] + hidden_dims[0], hidden_dims[2])

        # Output Layer
        self.out = nn.Linear(hidden_dims[2], output_dim)

        self.slope = DQNConfig.ACTIVATION_SLOPE

    def forward(self, state, action):
        """
        :param state: Tensor [Batch, State_Dim]
        :param action: Tensor [Batch, Action_Dim]
        """
        # Concatenate s and a
        x = torch.cat([state, action], dim=1)

        # Hidden 1
        x1 = F.leaky_relu(self.fc1(x), negative_slope=self.slope)

        # Hidden 2
        x2 = F.leaky_relu(self.fc2(x1), negative_slope=self.slope)

        # Skip Connection: Concatenate x1 and x2 before feeding to Layer 3
        # This preserves information from Layer 1 directly to Layer 3
        x_skip = torch.cat([x2, x1], dim=1)

        # Hidden 3
        x3 = F.leaky_relu(self.fc3(x_skip), negative_slope=self.slope)

        # Output Q-value
        q_val = self.out(x3)

        return q_val