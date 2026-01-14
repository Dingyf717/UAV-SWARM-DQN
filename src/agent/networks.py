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

        # Layer 3 (Skip connection target)
        # Input to fc3 is: Output of fc2 + Output of fc1 (Skip Connection)
        # Note: Depending on implementation, skip connection usually implies element-wise sum.
        # However, layer 1 is 64 dim, layer 2 output is 128 dim. They cannot be summed directly.
        # Paper citation [30] (ResNet) usually implies same dimension or projection.
        # Looking at Fig 3(a/b) arrows: The arrow goes from Layer 1 Output -> Layer 3 Input.
        # Layer 3 input size must match the combined features.
        # Option A: Concatenation (64+128).
        # Option B: ResNet style add (requires projection).
        # Given "Skip connection" usually means addition in DL, but dim mismatch exists (64 vs 128).
        # However, if we look at Table I structure: (64, 128, 64).
        # Most likely implementation for simple vectors: Concatenation or Linear Projection.
        # Let's assume Concatenation for safety unless dimensions match.
        # BUT, standard ResNet adds. Let's look closer at Fig 3.
        # The arrow bypasses the middle layer.
        # Let's try to project Layer 1 to match Layer 2 output if we want to add,
        # OR simply feed Layer 1 output INTO Layer 3 alongside Layer 2 output.
        # Let's assume the latter: FC3 input = FC2_out (128) + FC1_out (64) = 192 dims?
        # No, Table I says "neurons of hidden layers: (64, 128, 64)". This refers to OUTPUT size usually.
        # Let's implement a Linear projection for the skip connection to allow addition,
        # OR just concatenate. Concatenation is standard in older RL papers.
        # Let's use Concatenation: Input to L3 = L2_out (128) + L1_out (64).

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