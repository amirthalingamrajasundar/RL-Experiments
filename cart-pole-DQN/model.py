"""
Neural network architecture shared across all DQN variants.
Uses PyTorch to define a simple fully connected (dense) network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    """
    Deep Q-Network with two hidden layers.
    Approximates the Q-value function Q(s, a).
    Input: State vector.
    Output: Q-values for every possible action.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNetwork, self).__init__()
        # First hidden layer: maps state to hidden dimension
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Second hidden layer: adds non-linearity and depth
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer: maps hidden representation to Q-values for each action
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Apply Rectified Linear Unit (ReLU) activation to hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No activation on the output layer (Q-values can be positive or negative)
        return self.fc3(x)


def create_network(state_dim, action_dim, hidden_dim=128):
    """Helper factory function to instantiate a Q-network."""
    return DQNetwork(state_dim, action_dim, hidden_dim)