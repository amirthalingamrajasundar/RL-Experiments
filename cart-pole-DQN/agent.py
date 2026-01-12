"""
Base DQN Agent with configurable Double DQN and Prioritized Replay.
This class encapsulates the reinforcement learning logic.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from model import create_network
from utils.memory import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    DQN Agent capable of running:
    1. Standard DQN (Nature 2015 version with target network)
    2. Double DQN (DDQN)
    3. DQN with Prioritized Experience Replay (PER)
    4. DDQN with PER
    """
    
    def __init__(self, state_dim, action_dim, config, 
                 use_double_dqn=False, use_prioritized_replay=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        
        # Automatically detect if GPU is available for faster training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Networks ---
        # Policy Net: Used to select actions and currently being updated
        self.policy_net = create_network(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        # Target Net: A stable copy of policy net used to calculate target Q-values
        self.target_net = create_network(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync initially
        self.target_net.eval() # Set to evaluation mode (no gradient tracking needed)
        
        # --- Optimizer & Loss ---
        # Adam is a standard, efficient optimizer for deep learning
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        # We use MSE loss. 'reduction=none' keeps individual losses for PER weighting
        self.criterion = nn.MSELoss(reduction='none')
        
        # --- Replay Buffer Selection ---
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                config.BUFFER_SIZE,
                alpha=config.PER_ALPHA,
                beta_start=config.PER_BETA_START,
                beta_frames=config.PER_BETA_FRAMES,
                epsilon=config.PER_EPSILON,
                min_priority=config.PER_MIN_PRIORITY
            )
        else:
            self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # --- Exploration Parameters ---
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        
        # --- Hyperparameters ---
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        
        # Stats trackers
        self.steps = 0
        self.episodes = 0
    
    def select_action(self, state, training=True):
        """
        Select action using Epsilon-Greedy policy.
        - With probability epsilon: choose a random action (exploration).
        - Otherwise: choose the action with highest Q-value (exploitation).
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploitation: Pass state through network to get Q-values
        with torch.no_grad():
            # Convert state to tensor and add batch dimension [1, state_dim]
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            # Return index of the max Q-value
            return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Helper to store experience in the memory buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
    
    def update(self):
        """
        Perform one step of gradient descent optimization on the policy network.
        This is the core learning loop of DQN.
        """
        # Ensure we have enough data to sample a full batch
        if len(self.memory) < self.config.MIN_BUFFER_SIZE:
            return None
        
        # --- 1. Sample Batch ---
        if self.use_prioritized_replay:
            # PER returns extra data: importance sampling weights and tree indices
            states, actions, rewards, next_states, dones, weights, idxs = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            # For uniform sampling, all weights are 1.0
            weights = torch.ones(self.batch_size).to(self.device)
            idxs = None
        
        # Convert numpy arrays to PyTorch tensors and move to configured device (CPU/GPU)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # --- 2. Compute Current Q-values ---
        # Q(s, a) from policy net. gather selects the Q-values for the specific actions taken.
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # --- 3. Compute Target Q-values ---
        # Target = r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            if self.use_double_dqn:
                # --- Double DQN Logic ---
                # Decouple action selection from value estimation to reduce overestimation bias.
                # 1. Select best next action using POLICY net: argmax_a' Q_policy(s', a')
                # 2. Evaluate that action using TARGET net: Q_target(s', next_action)
                next_actions = self.policy_net(next_states).argmax(1)
                target_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # --- Standard DQN Logic ---
                # Use target net for both selection and evaluation: max_a' Q_target(s', a')
                next_actions = self.target_net(next_states).argmax(1)
                target_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Hint: If done is 1 (episode ended), future Q-value is 0.
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # --- 4. Compute Loss ---
        # Calculate element-wise loss (MSE between current_q and target_q)
        losses = self.criterion(current_q, target_q)
        # Weight the loss by importance sampling weights (for PER). 
        # For standard DQN, weights are all 1.0, so this just takes the mean.
        loss = (losses * weights).mean()
        
        # --- 5. Optimize ---
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()            # Backpropagate gradients
        # Clip gradients to prevent exploding gradients (improves stability)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()      # Update network weights
        
        # --- 6. Update Priorities (PER only) ---
        if self.use_prioritized_replay and idxs is not None:
            with torch.no_grad():
                # This ensures the new priorities are accurate for the next sample.
                # Recompute current Q-values using the updated policy network
                # After calculating new current Q-values, new TD errors are |target_q - current_q|
                updated_current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                td_errors = torch.abs(target_q - updated_current_q)
            # Update the SumTree with new priorities based on these TD errors
            self.memory.update_priorities(idxs, td_errors.cpu().numpy())
        
        return loss.item()
    
    def update_target_network(self):
        """Hard update: Copy all weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save entire agent state to a file."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        """Load agent state from a file."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episodes = checkpoint['episodes']
        self.steps = checkpoint['steps']
    
    def get_name(self):
        """Return a string identifier for the current agent configuration."""
        name = "DDQN" if self.use_double_dqn else "DQN"
        if self.use_prioritized_replay:
            name += "+PER"
        return name