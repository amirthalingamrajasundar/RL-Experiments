"""
Experience replay buffers: Uniform and Prioritized.

Replay buffers store agent experiences (transitions) to break temporal correlations
in the data and allow for multiple updates per experience, improving sample efficiency.
"""
import numpy as np
import random
from collections import deque
from utils.sumtree import SumTree


class ReplayBuffer:
    """
    Standard uniform sampling replay buffer.
    Stores transitions in a circular buffer (deque) and samples them uniformly randomly.
    """
    
    def __init__(self, capacity):
        # deque automatically handles removing old items when capacity is reached
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.
        
        Returns:
             tuple: Arrays of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch into separate arrays for each component
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        """Current number of elements in the buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    
    Samples important transitions more frequently. Importance is measured by the
    TD-error (temporal difference error).
    
    Key Concepts:
    - Alpha: Controls how much prioritization is used (0 = uniform, 1 = full prioritization).
    - Beta: Controls Importance Sampling (IS) weight correction (0 = no correction, 1 = full correction).
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, 
                 epsilon=1e-6, min_priority=0.01):
        """
        Initialize PER buffer.
        
        Args:
            capacity (int): Max number of experiences.
            alpha (float): Prioritization exponent.
            beta_start (float): Initial value of IS weight beta.
            beta_frames (int): Number of frames over which to anneal beta to 1.0.
            epsilon (float): Small constant added to TD error to ensure non-zero probability.
            min_priority (float): Minimum priority to prevent experience starvation.
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.min_priority = min_priority
        self.frame = 1  # Counter for beta annealing
        
        # Initial max priority (ensure new experiences are sampled at least once)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Add new experience with maximum priority."""
        data = (state, action, reward, next_state, done)
        # New experiences get max priority so they are likely to be sampled soon
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)
    
    def sample(self, batch_size):
        """
        Sample a batch based on priorities and calculate IS weights.
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones, is_weights, tree_indices)
        """
        batch = []
        idxs = []
        priorities = []
        # Divide the total priority range into 'batch_size' segments
        segment = self.tree.total() / batch_size
        
        # Anneal beta linearly from beta_start to 1.0
        # As training converges, we want full importance sampling correction (beta=1.0)
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        for i in range(batch_size):
            # Sample one uniform random value from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # Retrieve experience at this value
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        # --- Importance Sampling (IS) Weight Calculation ---
        # P(i) = priority_i / total_priority
        # IS Weight = (N * P(i))^(-beta)
        total_priority = self.tree.total()
        P = np.array(priorities) / total_priority
        weights = (self.tree.n_entries * P) ** (-self.beta)

        # Normalize weights by the max weight for stability (scales them to [0, 1])
        weights = weights / weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(weights), idxs)
    
    def update_priorities(self, idxs, td_errors):
        """
        Update priorities of sampled transitions based on new TD errors.
        Called after the agent performs a learning step.
        """
        for idx, td_error in zip(idxs, td_errors):
            # Priority = (|TD_error| + epsilon)^alpha
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha

            # We use a minimum priority to ensure "easy" experiences still have a small chance of being replayed
            priority = max(priority, self.min_priority)
            self.tree.update(idx, priority)
            # Track max priority for new experiences
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries