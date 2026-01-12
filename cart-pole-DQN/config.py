"""
Configuration file for DQN variants.
All algorithms use the same base hyperparameters to ensure a fair comparison of their core mechanisms.
"""

# --- Environment Settings ---
ENV_NAME = "CartPole-v1"
SEED = 42  # Fixed seed for reproducibility

# --- Main Training Loop ---
NUM_EPISODES = 500    # Total number of episodes to train for
MAX_STEPS = 500       # Max steps per episode (prevents infinite loops if agent gets stuck)

# --- Neural Network Architecture ---
HIDDEN_DIM = 128      # Number of neurons in hidden layers

# --- Q-Learning Hyperparameters ---
LEARNING_RATE = 0.001 # Step size for gradient descent optimizer (Adam)
GAMMA = 0.99          # Discount factor for future rewards (closer to 1.0 = more long-term focus)
BATCH_SIZE = 64       # Number of transitions sampled from replay buffer per update step

# --- Exploration Strategy (Epsilon-Greedy) ---
EPSILON_START = 1.0   # Initial exploration rate (100% random actions)
EPSILON_END = 0.01    # Minimum exploration rate (1% random actions)
EPSILON_DECAY = 0.995 # Multiplicative decay factor per episode

# --- Experience Replay Buffer ---
BUFFER_SIZE = 10000    # Maximum number of transitions the buffer can hold
MIN_BUFFER_SIZE = 1000 # Minimum transitions needed before training starts

# --- Double DQN Specific ---
TARGET_UPDATE_FREQ = 10  # How often (in episodes) to copy policy net weights to target net

# --- Prioritized Experience Replay (PER) Specific ---
PER_ALPHA = 0.6         # How much prioritization to use (0.0=none, 1.0=full)
PER_BETA_START = 0.4    # Initial importance sampling weight (corrects bias from non-uniform sampling)
PER_BETA_FRAMES = NUM_EPISODES  # Number of episodes over which beta anneals to 1.0
PER_EPSILON = 1e-6      # Small constant added to TD error to avoid zero priority
PER_MIN_PRIORITY = 0.01 # Minimum priority to prevent experience starvation

# --- Logging and Saving ---
LOG_INTERVAL = 10       # Print training progress every N episodes
SAVE_INTERVAL = 100     # Save model checkpoints every N episodes