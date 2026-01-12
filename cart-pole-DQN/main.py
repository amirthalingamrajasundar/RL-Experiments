"""
Main entry point for training DQN variants.
Allows running individual agent configurations or a full comparison of all 4 variants.

Usage Examples:
    python3 main.py                              # Train basic DQN
    python3 main.py --double_dqn                 # Train Double DQN (DDQN)
    python3 main.py --prioritized_replay         # Train DQN with PER
    python3 main.py --double_dqn --prioritized_replay  # Train DDQN with PER
    python3 main.py --run_all                    # Run all 4 sequentially and plot comparison
"""
import argparse
import gymnasium as gym
import torch
import numpy as np
import random
import os

import config
from agent import DQNAgent
from trainer import Trainer
from utils.logger import Logger
from utils.plotter import plot_learning_curves

import warnings
warnings.simplefilter('ignore')


def set_seeds(seed):
    """
    Set random seeds for all libraries to ensure reproducibility.
    This guarantees that two runs with the same seed get the same initial weights and environment randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_agent(use_double_dqn, use_prioritized_replay, log_dir='logs'):
    """
    Configure, train, and save a single agent variant.
    """
    # Set seeds before creating anything
    set_seeds(config.SEED)
    
    # Create a temporary environment just to get state/action dimensions
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Initialize the agent with specific configuration
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        use_double_dqn=use_double_dqn,
        use_prioritized_replay=use_prioritized_replay
    )
    
    # Initialize logger for tracking progress
    logger = Logger(log_dir, agent.get_name())
    
    # Initialize trainer
    trainer = Trainer(agent, config.ENV_NAME, config, logger)
    
    # Start training loop
    rewards = trainer.train(config.NUM_EPISODES)
    
    # Save the trained model weights
    agent.save(f'{log_dir}/{agent.get_name()}_final.pth')
    
    # Print final statistics
    logger.print_summary()

    # --- Video Evaluation ---
    video_dir = os.path.join(log_dir, 'videos', agent.get_name())
    trainer.evaluate(num_episodes=1, video_dir=video_dir)
    
    return agent.get_name(), rewards


def run_all_experiments():
    """
    Sequentially run all four DQN variants and generate comparison plots.
    This may take a while depending on hardware!
    """
    print("\n" + "="*60)
    print("Running Complete Comparison: DQN, DDQN, DQN+PER, DDQN+PER")
    print("="*60)
    
    # Define the 4 configurations: (use_ddqn, use_per)
    configs = [
        (False, False),  # Standard DQN
        (True, False),   # Double DQN
        (False, True),   # DQN + PER
        (True, True),    # DDQN + PER
    ]
    
    results = {}
    
    # Loop through each configuration and train
    for use_ddqn, use_per in configs:
        name, rewards = train_agent(use_ddqn, use_per)
        results[name] = rewards
        print("\n")
    
    # Generate and save comparison plots
    print("\n" + "="*60)
    print("Generating Comparison Plots")
    print("="*60)
    plot_learning_curves(results)
    
    # Print a final ranked summary table
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    # Sort results by their average reward over the last 100 episodes
    sorted_results = sorted(results.items(), 
                          key=lambda x: np.mean(x[1][-100:]), 
                          reverse=True)
    
    for rank, (name, rewards) in enumerate(sorted_results, 1):
        avg_reward = np.mean(rewards[-100:])
        print(f"{rank}. {name:12s}: {avg_reward:6.2f} (avg last 100 episodes)")
    
    print("="*60)


def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Train DQN variants on CartPole-v1')
    
    parser.add_argument('--double_dqn', action='store_true',
                       help='Use Double DQN (default: False)')
    parser.add_argument('--prioritized_replay', action='store_true',
                       help='Use Prioritized Experience Replay (default: False)')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all four variants and compare')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs and models (default: logs)')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_experiments()
    else:
        # Run a single specific configuration based on flags
        name, rewards = train_agent(
            use_double_dqn=args.double_dqn,
            use_prioritized_replay=args.prioritized_replay,
            log_dir=args.log_dir
        )
        
        # Generate a simple plot just for this one run
        from utils.plotter import plot_rewards
        plot_rewards({name: rewards}, save_path=f'{args.log_dir}/{name}_training.png')


if __name__ == '__main__':
    main()