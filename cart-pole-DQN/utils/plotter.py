"""
Visualization utilities.
Uses Matplotlib to create training curves and bar charts for performance comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def smooth(data, window=10):
    """
    Smooth data using a moving average window.
    Helps to visualize general trends in noisy reward signals.
    """
    if len(data) < window:
        return data
    # Use numpy's convolution to calculate moving average efficiently
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_rewards(rewards_dict, save_path=None):
    """
    Plot training rewards over time for multiple agents.
    
    Args:
        rewards_dict: Dictionary {agent_name: list_of_rewards}
        save_path: Optional path to save the figure to disk.
    """
    plt.figure(figsize=(12, 6))
    
    # Standard colors for distinct lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (name, rewards) in enumerate(rewards_dict.items()):
        color = colors[idx % len(colors)]
        # Plot raw rewards faintly in the background
        plt.plot(rewards, alpha=0.2, color=color)
        
        # Plot smoothed rewards with a solid, thicker line
        smoothed = smooth(rewards, window=20)
        # Adjust x-axis for smoothed data (convolution shrinks the array size slightly)
        episodes = np.arange(len(rewards) - len(smoothed), len(rewards))
        plt.plot(episodes, smoothed, label=name, linewidth=2, color=color)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_comparison_bar(rewards_dict, save_path=None):
    """
    Create a bar plot comparing the final average performance of agents.
    Includes error bars representing standard deviation over the last 100 episodes.
    """
    plt.figure(figsize=(10, 6))
    
    names = list(rewards_dict.keys())
    # Calculate mean and std dev for the last 100 episodes of each agent
    final_scores = [np.mean(rewards[-100:]) for rewards in rewards_dict.values()]
    std_scores = [np.std(rewards[-100:]) for rewards in rewards_dict.values()]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(names, final_scores, yerr=std_scores, capsize=5,
                   color=colors[:len(names)], alpha=0.7, edgecolor='black')
    
    # Add text labels on top of each bar for clarity
    for bar, score in zip(bars, final_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Average Reward (Last 100 Episodes)', fontsize=12)
    plt.title('Final Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_learning_curves(rewards_dict, save_dir='results'):
    """
    Wrapper function to generate and save both types of comparison plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Line plot of training curves
    plot_rewards(rewards_dict, 
                save_path=os.path.join(save_dir, 'training_curves.png'))
    
    # 2. Bar chart of final performance
    plot_comparison_bar(rewards_dict,
                       save_path=os.path.join(save_dir, 'performance_comparison.png'))