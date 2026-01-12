"""
Logging utilities for training.
Saves training metrics to JSON files for later analysis or reproduction.
"""
import os
import json
from datetime import datetime


class Logger:
    """Simple logger that saves metrics to a JSON file periodically."""
    
    def __init__(self, log_dir, agent_name):
        self.log_dir = log_dir
        self.agent_name = agent_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique filename using timestamp to avoid overwriting previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{agent_name}_{timestamp}.json")
        
        self.logs = []
    
    def log(self, episode, metrics):
        """
        Append new metrics for a specific episode to the in-memory log list.
        """
        log_entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics  # Unpack whatever metrics dictionary is passed
        }
        self.logs.append(log_entry)
        
        # Save to disk every 50 episodes to prevent data loss if run crashes
        if episode % 50 == 0:
            self.save()
    
    def save(self):
        """Flush in-memory logs to the JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def print_summary(self):
        """Print a brief summary of the logged run to the console."""
        if not self.logs:
            return
        
        print(f"\n{'='*60}")
        print(f"Training Summary: {self.agent_name}")
        print(f"{'='*60}")
        print(f"Total Episodes: {len(self.logs)}")
        
        # Check if specific keys exist before printing them
        if 'avg_reward_100' in self.logs[-1]:
            print(f"Final Avg Reward (100 eps): {self.logs[-1]['avg_reward_100']:.2f}")
        
        if 'loss' in self.logs[-1]:
            print(f"Final Loss: {self.logs[-1]['loss']:.4f}")
        
        print(f"Logs saved to: {self.log_file}")
        print(f"{'='*60}\n")