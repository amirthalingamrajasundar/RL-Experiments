"""
Training loop for DQN agents.
Manages the interaction between the agent and the OpenAI Gym environment.
"""
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
from gymnasium.wrappers import RecordVideo


class Trainer:
    """Handles training of DQN agents."""
    
    def __init__(self, agent, env_name, config, logger=None):
        self.agent = agent
        self.env = gym.make(env_name)
        self.config = config
        self.logger = logger
        
        # Lists to store metrics for plotting later
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
    
    def train(self, num_episodes):
        """
        Main training loop.
        Runs specified number of episodes, collecting data and training the agent.
        """
        print(f"\nTraining {self.agent.get_name()} for {num_episodes} episodes...")
        print(f"Environment: {self.config.ENV_NAME}")
        print(f"Device: {self.agent.device}")
        print("-" * 60)
        
        # tqdm provides a nice progress bar in the console
        pbar = tqdm(range(num_episodes), desc=self.agent.get_name())
        
        for episode in pbar:
            # Reset environment for new episode
            state, _ = self.env.reset(seed=self.config.SEED + episode)
            episode_reward = 0
            episode_loss = []
            
            # --- Inner Loop: Steps within an episode ---
            for step in range(self.config.MAX_STEPS):
                # 1. Agent selects an action based on current state
                action = self.agent.select_action(state, training=True)
                
                # 2. Environment executes action, returns next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 3. Store the experience in replay buffer
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # 4. Perform one learning update step (if buffer has enough data)
                loss = self.agent.update()
                if loss is not None:
                    episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # --- End of Episode Updates ---
            # Periodically update the target network to stabilize training
            if (episode + 1) % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            
            # Decay epsilon (reduce exploration over time)
            self.agent.decay_epsilon()
            self.agent.episodes += 1
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            if episode_loss:
                self.losses.append(np.mean(episode_loss))
            
            # Update progress bar description with recent performance
            avg_reward = np.mean(self.episode_rewards[-100:])
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'avg_100': f'{avg_reward:.1f}',
                'epsilon': f'{self.agent.epsilon:.3f}'
            })
            
            # Log detailed metrics to file if logger is present
            if self.logger and (episode + 1) % self.config.LOG_INTERVAL == 0:
                self.logger.log(episode + 1, {
                    'reward': episode_reward,
                    'avg_reward_100': avg_reward,
                    'epsilon': self.agent.epsilon,
                    'loss': np.mean(episode_loss) if episode_loss else 0
                })
        
        self.env.close()
        print(f"\nTraining complete!")
        print(f"Final avg reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
        
        return self.episode_rewards
    
    def evaluate(self, num_episodes=10, render=False, video_dir=None):
        """
        Evaluate agent performance without exploration (epsilon = 0).
        Optionally render the environment to watch the agent play.
        video_dir (str): If provided, save episode videos to this directory.
        """
        if not video_dir:
            print(f"\nEvaluating {self.agent.get_name()} for {num_episodes} episodes...")
        
        # Determine render mode based on arguments
        if video_dir:
            render_mode = 'rgb_array'  # Required for video recording
        elif render:
            render_mode = 'human'      # Required for on-screen display
        else:
            render_mode = None         # No rendering (faster)

        eval_env = gym.make(self.config.ENV_NAME, render_mode=render_mode)

        # Apply video wrapper if a directory is provided
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
            # episode_trigger=lambda x: True ensures EVERY evaluation episode is recorded
            eval_env = RecordVideo(
                eval_env, 
                video_folder=video_dir,
                episode_trigger=lambda x: True,
                name_prefix=f"{self.agent.get_name()}_eval"
            )
            print(f"\nRecording videos to: {video_dir}")
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.config.SEED + 1000 + episode)
            episode_reward = 0
            
            for step in range(self.config.MAX_STEPS):
                # Select action with training=False (disables epsilon-greedy exploration)
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: {episode_reward:.1f}")
        
        eval_env.close()
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        if video_dir:
            print(f"Videos for {self.agent.get_name()} saved to: {video_dir}\n")
        else:
            print(f"\nEvaluation Results:")
            print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        return eval_rewards