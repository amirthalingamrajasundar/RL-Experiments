import numpy as np
import time
from typing import Dict, List, Tuple
import sys

# We assume the GridWorld environment is in a file named env.py in the same directory.
from env import GridWorld


class BaseAgent:
    """
    A base class for Dynamic Programming (DP) agents.

    DP methods are "model-based," meaning the agent needs to know the full dynamics of the
    environment to solve it. This includes the transition probabilities (P) and reward function (R).

    This base class handles functionalities common to all our DP agents, such as:
    - Storing the environment, discount factor (gamma), and convergence threshold (theta).
    - Calculating Q-values, which is a core operation based on the environment's model.
    - Visualizing the final policy and plotting the learning progress.
    - Running interactive demos of the learned policy.
    """

    def __init__(self, env: GridWorld, gamma: float = 0.9, theta: float = 1e-6):
        """
        Initializes the BaseAgent.

        Args:
            env (GridWorld): The environment the agent will learn in.
            gamma (float): The discount factor, which determines the agent's preference for
                           immediate rewards over future rewards. A value closer to 1 gives
                           more weight to future rewards.
            theta (float): A small positive number used as a stopping criterion for the
                           iterative algorithms. When the maximum change in the value
                           function is less than theta, we consider it converged.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.states = self.env.get_states()
        
        # The Value Table (V) stores the value V(s) for each state s.
        # We initialize it to all zeros.
        self.value_table = {state: 0.0 for state in self.states}
        
        # The Policy (π) stores the action π(s) to take in each state s.
        # We initialize it with a default action for all valid (non-terminal, non-obstacle) states.
        self.policy = {
            state: self.env.get_actions()[0] for state in self.states 
            if state not in self.env.obstacles and state != self.env.goal
        }

    def _calculate_q_value(self, state: Tuple[int, int], action: int) -> float:
        """
        Calculates the Q-value, Q(s, a), for a given state-action pair.

        This function represents the core of a model-based approach. It calculates the
        expected return of taking action 'a' in state 's' by looking one step ahead.
        It uses the Bellman equation: Q(s, a) = Σ [ P(s'|s,a) * (R(s,a,s') + gamma * V(s')) ]
        
        Since our environment is stochastic (has a `slip_prob`), we must account for all
        possible outcomes of intending to take a certain action.
        """
        # The agent intends to take `action`.
        # With probability (1 - slip_prob), it succeeds.
        # With probability `slip_prob`, it "slips" and takes a random action instead.
        
        possible_actions = self.env.get_actions()
        slip_prob_per_action = self.env.slip_prob / len(possible_actions)
        
        expected_value = 0.0
        
        # We iterate over all *actual* actions that could happen due to stochasticity.
        for actual_action in possible_actions:
            # Calculate the total probability of `actual_action` occurring if we intend to take `action`.
            prob = slip_prob_per_action
            if actual_action == action:
                # This is the probability of succeeding + the probability of slipping *and* randomly choosing the intended action.
                prob += (1.0 - self.env.slip_prob)

            # To get the outcome (next_state, reward), we can use the environment's logic
            # without changing the agent's actual position in the environment. This is like "imagining" the step.
            original_pos = self.env.pos
            self.env.pos = state
            next_state, reward, _ = self.env.step(actual_action)
            self.env.pos = original_pos  # Restore the environment's state

            # Add the value of this possible outcome, weighted by its probability, to the total expected value.
            expected_value += prob * (reward + self.gamma * self.value_table[next_state])
            
        return expected_value

    def run_single_episode(self, policy: Dict[Tuple[int, int], int]) -> float:
        """
        Runs a single episode using a given policy. This is used to track the
        performance of the agent during the learning process.

        Args:
            policy: The policy to follow during the episode.

        Returns:
            The total reward accumulated during the episode.
        """
        self.env.reset()
        done = False
        episode_reward = 0.0
        # A step limit prevents infinite loops if the policy is suboptimal.
        for _ in range(100):
            if done:
                break
            state = self.env.pos
            action = policy.get(state)
            if action is None: # This happens if we are in the Goal state or an unexpected state.
                break
            
            _, reward, done = self.env.step(action)
            episode_reward += reward
        return episode_reward

    def run_interactive_episode(self):
        """
        Allows a user to step through an episode played by the learned policy.
        This is a great tool for visualizing and debugging the agent's behavior.
        """
        print("\n--- Starting Interactive Policy Run ---")
        self.env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        # Loop until the episode ends or we hit the step limit.
        while not done and step_count < 100:
            # Render the grid, showing the policy arrows and the agent's current position 'A'.
            self.env.render(policy=self.policy)
            state = self.env.pos
            action = self.policy.get(state)
            
            print(f"Step: {step_count + 1}")
            print(f"Current State: {state}")
            if action is None:
                print("Goal reached or policy not defined for this state. Episode ends.")
                break
            
            action_symbol = self.env.action_symbols.get(action, '?')
            print(f"Policy suggests action: {action_symbol}")
            
            # Wait for the user to press Enter to proceed to the next step.
            input("Press Enter to take the next step...")
            
            _, reward, done = self.env.step(action)
            episode_reward += reward
            step_count += 1

            print(f"Reward for this step: {reward}, Total Reward: {episode_reward:.2f}")
            time.sleep(0.1) # A small delay for a better user experience.
        
        # Final render to show the last state.
        self.env.render(policy=self.policy)
        if done:
            print("Goal reached!")
        else:
            print("Episode ended (step limit reached).")
        print(f"Final Total Reward: {episode_reward:.2f}")

    def save_policy_visualization(self, filename: str):
        """
        Saves a graphical visualization of the policy to a file.
        
        Args:
            filename (str): The name of the file to save the plot (e.g., 'policy.png').
        """
        self.env.save_policy_plot(policy=self.policy, filename=filename)
        
    def solve(self):
        """A placeholder for the main solving method in child classes."""
        raise NotImplementedError("The 'solve' method must be implemented by a subclass.")


class ValueIterationAgent(BaseAgent):
    """
    An agent that solves the GridWorld using the Value Iteration algorithm.
    Value Iteration finds the optimal value function V*(s) first, and then
    extracts the optimal policy π*(s) from it.
    """

    def solve(self) -> List[float]:
        """
        Performs the Value Iteration algorithm.

        It repeatedly applies the Bellman Optimality update to the value function for all states
        until the function converges.

        Returns:
            A list of rewards, where each entry is the total reward from a test
            episode run after one full update cycle.
        """
        print("--- Starting Value Iteration ---")
        rewards_history = []
        iteration = 0
        # Main loop: continue until the value function changes by a very small amount (theta).
        while True:
            iteration += 1
            # `delta` will track the largest change in the value of any state in this sweep.
            delta = 0.0

            # All calculations in this sweep will read from the original `self.value_table` (V_k)
            # and write to the `new_value_table` (V_k+1).
            new_value_table = self.value_table.copy()

            # Sweep through all states.
            for state in self.states:
                # Terminal states and obstacles have a fixed value of 0 and no actions.
                if state == self.env.goal or state in self.env.obstacles:
                    continue
                
                # This is the core of Value Iteration: applying the Bellman Optimality Equation.
                # V(s) = max_a Q(s, a)
                max_q = -np.inf
                for action in self.env.get_actions():
                    q_value = self._calculate_q_value(state, action)
                    max_q = max(max_q, q_value)
                new_value_table[state] = max_q
                
                # Update delta with the absolute change in the state's value.
                delta = max(delta, abs(new_value_table[state] - self.value_table[state]))

            # Now, update the main value table with the new, synchronously calculated values.
            self.value_table = new_value_table
            
            # --- Performance Tracking for Plotting ---
            # After each full sweep, we extract the current greedy policy to test its performance.
            current_policy = self.policy.copy()
            
            for state in self.states:
                if state == self.env.goal or state in self.env.obstacles:
                    continue
                best_q = -np.inf
                for action in self.env.get_actions():
                    q_value = self._calculate_q_value(state, action)
                    if q_value > best_q:
                        best_q = q_value
                        current_policy[state] = action

            # Run one test episode and record the reward.
            reward = self.run_single_episode(current_policy)
            rewards_history.append(reward)
            print(f"VI Cycle {iteration}: Reward={reward:.2f}, Delta={delta:.6f}", end='\r')
            sys.stdout.flush()

            # Check for convergence.
            if delta < self.theta:
                print("\nValue function converged.")
                self.policy = current_policy  # Set the final, optimal policy.
                break
        
        print("Value Iteration complete.")
        return rewards_history


class PolicyIterationAgent(BaseAgent):
    """
    An agent that solves the GridWorld using the Policy Iteration algorithm.
    Policy Iteration alternates between two steps:
    1. Policy Evaluation: Calculate the value function Vπ for the current policy π.
    2. Policy Improvement: Improve the policy by acting greedily with respect to Vπ.
    """

    def policy_evaluation(self):
        """
        This is the "E" step. We evaluate the current policy by iteratively updating
        the value function using the Bellman Expectation Equation until it converges.
        Vπ(s) = E[R + gamma * Vπ(s') | s, π(s)]
        """

        while True:
            delta = 0.0

            # All calculations in this sweep will read from the original `self.value_table` (V_k)
            # and write to the `new_value_table` (V_k+1).
            new_value_table = self.value_table.copy()

            for state in self.states:
                if state == self.env.goal or state in self.env.obstacles:
                    continue
                
                # Consider only the action given by the current policy.
                # And calculate the new value using the OLD value table.
                new_value_table[state] = self._calculate_q_value(state, self.policy[state])
                
                delta = max(delta, abs(new_value_table[state] - self.value_table[state]))

            # Now, update the main value table with the new, synchronously calculated values.
            self.value_table = new_value_table

            if delta < self.theta:
                break # The value function for the current policy has converged.

    def policy_improvement(self) -> bool:
        """
        This is the "I" step. We improve the policy by acting greedily with respect
        to the value function calculated in the evaluation step.

        Returns:
            bool: True if the policy is stable (no changes were made), False otherwise.
        """
        is_policy_stable = True
        for state in self.policy:
            old_action = self.policy[state]
            best_action = old_action
            
            # Find the best action by looking one step ahead, similar to Value Iteration.
            # And update action for current state greedily
            for action in self.env.get_actions():
                if self._calculate_q_value(state, action) > self._calculate_q_value(state, best_action):
                    best_action = action
            self.policy[state] = best_action
            
            # If the best action is different from our old action, the policy is not yet stable.
            if old_action != best_action:
                is_policy_stable = False
        return is_policy_stable
        
    def solve(self) -> List[float]:
        """
        Performs the Policy Iteration algorithm, alternating between evaluation and
        improvement until the policy no longer changes.
        """
        print("\n--- Starting Policy Iteration ---")
        rewards_history = []
        iteration = 0
        # The main loop continues until the policy becomes stable.
        while True:
            iteration += 1
            
            # 1. Policy Evaluation Step
            self.policy_evaluation()
            
            # 2. Policy Improvement Step
            is_stable = self.policy_improvement()
            
            # --- Performance Tracking for Plotting ---
            # After each improvement step, run a test episode with the new policy.
            reward = self.run_single_episode(self.policy)
            rewards_history.append(reward)
            print(f"PI Cycle {iteration}: Reward={reward:.2f}, Policy Stable={is_stable}", end='\r')
            sys.stdout.flush()

            if is_stable:
                print("\nPolicy has stabilized.")
                break
        
        print("Policy Iteration complete.")
        return rewards_history