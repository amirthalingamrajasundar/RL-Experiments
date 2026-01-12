# =============================================================================
# Main Execution Script for Dynamic Programming Agents
# =============================================================================
#
# Description:
# This script serves as the main entry point for running and evaluating the
# Dynamic Programming (DP) agents (Value Iteration and Policy Iteration) on the
# GridWorld environment.
#
# Usage:
# 1. Ensure you have the following files in the same directory:
#    - env.py: The GridWorld environment.
#    - agents.py: The file containing the DP agent implementations.
#    - random_oracle.pyc: The function to generate your unique map.
#
# 2. Run this script from your terminal: python main.py
#
# 3. You will be prompted to enter a seed to generate a randomized grid.
#
# What this script does:
# - Sets up a unique GridWorld environment based on a random seed.
# - Runs the Value Iteration algorithm to find the optimal policy.
# - Runs the Policy Iteration algorithm to find the optimal policy.
# - For each agent, it visualizes the final policy and offers an interactive demo.
# - Generates and saves a plot (`learning_progress.png`) comparing the
#   learning speed of both agents.
#
# =============================================================================

# --- Import necessary classes and functions ---
# GridWorld: The environment our agents will operate in.
# ValueIterationAgent, PolicyIterationAgent, BaseAgent: The DP agents and their base class.
# get_start_goal_and_obstacle_states: A function to create a unique problem set for you.
from env import GridWorld
from agents import ValueIterationAgent, PolicyIterationAgent, BaseAgent
import sys


# Import `get_start_goal_and_obstacle_states` function according to the version of Python
version = sys.version_info[:2]  # (major, minor)

if version == (3, 7):
    from random_oracle_3_7 import get_start_goal_and_obstacle_states
elif version == (3, 8):
    from random_oracle_3_8 import get_start_goal_and_obstacle_states
elif version == (3, 9):
    from random_oracle_3_9 import get_start_goal_and_obstacle_states
elif version == (3, 10):
    from random_oracle_3_10 import get_start_goal_and_obstacle_states
elif version == (3, 11):
    from random_oracle_3_11 import get_start_goal_and_obstacle_states
elif version == (3, 12):
    from random_oracle_3_12 import get_start_goal_and_obstacle_states
elif version == (3, 13):
    from random_oracle_3_13 import get_start_goal_and_obstacle_states
else:
    raise ImportError("No suitable random oracle found for this Python version.")

# The `if __name__ == "__main__":` block is the standard entry point for a Python script.
# The code inside this block will only run when the script is executed directly.
if __name__ == "__main__":
    
    # =========================================================================
    # --- 1. Environment Setup ---
    # =========================================================================
    
    # The 'random oracle' function takes a seed to generate a unique but
    # consistent set of start, goal, and obstacle positions. Using the same seed
    # produces the same configuration every time, which is useful for reproducibility.
    seed = int(input("Enter a seed (any integer) to generate a randomized GridWorld: "))
    start, goal, obstacles = get_start_goal_and_obstacle_states(seed)
    
    print(f"\nYour unique environment has been generated:")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Obstacles: {obstacles}")

    presence_of_obstacles = True if input("\nHave you faced obstacles in your path? (y/n): ").lower() == 'y' else False

    if not presence_of_obstacles:
        obstacles = None

    rainy_day = True if input("Is it a rainy day? (y/n): ").lower() == 'y' else False

    slip_prob = 0
    if rainy_day:
        slip_prob = float(input("Enter the slip probability for rainy days: "))

    gamma_vi = float(input("\nEnter the discount factor (gamma) for Value Iteration: "))
    gamma_pi = float(input("Enter the discount factor (gamma) for Policy Iteration: "))


    # We now create an instance of our GridWorld environment with these unique parameters.
    env = GridWorld(
        size=10,
        start=start,
        goal=goal,
        obstacles=obstacles,    # Use the obstacles generated from your ID.
        stochastic=rainy_day,   # You can toggle this between True and False to see how the agents
                                # perform in deterministic vs. stochastic environments.
        slip_prob=slip_prob,    # The probability of a random action if `stochastic` is True.
        seed=42                 # The seed ensures that "random" slips are the same every run,
                                # which is essential for reproducible results and debugging.
    )
    
    # =========================================================================
    # --- 2. Agent Execution & Evaluation ---
    # =========================================================================

    # --- Run Value Iteration ---
    print("\n" + "="*40)
    print("--- Running Value Iteration Agent ---")
    print("="*40)
    # We create an instance of the Value Iteration agent.
    # `gamma` is the discount factor; a value close to 1 (e.g., 0.99) makes the
    # agent more "farsighted," valuing future rewards more heavily.
    vi_agent = ValueIterationAgent(env, gamma=gamma_vi, theta=1e-3)
    
    # This is the main function call that starts the learning process. The agent will
    # run the Value Iteration algorithm until its value function converges.
    vi_agent.solve()

    # Save the policy plot to a file
    vi_agent.save_policy_visualization(filename="value_iteration_policy.png")
    
    # This part allows you to interactively test the learned policy. You can press Enter
    # to step through an episode and see the agent follow its strategy.
    if input("\nRun interactive demo for Value Iteration? (y/n): ").lower() == 'y':
        vi_agent.run_interactive_episode()
    
    # --- Run Policy Iteration ---
    print("\n" + "="*40)
    print("--- Running Policy Iteration Agent ---")
    print("="*40)
    # We now do the same for the Policy Iteration agent.
    pi_agent = PolicyIterationAgent(env, gamma=gamma_pi, theta=1e-3)
    
    # The `solve` method for this agent will alternate between policy evaluation
    # and policy improvement until the policy is stable.
    pi_agent.solve()

    # Save the policy plot to a file
    pi_agent.save_policy_visualization(filename="policy_iteration_policy.png")

    # Offer an interactive demo for the policy iteration agent.
    if input("\nRun interactive demo for Policy Iteration? (y/n): ").lower() == 'y':
        pi_agent.run_interactive_episode()