import numpy as np
import os
import platform
import time
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorld:
    """
    A simple GridWorld environment for Reinforcement Learning.

    This environment is designed for Dynamic Programming methods. The agent's goal is
    to navigate from a start position to a goal position while avoiding obstacles.

    Core RL Components:
    - States: A grid of (row, col) positions.
    - Actions: Up, Down, Left, Right.
    - Rewards: The agent receives a negative reward for each step, a larger penalty
      for hitting an obstacle, and zero reward upon reaching the goal.
    - Transitions: The environment can be deterministic or stochastic. In the
      stochastic case, the agent has a certain probability of "slipping" and
      taking a random action instead of the intended one.
    - Terminal State: The goal state is a terminal, absorbing state. Once the
      agent reaches the goal, the episode ends.
    """
    # --- Class-level constants for actions for better readability ---
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self,
                 size: int = 10,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (9, 9),
                 obstacles: Optional[List[Tuple[int, int]]] = None,
                 step_reward: float = -1.0,
                 obstacle_penalty: float = -10.0,
                 stochastic: bool = True,
                 slip_prob: float = 0.1,
                 seed: Optional[int] = 42):
        """
        Initializes the GridWorld environment.

        Args:
            size (int): The size of the square grid (size x size).
            start (Tuple[int, int]): The starting position (row, col) of the agent.
            goal (Tuple[int, int]): The goal position (row, col).
            obstacles (Optional[List[Tuple[int, int]]]): A list of obstacle positions.
            step_reward (float): The reward received for each step taken.
            obstacle_penalty (float): The penalty for hitting an obstacle.
            stochastic (bool): If True, the environment has stochastic transitions.
            slip_prob (float): The probability of the agent "slipping" and taking a
                               random action. Only used if stochastic is True.
            seed (Optional[int]): A seed for the random number generator for
                                  reproducibility.
        """
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else []

        # --- Reward Structure ---
        self.step_reward = step_reward
        self.obstacle_penalty = obstacle_penalty

        # --- Transition Dynamics ---
        self.stochastic = stochastic
        self.slip_prob = slip_prob

        if not self.stochastic:
            self.slip_prob = 0

        # --- Internal State ---
        self.pos = self.start
        self.rng = np.random.default_rng(seed)

        # --- Action to movement mapping ---
        self.actions = {
            self.UP: (-1, 0),    # Move up (row-1)
            self.DOWN: (1, 0),   # Move down (row+1)
            self.LEFT: (0, -1),  # Move left (col-1)
            self.RIGHT: (0, 1)   # Move right (col+1)
        }
        self.action_symbols = {
            self.UP: "↑",
            self.DOWN: "↓",
            self.LEFT: "←",
            self.RIGHT: "→"
        }

    def reset(self) -> Tuple[int, int]:
        """
        Resets the agent's position to the start state.

        Returns:
            Tuple[int, int]: The starting position of the agent.
        """
        self.pos = self.start
        return self.pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Executes one time step in the environment.

        This function models the transition dynamics P(s'|s, a) and the
        reward function R(s, a, s').

        Args:
            action (int): The action chosen by the agent.

        Returns:
            Tuple[Tuple[int, int], float, bool]: A tuple containing:
                - next_state (Tuple[int, int]): The agent's new position.
                - reward (float): The reward received after taking the action.
                - done (bool): True if the agent has reached a terminal state.
        """
        # --- 1. Handle Absorbing Goal State ---
        # If the agent is already at the goal, any action results in staying
        # at the goal with zero reward. The episode is effectively over.
        if self.pos == self.goal:
            return self.pos, 0.0, True

        # --- 2. Apply Stochasticity (Transition Dynamics) ---
        # With probability `slip_prob`, the agent takes a random action.
        if self.stochastic and self.rng.random() < self.slip_prob:
            action = self.rng.choice(list(self.actions.keys()))

        # --- 3. Calculate Intended Next Position ---
        move = self.actions[action]
        next_pos = (self.pos[0] + move[0], self.pos[1] + move[1])

        # --- 4. Enforce Grid Boundaries ---
        # If the agent tries to move off the grid, it stays in its current position.
        row, col = next_pos
        if not (0 <= row < self.size and 0 <= col < self.size):
            next_pos = self.pos  # Bounce back by staying in place

        # --- 5. Determine Reward and Termination ---
        reward = self.step_reward
        done = False

        if next_pos == self.goal:
            # Reaching the goal ends the episode
            done = True
        elif next_pos in self.obstacles:
            # Hitting an obstacle gives a penalty and sends the agent back
            reward = self.obstacle_penalty
            next_pos = self.pos  # Bounce back to the original position

        # --- 6. Update State and Return ---
        self.pos = next_pos
        return self.pos, reward, done

    def get_states(self) -> List[Tuple[int, int]]:
        """Returns a list of all possible states (grid cells) in the environment."""
        return [(r, c) for r in range(self.size) for c in range(self.size)]

    def get_actions(self) -> List[int]:
        """Returns a list of all possible actions."""
        return list(self.actions.keys())

    def render(self, policy: Optional[Dict[Tuple[int, int], int]] = None):
        """
        Renders the grid world to the console.

        This method provides a visual representation of the current state of the
        environment. It can also be used to visualize a policy.

        Args:
            policy (Optional[Dict[Tuple[int, int], int]]): A dictionary mapping
                states to actions. If provided, the grid will display arrows
                indicating the policy's recommended action for each state.
        """
        # --- Clear console for clean rendering (cross-platform) ---
        system_type = platform.system().lower()
        os.system("cls" if "windows" in system_type else "clear")

        # --- Build grid string ---
        grid_str = ""
        horizontal_border = "+" + "---+" * self.size + "\n"
        grid_str += horizontal_border

        for r in range(self.size):
            row_str = "|"
            for c in range(self.size):
                state = (r, c)
                cell_content = " "
                if state == self.pos:
                    cell_content = "A"  # Agent
                elif state == self.start:
                    cell_content = "S"  # Start
                elif state == self.goal:
                    cell_content = "G"  # Goal
                elif state in self.obstacles:
                    cell_content = "X"  # Obstacle
                elif policy is not None and state in policy:
                    # Display the policy's action if available
                    action = policy[state]
                    cell_content = self.action_symbols.get(action, "?")

                row_str += f" {cell_content} |"
            grid_str += row_str + "\n"
            grid_str += horizontal_border

        print(grid_str)

    def save_policy_plot(self, policy: Dict[Tuple[int, int], int], filename: str):
        """
        Generates and saves a Matplotlib plot of the grid showing the policy.

        Args:
            policy (Dict): The policy mapping states to actions.
            filename (str): The path to save the output image file.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw the grid cells and special states
        for r in range(self.size):
            for c in range(self.size):
                state = (r, c)
                color = 'white'
                if state == self.goal: color = 'lightgreen'
                elif state == self.start: color = 'lightblue'
                elif state in self.obstacles: color = 'grey'
                
                rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                
                if state == self.start:
                    ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center', fontsize=20)
                elif state == self.goal:
                    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=20)

        # Draw the policy arrows
        for state, action in policy.items():
            r, c = state
            symbol = self.action_symbols.get(action, "")
            ax.text(c + 0.5, r + 0.5, symbol, ha='center', va='center', fontsize=18, color='black')
        
        # Configure the plot
        ax.set_xticks(np.arange(self.size + 1))
        ax.set_yticks(np.arange(self.size + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.invert_yaxis() # Have (0,0) at the top-left
        ax.set_title("Agent's Learned Policy", fontsize=16)
        
        # Save the figure and close the plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nPolicy plot saved to '{filename}'")