import numpy as np
import gym
from gym import spaces

class ClawMachineEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple gridworld where an agent learns to pick up a toy.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=5):
        super(ClawMachineEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(5) # 0: up, 1: down, 2: left, 3: right, 4: grab
        self.observation_space = spaces.MultiDiscrete([grid_size]*grid_size)

        # Initialize state
        self.grid_size = grid_size
        self.state = np.zeros((grid_size, grid_size), dtype=int)
        self.claw_position = [0, 0]

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.claw_position = [0, 0]
        return self.state.copy()

    def step(self, action):
        """
        The agent takes a step in the environment.
        """
        reward = 0
        done = False

        if action == 0: # up
            self.claw_position[1] = max(0, self.claw_position[1] - 1)
        elif action == 1: # down
            self.claw_position[1] = min(self.grid_size - 1, self.claw_position[1] + 1)
        elif action == 2: # left
            self.claw_position[0] = max(0, self.claw_position[0] - 1)
        elif action == 3: # right
            self.claw_position[0] = min(self.grid_size - 1, self.claw_position[0] + 1)
        elif action == 4: # grab
            if self.state[self.claw_position[1], self.claw_position[0]] == 1: # if there is a toy
                reward = 1
                self.state[self.claw_position[1], self.claw_position[0]] = 0 # remove the toy
                done = True
            else:
                reward = 0
        return self.state.copy(), reward, done, {}

    def render(self, mode='console'):
        pass