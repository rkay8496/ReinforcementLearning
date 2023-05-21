import numpy as np
import gym
from gym import spaces
from stable_baselines3 import A2C

class ClawMachineEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(ClawMachineEnv, self).__init__()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)

        self.state = None
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.state[self.grid_size//2][self.grid_size//2] = 1
        return self.state

    def step(self, action):
        # Take the action (move the claw)
        if action == 0: # up
            self.state = np.roll(self.state, shift=-1, axis=0)
        elif action == 1: # down
            self.state = np.roll(self.state, shift=1, axis=0)
        elif action == 2: # left
            self.state = np.roll(self.state, shift=-1, axis=1)
        elif action == 3: # right
            self.state = np.roll(self.state, shift=1, axis=1)
        elif action == 4: # grab
            if self.state[0][self.grid_size//2] == 1:
                self.state[0][self.grid_size//2] = 0
                return self.state, 1, True, {}
            else:
                return self.state, 0, True, {}

        return self.state, 0, False, {}


# Create environment
env = ClawMachineEnv(grid_size=5)

# Initialize agent
model = A2C("MlpPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("a2c_claw_machine")

# Load the trained agent
model = A2C.load("a2c_claw_machine")

# Enjoy trained agent
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()
