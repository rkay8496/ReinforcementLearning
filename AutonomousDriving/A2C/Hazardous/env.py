import gym
from gym import spaces
import numpy as np

class AutonomousDrivingEnv(gym.Env):
    def __init__(self):
        super(AutonomousDrivingEnv, self).__init__()

        # Define the action space: Accelerate, Maintain speed, Brake
        self.action_space = spaces.Discrete(3)

        # Define the observation space: Car speed, Distance to the vehicle ahead
        self.observation_space = spaces.MultiDiscrete([21, 11])

        # Define the initial state
        self.state = self.reset()

    def reset(self):
        self.trace = {
            'speed': [],
            'distance': [],
            'action': [],
            '_safe': [],
        }
        # Initialize the state randomly
        self.state = self.observation_space.sample()
        self.trace['speed'].append((len(self.trace['speed']), int(self.state[0])))
        self.trace['distance'].append((len(self.trace['distance']), int(self.state[1])))
        self.trace['_safe'].append((len(self.trace['_safe']), True))
        return self.state

    def step(self, action):
        speed, distance = self.state

        self.trace['action'].append((len(self.trace['action']), int(action)))

        # Apply the action
        if action == 0: # Accelerate
            speed = min(speed + 1, 20)
        elif action == 1: # Maintain speed
            pass
        elif action == 2: # Brake
            speed = max(speed - 1, 0)
        else:
            raise ValueError("Invalid action")

        # Update the distance
        distance = max(distance - int(speed / 2), 0)

        # Check if the distance is less than 1 (collision)
        done = distance <= 1

        # Calculate the reward
        if done:
            reward = 0
        else:
            reward = 1

        self.state = np.array([speed, distance])
        self.trace['speed'].append((len(self.trace['speed']), int(self.state[0])))
        self.trace['distance'].append((len(self.trace['distance']), int(self.state[1])))
        self.trace['_safe'].append((len(self.trace['_safe']), not done))

        return self.state, reward, done, {}

    def do_post_process(self):
        self.trace['speed'].pop()
        self.trace['distance'].pop()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
