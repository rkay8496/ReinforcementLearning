import json

import gymnasium as gym
from gym import spaces
import numpy as np
import stl
import copy


class MovingObstacleEnv(gym.Env):
    def __init__(self, size=8.0, glitches=1, train=True, global_name='model'):
        super(MovingObstacleEnv, self).__init__()

        self.size = size
        self.glitches = glitches
        self.train = train
        self.global_name = global_name

        if not self.train:
            self.f = open(self.global_name + '.json', 'w')

        self.glitches_counter = 0
        self.robot_turn = 0

        # Define the action space: stay, left, right, up, down
        self.action_space = spaces.Discrete(5)

        # Define the observation space:
        self.observation_space = spaces.MultiDiscrete([self.size - 1, self.size - 1, self.size, self.size])

        # Define the initial state
        self.state, _ = self.reset()

        self.trace = {
            'observations': [],
            'actions': [],
            '_safe': [],
        }

    def reset(self, seed=5345):
        self.glitches_counter = 0
        self.robot_turn = 0

        self.state = [self.size - 2, self.size - 2, 0, 0]
        self.trace = {
            'observations': [
                copy.deepcopy(self.state)
            ],
            'actions': [

            ],
            '_safe': [
                True
            ],
        }
        return self.state, {}

    def step(self, action):
        self.trace['actions'].append(action)
        self.update_agent_location(action)
        self.robot_turn += 1

        if self.robot_turn < 2:
            if self.glitches_counter < self.glitches:
                is_glitch = np.random.choice([True, False])
                if is_glitch:
                    self.update_obstacle_location()
                    self.glitches_counter += 1
                    self.robot_turn = 0
        else:
            self.update_obstacle_location()
            self.robot_turn = 0

        self.trace['observations'].append(copy.deepcopy(self.state))

        terminated = False
        info = {
            'satisfiable': False
        }
        reward = 0

        if self.state[2] < 0 or self.state[2] > self.size - 1 or self.state[3] < 0 or self.state[3] > self.size - 1:
            if self.state[2] < 0:
                self.state[2] = 0
            if self.state[2] > self.size - 1:
                self.state[2] = self.size - 1
            if self.state[3] < 0:
                self.state[3] = 0
            if self.state[3] > self.size - 1:
                self.state[3] = self.size - 1
            terminated = True
            info['satisfiable'] = False
            if not self.train:
                for key in self.trace.keys():
                    var_trace = self.trace[key][:]
                    if key == 'observations':
                        self.trace[key].clear()
                        for idx, item in enumerate(var_trace):
                            obj = []
                            for elem in item:
                                obj.append(int(elem))
                            self.trace[key].append(obj)
                    elif key == 'actions':
                        self.trace[key].clear()
                        for idx, item in enumerate(var_trace):
                            self.trace[key].append(int(item))
                self.trace['_safe'].append(not terminated)
                self.f.write(json.dumps(self.trace) + '\n')
            return self.state, reward, terminated, False, info

        if (self.state[2] != self.state[0] or self.state[3] != self.state[1]) and \
                (self.state[2] != self.state[0] + 1 or self.state[3] != self.state[1]) and \
                (self.state[2] != self.state[0] or self.state[3] != self.state[1] + 1) and \
                (self.state[2] != self.state[0] + 1 or self.state[3] != self.state[1] + 1):
            reward = 1
            terminated = False
            info['satisfiable'] = True
            self.trace['_safe'].append(not terminated)
        else:
            terminated = True
            info['satisfiable'] = False
            if not self.train:
                for key in self.trace.keys():
                    var_trace = self.trace[key][:]
                    if key == 'observations':
                        self.trace[key].clear()
                        for idx, item in enumerate(var_trace):
                            obj = []
                            for elem in item:
                                obj.append(int(elem))
                            self.trace[key].append(obj)
                    elif key == 'actions':
                        self.trace[key].clear()
                        for idx, item in enumerate(var_trace):
                            self.trace[key].append(int(item))
                self.trace['_safe'].append(not terminated)
                self.f.write(json.dumps(self.trace) + '\n')

        return self.state, reward, terminated, False, info

    def update_obstacle_location(self):
        if self.state[0] > self.state[2]:
            self.state[0] = max(0, self.state[0] - 1)
        elif self.state[0] < self.state[2]:
            self.state[0] = min(self.size - 2, self.state[0] + 1)
        if self.state[1] > self.state[3]:
            self.state[1] = max(0, self.state[1] - 1)
        elif self.state[1] < self.state[3]:
            self.state[1] = min(self.size - 2, self.state[1] + 1)

    def update_agent_location(self, action):
        if action == 1:
            self.state[2] -= 1
        elif action == 2:
            self.state[2] += 1
        elif action == 3:
            self.state[3] -= 1
        elif action == 4:
            self.state[3] += 1

    def do_post_process(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.train:
            self.f.close()
