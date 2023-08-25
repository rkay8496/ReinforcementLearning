import json

import gymnasium as gym
from gym import spaces
import numpy as np
import copy


class ServingRobotsEnv(gym.Env):
    def __init__(self, train=True, global_name='model'):
        super(ServingRobotsEnv, self).__init__()

        self.size = 4
        self.train = train
        self.global_name = global_name

        if not self.train:
            self.f = open('models/' + self.global_name + '.json', 'w')

        # 0: Left, 1: Right, 2: Up, 3: Down
        self.action_space = spaces.MultiDiscrete([4, 4])

        # [0, 1]: locations, [2:10]: requests(0, 1, 8, 9, 12, 13, 3, 7, 11)
        self.observation_space = spaces.MultiDiscrete([self.size * self.size, self.size * self.size, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        self.state, _ = self.reset()

        self.trace = {
            'observations': [],
            'actions': [],
            '_done': []
        }

    def reset(self, seed=5345):
        # self.state = self.observation_space.sample()
        # while 1 not in self.state[2:]:
        #     self.state = self.observation_space.sample()
        # self.state[0] = 2
        # self.state[1] = 15
        self.state = [5, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.trace = {
            'observations': [
                copy.deepcopy(self.state)
            ],
            'actions': [

            ],
            '_done': [False]
        }
        return self.state, {}

    def step(self, action):
        info = {}
        reward = 0

        self.trace['actions'].append(action)

        battery, served, cross_paths = self.update_observation(action)
        self.trace['observations'].append(copy.deepcopy(self.state))

        reward = battery + served + cross_paths

        all_serving_done = all(x == 0 for x in self.state[2:])
        terminated = all_serving_done

        self.trace['_done'].append(terminated)

        if terminated:
            if not self.train:
                for key in self.trace.keys():
                    if key == 'observations' or key == 'actions':
                        for idx, item in enumerate(self.trace[key]):
                            if not isinstance(item, list):
                                self.trace[key][idx] = item.tolist()
                self.f.write(json.dumps(self.trace) + '\n')

        return self.state, reward, terminated, False, info

    def update_observation(self, action):
        battery, served, cross_paths = 0, 0, 0

        if self.state[0] == 0:
            if action[0] == 0:
                self.state[0] = 0
            elif action[0] == 1:
                self.state[0] = 1
            elif action[0] == 2:
                self.state[0] = 0
            elif action[0] == 3:
                self.state[0] = 4
        elif self.state[0] == 1:
            if action[0] == 0:
                self.state[0] = 0
            elif action[0] == 1:
                self.state[0] = 2
            elif action[0] == 2:
                self.state[0] = 1
            elif action[0] == 3:
                self.state[0] = 5
        elif self.state[0] == 2:
            if action[0] == 0:
                self.state[0] = 1
            elif action[0] == 1:
                self.state[0] = 3
            elif action[0] == 2:
                self.state[0] = 2
            elif action[0] == 3:
                self.state[0] = 6
        elif self.state[0] == 3:
            if action[0] == 0:
                self.state[0] = 2
            elif action[0] == 1:
                self.state[0] = 3
            elif action[0] == 2:
                self.state[0] = 3
            elif action[0] == 3:
                self.state[0] = 7
        elif self.state[0] == 4:
            if action[0] == 0:
                self.state[0] = 4
            elif action[0] == 1:
                self.state[0] = 5
            elif action[0] == 2:
                self.state[0] = 0
            elif action[0] == 3:
                self.state[0] = 8
        elif self.state[0] == 5:
            if action[0] == 0: 
                self.state[0] = 4
            elif action[0] == 1:
                self.state[0] = 6
            elif action[0] == 2:
                self.state[0] = 1
            elif action[0] == 3:
                self.state[0] = 9
        elif self.state[0] == 6:
            if action[0] == 0: 
                self.state[0] = 5
            elif action[0] == 1:
                self.state[0] = 7
            elif action[0] == 2:
                self.state[0] = 2
            elif action[0] == 3:
                self.state[0] = 10
        elif self.state[0] == 7:
            if action[0] == 0: 
                self.state[0] = 6
            elif action[0] == 1:
                self.state[0] = 7
            elif action[0] == 2:
                self.state[0] = 3
            elif action[0] == 3:
                self.state[0] = 11
        elif self.state[0] == 8:
            if action[0] == 0: 
                self.state[0] = 8
            elif action[0] == 1:
                self.state[0] = 9
            elif action[0] == 2:
                self.state[0] = 4
            elif action[0] == 3:
                self.state[0] = 12
        elif self.state[0] == 9:
            if action[0] == 0: 
                self.state[0] = 8
            elif action[0] == 1:
                self.state[0] = 10
            elif action[0] == 2:
                self.state[0] = 5
            elif action[0] == 3:
                self.state[0] = 13
        elif self.state[0] == 10:
            if action[0] == 0: 
                self.state[0] = 9
            elif action[0] == 1:
                self.state[0] = 11
            elif action[0] == 2:
                self.state[0] = 6
            elif action[0] == 3:
                self.state[0] = 14
        elif self.state[0] == 11:
            if action[0] == 0: 
                self.state[0] = 10
            elif action[0] == 1:
                self.state[0] = 11
            elif action[0] == 2:
                self.state[0] = 7
            elif action[0] == 3:
                self.state[0] = 15
        elif self.state[0] == 12:
            if action[0] == 0: 
                self.state[0] = 12
            elif action[0] == 1:
                self.state[0] = 13
            elif action[0] == 2:
                self.state[0] = 8
            elif action[0] == 3:
                self.state[0] = 12
        elif self.state[0] == 13:
            if action[0] == 0: 
                self.state[0] = 12
            elif action[0] == 1:
                self.state[0] = 14
            elif action[0] == 2:
                self.state[0] = 9
            elif action[0] == 3:
                self.state[0] = 13
        elif self.state[0] == 14:
            if action[0] == 0: 
                self.state[0] = 13
            elif action[0] == 1:
                self.state[0] = 15
            elif action[0] == 2:
                self.state[0] = 10
            elif action[0] == 3:
                self.state[0] = 14
        elif self.state[0] == 15:
            if action[0] == 0: 
                self.state[0] = 14
            elif action[0] == 1:
                self.state[0] = 15
            elif action[0] == 2:
                self.state[0] = 11
            elif action[0] == 3:
                self.state[0] = 15

        if self.state[0] == 2 or self.state[0] == 4 or self.state[0] == 5 or self.state[0] == 6 or \
                self.state[0] == 10 or self.state[0] == 14 or self.state[0] == 15:
            battery += -0.05
        else:
            battery += -0.3

        if self.state[1] == 0:
            if action[1] == 0:
                self.state[1] = 0
            elif action[1] == 1:
                self.state[1] = 1
            elif action[1] == 2:
                self.state[1] = 0
            elif action[1] == 3:
                self.state[1] = 4
        elif self.state[1] == 1:
            if action[1] == 0:
                self.state[1] = 0
            elif action[1] == 1:
                self.state[1] = 2
            elif action[1] == 2:
                self.state[1] = 1
            elif action[1] == 3:
                self.state[1] = 5
        elif self.state[1] == 2:
            if action[1] == 0:
                self.state[1] = 1
            elif action[1] == 1:
                self.state[1] = 3
            elif action[1] == 2:
                self.state[1] = 2
            elif action[1] == 3:
                self.state[1] = 6
        elif self.state[1] == 3:
            if action[1] == 0:
                self.state[1] = 2
            elif action[1] == 1:
                self.state[1] = 3
            elif action[1] == 2:
                self.state[1] = 3
            elif action[1] == 3:
                self.state[1] = 7
        elif self.state[1] == 4:
            if action[1] == 0:
                self.state[1] = 4
            elif action[1] == 1:
                self.state[1] = 5
            elif action[1] == 2:
                self.state[1] = 0
            elif action[1] == 3:
                self.state[1] = 8
        elif self.state[1] == 5:
            if action[1] == 0: 
                self.state[1] = 4
            elif action[1] == 1:
                self.state[1] = 6
            elif action[1] == 2:
                self.state[1] = 1
            elif action[1] == 3:
                self.state[1] = 9
        elif self.state[1] == 6:
            if action[1] == 0: 
                self.state[1] = 5
            elif action[1] == 1:
                self.state[1] = 7
            elif action[1] == 2:
                self.state[1] = 2
            elif action[1] == 3:
                self.state[1] = 10
        elif self.state[1] == 7:
            if action[1] == 0: 
                self.state[1] = 6
            elif action[1] == 1:
                self.state[1] = 7
            elif action[1] == 2:
                self.state[1] = 3
            elif action[1] == 3:
                self.state[1] = 11
        elif self.state[1] == 8:
            if action[1] == 0: 
                self.state[1] = 8
            elif action[1] == 1:
                self.state[1] = 9
            elif action[1] == 2:
                self.state[1] = 4
            elif action[1] == 3:
                self.state[1] = 12
        elif self.state[1] == 9:
            if action[1] == 0: 
                self.state[1] = 8
            elif action[1] == 1:
                self.state[1] = 10
            elif action[1] == 2:
                self.state[1] = 5
            elif action[1] == 3:
                self.state[1] = 13
        elif self.state[1] == 10:
            if action[1] == 0: 
                self.state[1] = 9
            elif action[1] == 1:
                self.state[1] = 11
            elif action[1] == 2:
                self.state[1] = 6
            elif action[1] == 3:
                self.state[1] = 14
        elif self.state[1] == 11:
            if action[1] == 0: 
                self.state[1] = 10
            elif action[1] == 1:
                self.state[1] = 11
            elif action[1] == 2:
                self.state[1] = 7
            elif action[1] == 3:
                self.state[1] = 15
        elif self.state[1] == 12:
            if action[1] == 0: 
                self.state[1] = 12
            elif action[1] == 1:
                self.state[1] = 13
            elif action[1] == 2:
                self.state[1] = 8
            elif action[1] == 3:
                self.state[1] = 12
        elif self.state[1] == 13:
            if action[1] == 0: 
                self.state[1] = 12
            elif action[1] == 1:
                self.state[1] = 14
            elif action[1] == 2:
                self.state[1] = 9
            elif action[1] == 3:
                self.state[1] = 13
        elif self.state[1] == 14:
            if action[1] == 0: 
                self.state[1] = 13
            elif action[1] == 1:
                self.state[1] = 15
            elif action[1] == 2:
                self.state[1] = 10
            elif action[1] == 3:
                self.state[1] = 14
        elif self.state[1] == 15:
            if action[1] == 0: 
                self.state[1] = 14
            elif action[1] == 1:
                self.state[1] = 15
            elif action[1] == 2:
                self.state[1] = 11
            elif action[1] == 3:
                self.state[1] = 11

        if self.state[1] == 2 or self.state[1] == 4 or self.state[1] == 5 or self.state[1] == 6 or \
                self.state[1] == 10 or self.state[1] == 14 or self.state[1] == 15:
            battery += -0.05
        else:
            battery += -0.3

        # if requested and arrived, do served(locations: 0, 1, 8, 9, 12, 13, 3, 7, 11)
        if self.state[2] == 1:
            if self.state[0] == 0 or self.state[1] == 0:
                self.state[2] = 0
                served += 1
        if self.state[3] == 1:
            if self.state[0] == 1 or self.state[1] == 1:
                self.state[3] = 0
                served += 1
        if self.state[4] == 1:
            if self.state[0] == 8 or self.state[1] == 8:
                self.state[4] = 0
                served += 1
        if self.state[5] == 1:
            if self.state[0] == 9 or self.state[1] == 9:
                self.state[5] = 0
                served += 1
        if self.state[6] == 1:
            if self.state[0] == 12 or self.state[1] == 12:
                self.state[6] = 0
                served += 1
        if self.state[7] == 1:
            if self.state[0] == 13 or self.state[1] == 13:
                self.state[7] = 0
                served += 1
        if self.state[8] == 1:
            if self.state[0] == 3 or self.state[1] == 3:
                self.state[8] = 0
                served += 1
        if self.state[9] == 1:
            if self.state[0] == 7 or self.state[1] == 7:
                self.state[9] = 0
                served += 1
        if self.state[10] == 1:
            if self.state[0] == 11 or self.state[1] == 11:
                self.state[10] = 0
                served += 1

        if self.state[0] == self.state[1]:
            cross_paths += -0.7

        return battery, served, cross_paths

    def do_post_process(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.train:
            self.f.close()
