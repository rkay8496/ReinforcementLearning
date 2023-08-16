import json

import gymnasium as gym
from gym import spaces
import numpy as np
import copy


class NuclearPlantRobotEnv(gym.Env):
    def __init__(self, train=True, global_name='model'):
        super(NuclearPlantRobotEnv, self).__init__()

        self.size = 6
        self.train = train
        self.global_name = global_name

        if not self.train:
            self.f = open('models/' + self.global_name + '.json', 'w')

        self.action_space = spaces.MultiDiscrete([22, 22])

        self.observation_space = spaces.MultiDiscrete([self.size, self.size, 4, 4, 4, 4, 4, 4])

        self.state, _ = self.reset()

        self.trace = {
            'observations': [],
            'actions': [],
            '_safe': []
        }

    def reset(self, seed=5345):
        # self.state = np.array([0, 5, 1, 0, 0, 0, 0, 1])
        self.state = self.observation_space.sample()
        rooms = [0, 0, 0, 0, 0, 0]
        rooms[self.state[0]] += 1
        rooms[self.state[1]] += 1
        self.state[2:8] = rooms
        self.trace = {
            'observations': [
                copy.deepcopy(self.state)
            ],
            'actions': [

            ],
            '_safe': [True]
        }
        return self.state, {}

    def step(self, action):
        info = {}
        reward = 0

        self.trace['actions'].append(action)

        battery, exposure = self.update_observation(action)
        self.trace['observations'].append(copy.deepcopy(self.state))

        reward += battery + exposure

        all_visit = all(x > 2 for x in self.state[2:8])
        if all_visit:
            reward += 100
        terminated = all_visit

        self.trace['_safe'].append(not terminated)

        info['battery'] = battery
        info['exposure'] = exposure
        info['all_visit'] = all_visit
        info['finish'] = terminated
        if terminated:
            if not self.train:
                for key in self.trace.keys():
                    if key == 'observations' or key == 'actions':
                        for idx, item in enumerate(self.trace[key]):
                            self.trace[key][idx] = item.tolist()
                self.f.write(json.dumps(self.trace) + '\n')

        return self.state, reward, terminated, False, info

    def update_observation(self, action):
        battery, exposure = 0, 0
        battery, exposure = self.update_robot1(action, battery, exposure)
        battery, exposure = self.update_robot2(action, battery, exposure)
        return battery, exposure

    def update_robot2(self, action, battery, exposure):
        if self.state[1] == 0 and action[1] == 0:
            self.state[1] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -3
        elif self.state[1] == 0 and action[1] == 1:
            self.state[1] = 2
            self.state[4] = self.state[4] + 1 if self.state[4] < 3 else 3
            battery += -1.5
        elif self.state[1] == 0 and action[1] == 2:
            self.state[1] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -2
        elif self.state[1] == 1 and action[1] == 3:
            self.state[1] = 0
            self.state[2] = self.state[2] + 1 if self.state[2] < 3 else 3
            battery += -3
        elif self.state[1] == 1 and action[1] == 4:
            self.state[1] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -1.5
        elif self.state[1] == 1 and action[1] == 5:
            self.state[1] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -3.5
        elif self.state[1] == 1 and action[1] == 6:
            self.state[1] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -7
            exposure += -0.04
        elif self.state[1] == 2 and action[1] == 7:
            self.state[1] = 0
            self.state[2] = self.state[2] + 1 if self.state[2] < 3 else 3
            battery += -1.5
        elif self.state[1] == 2 and action[1] == 8:
            self.state[1] = 5
            self.state[7] = self.state[7] + 1 if self.state[7] < 3 else 3
            battery += -5.5
        elif self.state[1] == 3 and action[1] == 9:
            self.state[1] = 0
            self.state[2] = self.state[2] + 1 if self.state[2] < 3 else 3
            battery += -2
        elif self.state[1] == 3 and action[1] == 10:
            self.state[1] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -1.5
        elif self.state[1] == 3 and action[1] == 11:
            self.state[1] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -3.5
        elif self.state[1] == 3 and action[1] == 12:
            self.state[1] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -7
            exposure += -0.04
        elif self.state[1] == 3 and action[1] == 13:
            self.state[1] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -8
            exposure += -0.04
        elif self.state[1] == 3 and action[1] == 14:
            self.state[1] = 5
            self.state[7] = self.state[7] + 1 if self.state[7] < 3 else 3
            battery += -8
        elif self.state[1] == 4 and action[1] == 15:
            self.state[1] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -7
        elif self.state[1] == 4 and action[1] == 16:
            self.state[1] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -7
        elif self.state[1] == 4 and action[1] == 17:
            self.state[1] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -8
        elif self.state[1] == 4 and action[1] == 18:
            self.state[1] = 5
            self.state[7] = self.state[7] + 1 if self.state[7] < 3 else 3
            battery += -6
        elif self.state[1] == 5 and action[1] == 19:
            self.state[1] = 2
            self.state[4] = self.state[4] + 1 if self.state[4] < 3 else 3
            battery += -5.5
        elif self.state[1] == 5 and action[1] == 20:
            self.state[1] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -8
        elif self.state[1] == 5 and action[1] == 21:
            self.state[1] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -6
            exposure += -0.04
        else:
            battery += 0
            exposure += 0
        return battery, exposure

    def update_robot1(self, action, battery, exposure):
        if self.state[0] == 0 and action[0] == 0:
            self.state[0] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -3
        elif self.state[0] == 0 and action[0] == 1:
            self.state[0] = 2
            self.state[4] = self.state[4] + 1 if self.state[4] < 3 else 3
            battery += -1.5
        elif self.state[0] == 0 and action[0] == 2:
            self.state[0] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -2
        elif self.state[0] == 1 and action[0] == 3:
            self.state[0] = 0
            self.state[2] = self.state[2] + 1 if self.state[2] < 3 else 3
            battery += -3
        elif self.state[0] == 1 and action[0] == 4:
            self.state[0] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -1.5
        elif self.state[0] == 1 and action[0] == 5:
            self.state[0] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -3.5
        elif self.state[0] == 1 and action[0] == 6:
            self.state[0] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -7
            exposure += -0.04
        elif self.state[0] == 2 and action[0] == 7:
            self.state[0] = 0
            self.state[2] = self.state[2] + 1 if self.state[2] < 3 else 3
            battery += -1.5
        elif self.state[0] == 2 and action[0] == 8:
            self.state[0] = 5
            self.state[7] = self.state[7] + 1 if self.state[7] < 3 else 3
            battery += -5.5
        elif self.state[0] == 3 and action[0] == 9:
            self.state[0] = 0
            self.state[2] = self.state[2] + 1 if self.state[2] < 3 else 3
            battery += -2
        elif self.state[0] == 3 and action[0] == 10:
            self.state[0] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -1.5
        elif self.state[0] == 3 and action[0] == 11:
            self.state[0] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -3.5
        elif self.state[0] == 3 and action[0] == 12:
            self.state[0] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -7
            exposure += -0.04
        elif self.state[0] == 3 and action[0] == 13:
            self.state[0] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -8
            exposure += -0.04
        elif self.state[0] == 3 and action[0] == 14:
            self.state[0] = 5
            self.state[7] = self.state[7] + 1 if self.state[7] < 3 else 3
            battery += -8
        elif self.state[0] == 4 and action[0] == 15:
            self.state[0] = 1
            self.state[3] = self.state[3] + 1 if self.state[3] < 3 else 3
            battery += -7
        elif self.state[0] == 4 and action[0] == 16:
            self.state[0] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -7
        elif self.state[0] == 4 and action[0] == 17:
            self.state[0] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -8
        elif self.state[0] == 4 and action[0] == 18:
            self.state[0] = 5
            self.state[7] = self.state[7] + 1 if self.state[7] < 3 else 3
            battery += -6
        elif self.state[0] == 5 and action[0] == 19:
            self.state[0] = 2
            self.state[4] = self.state[4] + 1 if self.state[4] < 3 else 3
            battery += -5.5
        elif self.state[0] == 5 and action[0] == 20:
            self.state[0] = 3
            self.state[5] = self.state[5] + 1 if self.state[5] < 3 else 3
            battery += -8
        elif self.state[0] == 5 and action[0] == 21:
            self.state[0] = 4
            self.state[6] = self.state[6] + 1 if self.state[6] < 3 else 3
            battery += -6
            exposure += -0.04
        else:
            battery += 0
            exposure += 0
        return battery, exposure

    def do_post_process(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.train:
            self.f.close()
