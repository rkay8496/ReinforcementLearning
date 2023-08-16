import json

import gymnasium as gym
from gym import spaces
import numpy as np
import stl
import copy


class CinderellaEnv(gym.Env):
    def __init__(self, number_buckets=5, bucket_capacity=6, added_units=4, adjacent_buckets=2,
                 train=True, global_name='model'):
        super(CinderellaEnv, self).__init__()

        self.number_buckets = number_buckets
        self.bucket_capacity = bucket_capacity
        self.added_units = added_units
        self.adjacent_buckets = adjacent_buckets
        self.train = train
        self.global_name = global_name

        if not self.train:
            self.f = open(self.global_name + '.json', 'w')

        # Define the action space:
        self.action_space = spaces.MultiDiscrete([2] * self.number_buckets)

        # Define the observation space:
        self.observation_space = spaces.MultiDiscrete([self.bucket_capacity + self.added_units + 1] *
                                                      self.number_buckets)

        # Define the initial state
        self.state, _ = self.reset()

        if not self.train:
            self.trace = {
                'observations': [],
                'actions': [],
                '_safe': [],
            }

    def reset(self, seed=5345):
        while True:
            self.state = self.observation_space.sample()
            if sum(self.state) == self.added_units:
                break
        if not self.train:
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
        if not self.train:
            self.trace['actions'].append(action)

        terminated = False
        info = {
            'satisfiable': False
        }
        reward = 0

        terminated = self.update_buckets(action)
        if not self.train:
            self.trace['observations'].append(copy.deepcopy(self.state))
        info['satisfiable'] = not terminated
        if not terminated:
            if not self.train:
                self.trace['_safe'].append(not terminated)
            reward = 1

        if not self.train and terminated:
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
                        obj = []
                        for elem in item:
                            obj.append(int(elem))
                        self.trace[key].append(obj)
            self.trace['_safe'].append(not terminated)
            self.f.write(json.dumps(self.trace) + '\n')

        return self.state, reward, terminated, False, info

    def update_buckets(self, action):
        if 1 not in action:
            return True

        # A 배열에서 1인 값들의 인덱스를 찾습니다.
        indices_of_ones = [i for i, value in enumerate(action) if value == 1]

        # 1인 값들의 인덱스가 j개 이하이고 서로 인접한지 확인합니다.
        if len(indices_of_ones) > self.adjacent_buckets:
            return True

        if indices_of_ones:
            for i in range(len(indices_of_ones) - 1):
                if (indices_of_ones[i + 1] - indices_of_ones[i]) % len(action) != 1:
                    return True
            # 배열의 첫 번째 원소와 마지막 원소가 인접한지 확인합니다.
            if (indices_of_ones[0] - indices_of_ones[-1]) % len(action) != 1 and len(indices_of_ones) > 1:
                return True

        # A 배열에서 1인 값들의 인덱스가 B 배열에서 동일한 인덱스의 값이 0보다 큰지 확인합니다.
        for index in indices_of_ones:
            if self.state[index] <= 0:
                return True

        # A 배열에서 1인 값들의 인덱스에 해당하는 B 배열의 값들을 0으로 바꿉니다.
        for index in indices_of_ones:
            self.state[index] = 0

        while True:
            added = self.observation_space.sample()
            if sum(added) == self.added_units:
                self.state = [a + b for a, b in zip(self.state, added)]
                break

        for level in self.state:
            if level > self.bucket_capacity:
                return True
        return False

    def do_post_process(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.train:
            self.f.close()
