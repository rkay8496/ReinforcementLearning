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
        self.observation_space = spaces.MultiDiscrete([self.bucket_capacity] * self.number_buckets)

        # Define the initial state
        self.state, _ = self.reset()

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

        self.find_adjacent_indices(self.number_buckets, [0, 2, 4])

        self.update_agent(action)
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

    def find_adjacent_indices(n, indices):
        # 주어진 인덱스들을 오름차순으로 정렬합니다.
        sorted_indices = sorted(indices)

        # 인접한 인덱스를 저장할 리스트를 초기화합니다.
        adjacent_indices = []

        # 모든 인덱스에 대해 반복합니다.
        for i in range(len(sorted_indices)):
            # 현재 인덱스와 다음 인덱스를 찾습니다.
            current_index = sorted_indices[i]
            next_index = sorted_indices[(i + 1) % len(sorted_indices)]

            # 만약 현재 인덱스와 다음 인덱스가 인접하다면, 이 두 인덱스를 adjacent_indices에 추가합니다.
            if (next_index - current_index) % n == 1 or (current_index - next_index) % n == 1:
                if current_index not in adjacent_indices:
                    adjacent_indices.append(current_index)
                if next_index not in adjacent_indices:
                    adjacent_indices.append(next_index)

        # 인접한 인덱스의 리스트를 반환합니다.
        return adjacent_indices

    def update_obstacle_location(self):
        if self.state[0] > self.state[2]:
            self.state[0] = max(0, self.state[0] - 1)
        elif self.state[0] < self.state[2]:
            self.state[0] = min(self.size - 2, self.state[0] + 1)
        if self.state[1] > self.state[3]:
            self.state[1] = max(0, self.state[1] - 1)
        elif self.state[1] < self.state[3]:
            self.state[1] = min(self.size - 2, self.state[1] + 1)

    def update_agent(self, action):
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
