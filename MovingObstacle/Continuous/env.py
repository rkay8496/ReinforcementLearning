import gymnasium as gym
from gym import spaces
import numpy as np
import stl
import json
import os


class MovingObstacleEnv(gym.Env):
    def __init__(self, size=8.0, glitches=1):
        super(MovingObstacleEnv, self).__init__()

        self.size = size
        self.glitches = glitches
        self.glitches_counter = 0
        self.robot_turn = 0

        self.trace_name = './traces_' + str(self.size) + '_' + str(self.glitches) + '.json'
        if os.path.exists(self.trace_name):
            os.remove(self.trace_name)

        self.env_properties = [
            {

                'category': 'safety',
                'property': '',
                'quantitative': True
            },
            {
                'category': 'liveness',
                'property': '',
                'quantitative': True
            },
        ]

        self.env_specification = ''
        results = list(filter(lambda item: len(item['property']) > 0, self.env_properties))
        if len(results) > 0:
            self.env_specification += '('
            for x in results:
                self.env_specification += x['property'] + ' & '
            self.env_specification = self.env_specification[:-3]
            self.env_specification += ')'

        self.sys_properties = [
            {
                'category': 'safety',
                'property': '('
                            'G((({xr > xo + 1.0} | {xr < xo - 1.0}) & ({yr > yo + 1.0} | {yr < yo - 1.0})))'
                            ')',
                'quantitative': True
            },
            {
                'category': 'liveness',
                'property': '',
                'quantitative': True
            },
        ]

        self.sys_specification = ''
        results = list(filter(lambda item: len(item['property']) > 0, self.sys_properties))
        if len(results) > 0:
            self.sys_specification += '('
            for x in results:
                self.sys_specification += x['property'] + ' & '
            self.sys_specification = self.sys_specification[:-3]
            self.sys_specification += ')'

        self.specification = '(' + self.env_specification + ' -> ' + self.sys_specification + ')'

        # Define the action space: stay, left, right, up, down
        self.action_space = spaces.Box(low=0.0, high=size, shape=(2,), dtype=np.float32)

        # Define the observation space:
        self.observation_space = spaces.Box(low=0.0, high=size, shape=(2,), dtype=np.float32)

        # Define the initial state
        self.state, _ = self.reset()

    def reset(self, seed=5345):
        self.glitches_counter = 0
        self.robot_turn = 0

        observation = [self.size - 1.0, self.size - 1.0]
        self.trace = {
            'xo': [(0, observation[0]), (1, observation[0])],
            'yo': [(0, observation[1]), (1, observation[1])],
            'xr': [(0, 0.0)],
            'yr': [(0, 0.0)],
        }
        return observation, {}

    def step(self, action):
        if len(self.trace['xo']) <= len(self.trace['xr']):
            self.trace['xo'].append((len(self.trace['xo']), self.trace['xo'][len(self.trace['xo']) - 1][1]))
            self.trace['yo'].append((len(self.trace['yo']), self.trace['yo'][len(self.trace['yo']) - 1][1]))

        reward, terminated, info = self.check_sys(action)
        self.robot_turn += 1

        if self.robot_turn < 2:
            if self.glitches_counter < self.glitches + 1:
                is_glitch = np.random.choice([True, False])
                if is_glitch:
                    self.check_env()
                    self.glitches_counter += 1
                    self.robot_turn = 0
                else:
                    self.trace['xo'].append((len(self.trace['xo']), self.trace['xo'][len(self.trace['xo']) - 1][1]))
                    self.trace['yo'].append((len(self.trace['yo']), self.trace['yo'][len(self.trace['yo']) - 1][1]))
                    self.robot_turn += 1
        else:
            self.check_env()
            self.robot_turn = 0

        if terminated:
            f = open(self.trace_name, 'a')
            for key in self.trace.keys():
                var_trace = self.trace[key][:]
                self.trace[key].clear()
                for idx, item in enumerate(var_trace):
                    self.trace[key].append((idx, float(item[1])))
            f.write(json.dumps(self.trace) + '\n')
            f.close()
        return self.state, reward, terminated, False, {}

    def check_env(self):
        while True:
            observation = self.observation_space.sample()
            diff_xo = abs(self.trace['xo'][len(self.trace['xo']) - 1][1] - observation[0])
            diff_yo = abs(self.trace['yo'][len(self.trace['yo']) - 1][1] - observation[1])
            if not (diff_xo < 1.0 and diff_yo < 1.0):
                continue

            self.trace['xo'].append((len(self.trace['xo']), observation[0]))
            self.trace['yo'].append((len(self.trace['yo']), observation[1]))

            safety_eval = True
            if len(self.env_properties[0]['property']) > 0:
                phi = stl.parse(self.env_properties[0]['property'])
                safety_eval = True if phi(self.trace, quantitative=self.env_properties[0]['quantitative']) > 0 else False
            liveness_eval = True
            if len(self.env_properties[1]['property']) > 0:
                phi = stl.parse(self.env_properties[1]['property'])
                liveness_eval = True if phi(self.trace, quantitative=self.env_properties[1]['quantitative']) > 0 else False
            if safety_eval and liveness_eval:
                self.state = observation
                break
            elif safety_eval and not liveness_eval:
                self.state = observation
                break
            else:
                self.trace['xo'].pop(len(self.trace['xo']) - 1)
                self.trace['yo'].pop(len(self.trace['yo']) - 1)

    def check_sys(self, action):
        terminated = False
        info = {
            'satisfiable': False
        }
        reward = 0

        diff_xr = abs(self.trace['xr'][len(self.trace['xr']) - 1][1] - action[0])
        diff_yr = abs(self.trace['yr'][len(self.trace['yr']) - 1][1] - action[1])
        self.trace['xr'].append((len(self.trace['xr']), action[0]))
        self.trace['yr'].append((len(self.trace['yr']), action[1]))
        if not (diff_xr < 1.0 and diff_yr < 1.0):
            terminated = True
            info['satisfiable'] = False
            return reward, terminated, info

        safety_eval = True
        if len(self.sys_properties[0]['property']) > 0:
            phi = stl.parse(self.sys_properties[0]['property'])
            safety_eval = True if phi(self.trace, quantitative=self.sys_properties[0]['quantitative']) > 0 else False
        liveness_eval = True
        if len(self.sys_properties[1]['property']) > 0:
            phi = stl.parse(self.sys_properties[1]['property'])
            liveness_eval = True if phi(self.trace, quantitative=self.sys_properties[1]['quantitative']) > 0 else False
        if safety_eval and liveness_eval:
            reward = 1
            terminated = False
            info['satisfiable'] = True
        elif safety_eval and not liveness_eval:
            reward = 1
            terminated = False
            info['satisfiable'] = False
        else:
            terminated = True
            info['satisfiable'] = False
        return reward, terminated, info

    def do_post_process(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

