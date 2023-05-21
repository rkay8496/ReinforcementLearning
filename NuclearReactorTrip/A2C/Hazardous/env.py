import gym
from gym import spaces

class NuclearReactorTripSystem(gym.Env):
    def __init__(self):
        super(NuclearReactorTripSystem, self).__init__()

        # 이산적인 상태 정의 (4개의 상태: 정상(0), 경고(1), 위험(2), 트립(3))
        self.observation_space = spaces.Discrete(4)

        # 이산적인 행동 정의 (4개의 행동: 대기(0), 확인(1), 조치(2), 트립(3))
        self.action_space = spaces.Discrete(4)

        self.reset()

    def step(self, action):
        self.trace['action'].append((len(self.trace['action']), int(action)))

        done = False
        reward = 0

        # 이산적인 행동에 따른 상태 변화 정의
        if action == 0:  # 대기
            self._natural_progression()
        elif action == 1:  # 확인
            if self.state != 0:  # 정상 상태가 아니면
                reward = 1
            self._natural_progression()
        elif action == 2:  # 조치
            if self.state == 1:  # 경고 상태라면
                self.state = 0  # 정상 상태로
                reward = 2
            elif self.state == 2:  # 위험 상태라면
                self.state = 1  # 경고 상태로
                reward = 1
            else:
                reward = -1
                done = True
        elif action == 3:  # 트립
            if self.state == 2:  # 위험 상태라면
                self.state = 3  # 트립 상태로
                reward = 3
            else:
                reward = -1
                done = True

        self.trace['state'].append((len(self.trace['state']), int(self.state)))
        self.trace['_safe'].append((len(self.trace['_safe']), not done))

        return self.state, reward, done, {}

    def _natural_progression(self):
        if self.state != 3:  # 이미 트립 상태가 아니라면
            self.state += 1  # 상태를 1단계 위험하게

    def reset(self):
        self.trace = {
            'state': [(0, 0)],
            'action': [],
            '_safe': [(0, True)]
        }
        self.state = 0  # 정상 상태로 초기화
        return self.state

    def render(self, mode='human'):
        state_dict = {0: '정상', 1: '경고', 2: '위험', 3: '트립'}
        print(f"State: {state_dict[self.state]}")
