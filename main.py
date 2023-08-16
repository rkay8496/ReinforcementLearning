import gym

# 환경 초기화
env = gym.make('CartPole-v1')

# 환경의 렌더링을 위한 초기화
env.reset()

# 몇 초간 환경 시각화
for _ in range(100):
    env.render() # 환경을 시각화합니다.
    action = env.action_space.sample() # 무작위 행동을 선택합니다.
    env.step(action) # 선택한 행동을 환경에 적용합니다.

env.close() # 환경 종료