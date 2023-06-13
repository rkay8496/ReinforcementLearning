import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from NuclearReactorTrip.A2C.Mitigated.env import NuclearReactorTripSystem


# 신경망 정의 (Actor-Critic)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


# Plot rewards function
def plot_rewards(rewards, learn=True, interval=100):
    n = len(rewards)
    running_avg = np.empty(n)

    for t in range(n):
        running_avg[t] = np.mean(rewards[max(0, t - interval):(t + 1)])

    plt.plot(rewards)
    plt.plot(running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    if learn:
        plt.savefig('learn.png')
    else:
        plt.savefig('evaluate_8_1.png')
    plt.show()


def process_state(state):
    state_vec = np.zeros(4)
    state_vec[state] = 1
    return torch.FloatTensor(state_vec).unsqueeze(0)


def choose_action(action_probs):
    action_dim = action_probs.shape[-1]
    return np.random.choice(np.arange(action_dim), p=action_probs.detach().numpy().squeeze())


def compute_loss(action_probs, state_value, action, advantage):
    actor_loss = -torch.log(action_probs.squeeze()[action]) * advantage.detach()
    critic_loss = advantage.pow(2)
    return actor_loss + critic_loss


def train_one_episode(env, model, optimizer, gamma, state_dim):
    state = env.reset()
    total_reward = 0
    cnt = 0

    while cnt < 100:
        state_tensor = process_state(state)
        action_probs, state_value = model(state_tensor)
        action = choose_action(action_probs)

        next_state, reward, done, _ = env.step(action)
        cnt += 1

        next_state_tensor = process_state(next_state)
        _, next_state_value = model(next_state_tensor)

        # Advantage 계산
        target = reward + gamma * next_state_value * (1 - int(done))
        advantage = target - state_value

        # 손실 함수 계산 및 모델 업데이트
        loss = compute_loss(action_probs, state_value, action, advantage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def evaluate_one_episode(env, model, state_dim):
    state = env.reset()
    total_reward = 0
    cnt = 0

    while cnt < 100:
        state_tensor = process_state(state)
        action_probs, _ = model(state_tensor)
        action = choose_action(action_probs)

        next_state, reward, done, _ = env.step(action)
        cnt += 1

        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def main():
    # 환경 설정 및 하이퍼파라미터 정의
    env = NuclearReactorTripSystem()
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    num_episodes = 2000
    gamma = 0.99
    learning_rate = 0.00006

    # 모델 및 최적화기 초기화
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # A2C 학습 루프
    episode_rewards = []
    for episode in range(num_episodes):
        total_reward = train_one_episode(env, model, optimizer, gamma, state_dim)
        episode_rewards.append(total_reward)
        print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))

    # Save the trained model
    torch.save(model.state_dict(), 'model_01.pth')

    # Plot training rewards
    plot_rewards(episode_rewards, learn=True)

    # Load the trained model and evaluate it
    trained_model = ActorCritic(state_dim, action_dim)
    trained_model.load_state_dict(torch.load('model_01.pth'))
    trained_model.eval()

    # Evaluation loop
    f = open('traces_01.json', 'w')
    num_eval_episodes = 200
    eval_episode_rewards = []
    for episode in range(num_eval_episodes):
        total_reward = evaluate_one_episode(env, trained_model, state_dim)
        eval_episode_rewards.append(total_reward)
        print("Eval Episode: {}, Total Reward: {}".format(episode + 1, total_reward))
        f.write(json.dumps(env.trace) + '\n')

    # Plot evaluation rewards
    plot_rewards(eval_episode_rewards, learn=False)

    f.close()
    env.close()


if __name__ == "__main__":
    main()
