import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Manager, Queue
import json


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)


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
        plt.savefig('evaluate.png')
    plt.show()


def do_post_process(trace):
    trace['x'].pop()
    trace['y'].pop()
    trace['xv'].pop()
    trace['yv'].pop()
    trace['a'].pop()
    trace['av'].pop()
    trace['l'].pop()
    trace['r'].pop()


def train(env, model, optimizer, gamma, n_steps, max_episodes):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)

        values, log_probs, rewards = [], [], []
        episode_reward = 0

        done = False
        while not done:
            action_probs, value = model(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.FloatTensor(next_state)

            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode}, Episode Reward: {episode_reward}")
        episode_rewards.append(episode_reward)

        # Calculate the discounted rewards
        R = 0
        discounted_rewards = []
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        # Normalize the discounted rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Calculate the loss
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantages = discounted_rewards - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), 'model_01.pth')
    return episode_rewards


def evaluate(f, env, model, n_episodes):
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        trace = {
            'x': [(0, state[0])],
            'y': [(0, state[1])],
            'xv': [(0, state[2])],
            'yv': [(0, state[3])],
            'a': [(0, state[4])],
            'av': [(0, state[5])],
            'l': [(0, state[6])],
            'r': [(0, state[7])],
            'not': [],
            'le': [],
            'me': [],
            're': [],
            'aux0': [(0, True)],
        }

        while not done:
            # env.render()
            state_tensor = torch.FloatTensor(state)
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs)
            trace['not'].append((len(trace['not']), True if action.item() == 0 else False))
            trace['le'].append((len(trace['le']), True if action.item() == 1 else False))
            trace['me'].append((len(trace['me']), True if action.item() == 2 else False))
            trace['re'].append((len(trace['re']), True if action.item() == 3 else False))
            trace['aux0'].append((len(trace['aux0']), True))
            next_state, reward, done, _ = env.step(action.item())

            state = next_state
            episode_reward += reward

            trace['x'].append((len(trace['x']), state[0]))
            trace['y'].append((len(trace['y']), state[1]))
            trace['xv'].append((len(trace['xv']), state[2]))
            trace['yv'].append((len(trace['yv']), state[3]))
            trace['a'].append((len(trace['a']), state[4]))
            trace['av'].append((len(trace['av']), state[5]))
            trace['l'].append((len(trace['l']), state[6]))
            trace['r'].append((len(trace['r']), state[7]))

        print(f"Episode {episode}, Episode Reward: {episode_reward}")
        episode_rewards.append(episode_reward)
        do_post_process(trace)
        for key in trace.keys():
            var_trace = trace[key][:]
            if isinstance(var_trace[0][1], np.float32):
                trace[key].clear()
                for idx, item in enumerate(var_trace):
                    trace[key].append((idx, float(item[1])))
            else:
                continue
        f.write(json.dumps(trace) + '\n')

    return episode_rewards


def main():
    env = gym.make("LunarLander-v2")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, 128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-3)
    gamma = 0.99
    n_steps = 5
    max_episodes = 10000
    # episode_rewards = train(env, model, optimizer, gamma, n_steps, max_episodes)
    # plot_rewards(episode_rewards, learn=True, interval=100)
    # Load the trained model and evaluate it
    trained_model = ActorCritic(env.observation_space.shape[0], env.action_space.n, 128)
    trained_model.load_state_dict(torch.load('model_01.pth'))
    trained_model.eval()
    f = open('traces_01.json', 'w')
    episode_rewards = evaluate(f, env, trained_model, 10)
    f.close()
    plot_rewards(episode_rewards, learn=False, interval=100)


if __name__ == '__main__':
    main()




