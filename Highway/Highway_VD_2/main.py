import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
import matplotlib.pyplot as plt
import copy
import json


def plot_rewards(rewards, img_name, interval=100):
    n = len(rewards)
    running_avg = np.empty(n)

    for t in range(n):
        running_avg[t] = np.mean(rewards[max(0, t - interval):(t + 1)])

    with open(img_name + ".txt", "w") as file:
        file.write(str(running_avg))

    plt.plot(rewards)
    plt.plot(running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(img_name + '.png')
    # plt.show()
    plt.close()


def main():
    timesteps = [1e4, 1.5e4, 2e4, 2.5e4, 3e4]
    learning_rates = [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
    for timestep in timesteps:
        for lr in learning_rates:
            global_name = 'ppo_highway_fast_v0' + '_' + str(timestep) + '_' + str(lr)
            print(global_name + '>>>>>>>>>>>>>>>>>>>>>>>>>>')

            train = False

            if train:
                n_cpu = 10
                batch_size = 64
                env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
                model = PPO("MlpPolicy",
                            env,
                            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                            n_steps=batch_size * 20 // n_cpu,
                            batch_size=batch_size,
                            n_epochs=10,
                            learning_rate=lr,
                            gamma=0.8,
                            verbose=1,
                            tensorboard_log="models/",
                            # device=torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
                            )
                # Train the agent
                model.learn(total_timesteps=int(timestep), progress_bar=True)
                # Save the agent
                model.save('models/' + global_name)
            else:
                f = open(global_name + '.json', 'w')
                model = PPO.load("models/" + global_name)
                env = gym.make("highway-v0", render_mode="rgb_array")
                eval_episode_rewards = []
                for _ in range(10):
                    obs, info = env.reset()
                    total_reward = 0
                    trace = {
                        'observations': [copy.deepcopy(obs)],
                        'actions': [],
                        '_safe': [True],
                        '_done': [False],
                        '_truncated': [False],
                        '_crashed': [False],
                        '_speed': [25.0],
                    }
                    done = truncated = False
                    while not (done or truncated):
                        action, _ = model.predict(obs, deterministic=True)
                        trace['actions'].append(action)
                        obs, reward, done, truncated, info = env.step(action)
                        trace['_done'].append(done)
                        trace['_truncated'].append(truncated)
                        trace['_crashed'].append(info['crashed'])
                        trace['_speed'].append(info['speed'])
                        trace['observations'].append(copy.deepcopy(obs))
                        trace['_safe'].append(not (done or truncated))
                        total_reward += reward
                        if done or truncated:
                            eval_episode_rewards.append(total_reward)
                            observations = copy.deepcopy(trace['observations'])
                            trace['observations'].clear()
                            for i in observations:
                                obj = []
                                for j in i:
                                    rep = []
                                    for k in j:
                                        rep.append(float(k))
                                    obj.append(rep)
                                trace['observations'].append(obj)
                            actions = copy.deepcopy(trace['actions'])
                            trace['actions'].clear()
                            for i in actions:
                                trace['actions'].append(int(i))
                            speed = copy.deepcopy(trace['_speed'])
                            trace['_speed'].clear()
                            for i in speed:
                                trace['_speed'].append(float(i))
                            f.write(json.dumps(trace) + '\n')
                        # env.render()
                f.close()
                plot_rewards(eval_episode_rewards, global_name)
                env.close()


if __name__ == "__main__":
    main()
