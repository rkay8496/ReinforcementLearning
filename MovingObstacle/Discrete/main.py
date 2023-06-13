import sys
import gymnasium
sys.modules["gym"] = gymnasium

from env import MovingObstacleEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    settings = [(8, 1), (8, 2), (16, 5), (16, 6), (24, 9), (24, 10), (32, 13), (32, 14), (48, 21), (48, 22), (64, 29), (64, 30)]

    for item in settings:
        size = item[0]
        glitches = item[1]
        global_name = 'ppo_moving_obstacle_' + str(size) + '_' + str(glitches)
        print(global_name + '>>>>>>>>>>>>>>>>>>>>>>>>>>')

        train = True

        # Create environment
        env = MovingObstacleEnv(size=size, glitches=glitches, train=train, global_name=global_name)

        if train:
            # Instantiate the agent
            model = PPO("MlpPolicy", env, verbose=1,
                        device=torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu'))
            # Train the agent and display a progress bar
            model.learn(total_timesteps=int(5e4), progress_bar=True)
            # Save the agent
            model.save(global_name)
            del model  # delete trained model to demonstrate loading
        else:
            # Load the trained agent
            model = PPO.load(global_name, env=env)

            # Enjoy trained agent
            vec_env = model.get_env()
            num_eval_episodes = 100
            eval_episode_rewards = []
            for i in range(num_eval_episodes):
                obs = vec_env.reset()
                total_reward = 0
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = vec_env.step(action)
                    total_reward += rewards[0]
                    if dones[0]:
                        eval_episode_rewards.append(total_reward)
                        break
                    # vec_env.render()
            plot_rewards(eval_episode_rewards, global_name)
            vec_env.close()


if __name__ == '__main__':
    main()
