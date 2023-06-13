import sys
import gymnasium
sys.modules["gym"] = gymnasium

from env import MovingObstacleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
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

    # for item in settings:
    size = 8
    glitches = 1
    model_name = 'ppo_moving_obstacle_' + str(size) + '_' + str(glitches)
    print(model_name + '>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # Create environment
    env = MovingObstacleEnv(size=size, glitches=glitches)

    train = False
    if train:
        # Instantiate the agent
        model = PPO("MlpPolicy", env, verbose=1, device=torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu'))
        # Train the agent and display a progress bar
        model.learn(total_timesteps=int(3e4), progress_bar=True)
        # Save the agent
        model.save(model_name)
        del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = PPO.load(model_name, env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    num_eval_episodes = 1000
    total_reward = 0
    eval_episode_rewards = []
    for i in range(num_eval_episodes):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards[0]
        if dones[0]:
            eval_episode_rewards.append(total_reward)
            total_reward = 0
        # vec_env.render()
    plot_rewards(eval_episode_rewards, model_name)


if __name__ == '__main__':
    main()
