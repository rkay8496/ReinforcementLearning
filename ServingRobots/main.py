import sys
import gymnasium
sys.modules["gym"] = gymnasium

from env import ServingRobotsEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_rewards(rewards, img_name, interval=100):
    n = len(rewards)
    running_avg = np.empty(n)

    for t in range(n):
        running_avg[t] = np.mean(rewards[max(0, t - interval):(t + 1)])

    with open('models/' + img_name + ".txt", "w") as file:
        file.write(str(running_avg))

    plt.plot(rewards)
    plt.plot(running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('models/' + img_name + '.png')
    # plt.show()
    plt.close()


def main():
    timesteps = [2e4, 2.5e4, 3e4, 3.5e4, 4e4]
    learning_rates = [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]

    for timestep in timesteps:
        for lr in learning_rates:
            global_name = 'ppo_serving_robot' + '_' + str(timestep) + '_' + str(lr)
            print(global_name + '>>>>>>>>>>>>>>>>>>>>>>>>>>')

            train = False

            # Create environment
            env = ServingRobotsEnv(train=train, global_name=global_name)

            if train:
                # Instantiate the agent
                model = PPO("MlpPolicy", env, verbose=2, learning_rate=lr,
                            tensorboard_log="models/",
                            # device=torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
                            )
                # retrain
                # model = PPO.load("models/ppo_nuclear_plant_robot", env=env)
                # Train the agent and display a progress bar
                model.learn(total_timesteps=int(timestep), progress_bar=True)
                # Save the agent
                # model.save(global_name)
                model.save('models/' + global_name)
                del model  # delete trained model to demonstrate loading
            else:
                # Load the trained agent
                # model = PPO.load(global_name, env=env)
                model = PPO.load("models/" + global_name, env=env)

                # Enjoy trained agent
                vec_env = model.get_env()
                num_eval_episodes = 100
                eval_episode_rewards = []
                for i in range(num_eval_episodes):
                    obs = vec_env.reset()
                    total_reward = 0
                    # cnt = 0
                    while True:
                        action, _states = model.predict(obs, deterministic=False)
                        obs, rewards, dones, info = vec_env.step(action)
                        total_reward += rewards[0]
                        # cnt += 1
                        if dones[0]:
                            eval_episode_rewards.append(total_reward)
                            break
                        # vec_env.render()
                plot_rewards(eval_episode_rewards, global_name + '_rewards')
                vec_env.close()


if __name__ == '__main__':
    main()
