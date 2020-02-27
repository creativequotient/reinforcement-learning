import numpy as np
import torch
from DDPGAgent import DDPGAgent
from Noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
import os
import argparse
import json
import gym


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default="", type=str)
    parser.add_argument('-e', '--episodes', default=100, type=int)
    parser.add_argument('-el', '--episode_length', default=1000, type=int)
    parser.add_argument('--render', dest='render', action='store_true')
    parser.set_defaults(render=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = gym.make("LunarLanderContinuous-v2")

    print(f"Action space shape: {env.env.action_space.shape}")
    print(f"Action space upper bound: {env.env.action_space.high}")
    print(f"Action space lower bound: {env.env.action_space.low}")

    print(f"Observation space shape: {env.env.observation_space.shape}")
    print(f"Observation space upper bound: {np.max(env.env.observation_space.high)}")
    print(f"Observation space lower bound: {np.min(env.env.observation_space.low)}")

    # Load json parameters
    with open(f"experiments/{args.resume}/parameters.json", "r") as f:
        parameters = json.load(f)

    agent = DDPGAgent(**parameters)
    agent.load_agent(f"experiments/{args.resume}/saves")

    experiment_path = os.path.join("experiments", f"{args.resume}")

    print(agent.pi)
    print(agent.q)
    total_rewards = 0.0

    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0.0

        for step in range(args.episode_length):
            if args.render:
                env.render()

            # Get actions
            with torch.no_grad():
                action = agent.action(obs)

            # Take step in environment
            new_obs, reward, done, _ = env.step(action.detach().cpu().numpy())

            # Update obs
            obs = new_obs
            # Update rewards
            episode_reward += reward
            # End episode if done
            if done:
                break



        total_rewards += episode_reward
        episode_reward = round(episode_reward, 3)
        print(f"Episode: {episode} Average evaluation reward: {episode_reward}")

    print(f"{args.episodes} episode average: {round(total_rewards / args.episodes, 3)}")