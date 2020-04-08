import numpy as np
import torch
from DDPGAgent import DDPGAgent
import argparse
import json
import gym
import random
import torch as T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default="", type=str)
    parser.add_argument('-e', '--episodes', default=100, type=int)
    parser.add_argument('-el', '--episode_length', default=1000, type=int)
    parser.add_argument('--render', dest='render', action='store_true')
    parser.set_defaults(render=False)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load json parameters
    with open(f"{args.resume}/parameters.json", "r") as f:
        parameters = json.load(f)

    env = gym.make(parameters['env'])

    T.manual_seed(args.seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    print(f"================= {'Environment Information'.center(30)} =================")
    print(f"Action space shape: {env.env.action_space.shape}")
    print(f"Action space upper bound: {env.env.action_space.high}")
    print(f"Action space lower bound: {env.env.action_space.low}")

    print(f"Observation space shape: {env.env.observation_space.shape}")
    print(f"Observation space upper bound: {np.max(env.env.observation_space.high)}")
    print(f"Observation space lower bound: {np.min(env.env.observation_space.low)}")

    n_actions = env.action_space.shape[0] if type(env.action_space) == gym.spaces.box.Box else env.action_space.n

    agent = DDPGAgent(input_dims=env.observation_space.shape,
                      n_actions=n_actions,
                      **parameters)

    agent.load_agent(f"{args.resume}/saves")

    print(f"================= {'Agent Information'.center(30)} =================")
    print(agent)

    print(f"================= {'Begin Evaluation'.center(30)} =================")

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
            new_obs, reward, done, _ = env.step(action.detach().cpu().numpy() * env.action_space.high)

            # Update obs
            obs = new_obs
            # Update rewards
            episode_reward += reward
            # End episode if done
            if done:
                break

        total_rewards += episode_reward
        episode_reward = round(episode_reward, 3)
        print(f"Episode: {episode} Evaluation reward: {episode_reward}")

    print(f"{args.episodes} episode average: {round(total_rewards / args.episodes, 3)}")