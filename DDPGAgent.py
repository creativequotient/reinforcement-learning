from ReplayBuffer import ReplayBuffer
from Networks import Actor, Critic
import torch
import os
import numpy as np
import random


class DDPGAgent(object):
    def __init__(self,
                 pi_lr,
                 q_lr,
                 gamma,
                 batch_size,
                 min_replay_size,
                 replay_buffer_size,
                 tau,
                 input_dims,
                 n_actions,
                 **kwargs):

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Neural networks
        # Policy Network
        self.pi = Actor(alpha=pi_lr, input_dims=input_dims, n_actions=n_actions).to(self.device)
        self.target_pi = Actor(alpha=pi_lr, input_dims=input_dims, n_actions=n_actions).to(self.device)
        self.pi_optimizer = self.pi.optimizer

        # Evaluation Network
        self.q = Critic(beta=q_lr, input_dims=input_dims, n_actions=n_actions).to(self.device)
        self.target_q = Critic(beta=q_lr,input_dims=input_dims, n_actions=n_actions).to(self.device)
        self.q_optimizer = self.q.optimizer

        # Sync weights
        self.sync_weights()

        # Replay buffer
        self.min_replay_size = min_replay_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Constants
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions

    def action(self, observation):
        obs = torch.from_numpy(observation).type(torch.float).to(self.device)
        obs = obs.view((-1, *obs.shape))
        return self.pi(obs)[0]

    def target_action(self, observation):
        obs = torch.from_numpy(observation).type(torch.float).to(self.device)
        obs = obs.view((-1, *obs.shape))
        return self.target_pi(obs)[0]

    def random_action(self):
        return torch.FloatTensor(self.n_actions).uniform_(-1,1).to(self.device)

    def train(self):
        # Loss statistics
        loss_results = {}
        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs, actions, rewards, new_obs, done = sample['o'], sample['a'], sample['r'], sample['o2'], sample['d']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).view((-1,1))
        new_obs = torch.from_numpy(np.stack(new_obs)).to(dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device).view((-1,1))

        self.target_pi.eval()
        self.target_q.eval()
        self.q.eval()

        # Train q network
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - done) * self.target_q(new_obs, self.target_pi(new_obs))
        predicted = self.q(obs, actions)
        loss = ((targets - predicted) ** 2).mean()
        loss_results['critic_loss'] = loss.data

        self.q_optimizer.zero_grad()
        self.q.train()
        loss.backward()
        self.q_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = False

        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs = sample['o']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        self.q.eval()
        self.pi.eval()

        # Train pi network
        predicted = self.q(obs, self.pi(obs))
        loss = -predicted.mean()
        loss_results['actor_loss'] = loss.data
        self.pi_optimizer.zero_grad()
        self.pi.train()
        loss.backward()
        self.pi_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = True

        return loss_results

    def experience(self, o, a, r, o2, d):
        self.replay_buffer.record(o, a, r, o2, d)

    def update(self):
        with torch.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(), self.target_pi.parameters()):
                target_pi_param.data = (1.0 - self.tau) * target_pi_param.data + self.tau * pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(), self.target_q.parameters()):
                target_q_param.data = (1.0 - self.tau) * target_q_param.data + self.tau * q_param.data

    def sync_weights(self):
        with torch.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(), self.target_pi.parameters()):
                target_pi_param.data = pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(), self.target_q.parameters()):
                target_q_param.data = q_param.data

    def save_agent(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        torch.save(self.target_pi.state_dict(), target_pi_path)

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        torch.save(self.target_q.state_dict(), target_q_path)

        pi_path = os.path.join(save_path, "pi_network.pth")
        torch.save(self.pi.state_dict(), pi_path)

        q_path = os.path.join(save_path, "q_network.pth")
        torch.save(self.q.state_dict(), q_path)

    def load_agent(self, save_path):
        pi_path = os.path.join(save_path, "pi_network.pth")
        self.pi.load_state_dict(torch.load(pi_path))
        self.pi.eval()

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        self.target_pi.load_state_dict(torch.load(target_pi_path))
        self.target_pi.eval()

        q_path = os.path.join(save_path, "q_network.pth")
        self.q.load_state_dict(torch.load(q_path))
        self.q.eval()

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        self.target_q.load_state_dict(torch.load(target_q_path))
        self.target_q.eval()

        self.sync_weights()

    def __str__(self):
        return str(self.pi) + str(self.q)