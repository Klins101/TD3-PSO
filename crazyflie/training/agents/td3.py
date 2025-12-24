# TD3 for Crazyflie altitude control
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32).reshape(-1, 1),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32).reshape(-1, 1)
        )
    def __len__(self):
        return len(self.buffer)


class TD3:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(config['state_dim'], config['action_dim'],
                          config['hidden_dim'], config['max_action']).to(self.device)
        self.actor_target = Actor(config['state_dim'], config['action_dim'],
                                  config['hidden_dim'], config['max_action']).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(config['state_dim'], config['action_dim'],
                            config['hidden_dim']).to(self.device)
        self.critic_target = Critic(config['state_dim'], config['action_dim'],
                                    config['hidden_dim']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.total_steps = 0
        self.expl_noise = config['expl_noise']

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, self.expl_noise * self.config['max_action'],
                                    size=self.config['action_dim'])
            action = action + noise
            action = np.clip(action, -self.config['max_action'], self.config['max_action'])
        return action

    def update(self):
        if len(self.replay_buffer) < self.config['batch_size']:
            return None, None
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config['batch_size'])
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # update critic
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.config['policy_noise']).clamp(
                -self.config['noise_clip'], self.config['noise_clip'])
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.config['max_action'], self.config['max_action'])
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.config['gamma'] * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = None
        if self.total_steps % self.config['policy_delay'] == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)
            actor_loss = actor_loss.item()
        self.total_steps += 1
        return critic_loss.item(), actor_loss

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data +
                                   (1 - self.config['tau']) * target_param.data)

    def decay_exploration(self):
        self.expl_noise = max(self.expl_noise * self.config['expl_noise_decay'],
                             self.config['expl_noise_min'])

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'expl_noise': self.expl_noise,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.expl_noise = checkpoint['expl_noise']
