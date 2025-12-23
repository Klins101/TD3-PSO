import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)

    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        return self.q1(x)

class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=10000, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha  
        self.beta = beta    
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.episode_return = np.zeros((max_size, 1))  
        self.td_error = np.ones((max_size, 1)) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, episode_return=0.0):
        self.state[self.ptr] = state
        if np.isscalar(action):
            self.action[self.ptr] = [action]
        else:
            self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.episode_return[self.ptr] = episode_return
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_td_errors(self, indices, td_errors):
        for i, idx in enumerate(indices):
            self.td_error[idx] = abs(td_errors[i]) + 1e-6

    def sample(self, batch_size):
        # Compute return-weighted priorities
        if self.size > 0:
            # Normalize episode returns
            returns = self.episode_return[:self.size]
            normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            # Combine TD-error and return-based priorities
            td_priorities = np.power(self.td_error[:self.size] + 1e-6, self.alpha)
            return_weights = np.exp(self.beta * normalized_returns)
            priorities = td_priorities * return_weights
            probabilities = priorities / priorities.sum()
            # Sample based on priorities
            ind = np.random.choice(self.size, size=batch_size, p=probabilities.flatten())
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
            ind
        )

# TD3-PSO Agent
class TD3_PSO:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=64,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        buffer_size=10000,
        # PSO-specific parameters
        pso_freq=10000,  
        pso_particles=3,  
        pso_eta=0.1,  
        # Adaptive noise parameters
        noise_base=0.1,
        noise_min=0.05,
        noise_max=0.3,
        noise_adapt_k=0.5,
        adapt_freq=10  
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, buffer_size)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.pso_freq = pso_freq
        self.pso_particles = pso_particles
        self.pso_eta = pso_eta
        self.noise_std = noise_base
        self.noise_base = noise_base
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_adapt_k = noise_adapt_k
        self.adapt_freq = adapt_freq
        self.total_it = 0
        self.episode_count = 0
        self.recent_returns = []
        self.return_ema = 0.0  
        self.prev_return_ema = 0.0

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise

        actions_clipped = np.clip(action, -self.max_action, self.max_action)
        return actions_clipped[0] if len(actions_clipped) == 1 else actions_clipped

    def update_adaptive_noise(self, episode_return):
        self.recent_returns.append(episode_return)
        self.episode_count += 1

        if self.episode_count % self.adapt_freq == 0 and len(self.recent_returns) >= 2:
            self.prev_return_ema = self.return_ema
            alpha_ema = 0.1
            self.return_ema = alpha_ema * episode_return + (1 - alpha_ema) * self.return_ema
            delta_R = self.return_ema - self.prev_return_ema
            # Adaptive function: f(dR) = -tanh(dR / |R|)
            # When performance improves: f < 0 -> decrease noise (exploit)
            # When performance worsens: f > 0 -> increase noise (explore)
            if abs(self.return_ema) > 1e-6:
                f_delta = -np.tanh(delta_R / abs(self.return_ema))
            else:
                f_delta = 0.0
            # Update noise: delta_t = clip(delta_base * (1 + k * f(dR)), delta_min, delta_max)
            self.noise_std = np.clip(
                self.noise_base * (1 + self.noise_adapt_k * f_delta),
                self.noise_min,
                self.noise_max
            )

    def pso_perturbation(self, env):
        # Create particle population around current actor
        particles = []
        fitnesses = []
        particles.append(copy.deepcopy(self.actor.state_dict()))
        # Create perturbed particles
        for _ in range(self.pso_particles - 1):
            particle = copy.deepcopy(self.actor.state_dict())
            # Add small random perturbations to parameters
            for key in particle:
                if 'weight' in key or 'bias' in key:
                    noise = torch.randn_like(particle[key]) * 0.01
                    particle[key] = particle[key] + noise
            particles.append(particle)
        # evaluate each particle with short rollouts
        for particle_params in particles:
            self.actor.load_state_dict(particle_params)
            # Short evaluation rollout
            state = env.reset()
            total_reward = 0
            for _ in range(10):
                action = self.select_action(state, add_noise=False)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            fitnesses.append(total_reward)
        # Find best particle
        best_idx = np.argmax(fitnesses)
        best_particle = particles[best_idx]
        # Update main actor toward best particle
        current_params = self.actor.state_dict()
        for key in current_params:
            if 'weight' in key or 'bias' in key:
                current_params[key] = current_params[key] + self.pso_eta * (best_particle[key] - current_params[key])
        self.actor.load_state_dict(current_params)
        self.actor_target = copy.deepcopy(self.actor)

        return max(fitnesses), fitnesses

    def update(self, batch_size):
        self.total_it += 1
        # Sample replay buffer with return-weighted priorities
        state, action, next_state, reward, done, indices = self.replay_buffer.sample(batch_size)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            # Compute the target Q value using the minimum of two critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute TD errors for priority update
        td_errors = (current_Q1 - target_Q).detach().cpu().numpy()
        self.replay_buffer.update_td_errors(indices, td_errors)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Delayed policy updates
        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.item()
            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # Return losses and Q-value statistics
        with torch.no_grad():
            q_mean = (current_Q1.mean().item() + current_Q2.mean().item()) / 2
            q_std = (current_Q1.std().item() + current_Q2.std().item()) / 2
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_value,
            'q_mean': q_mean,
            'q_std': q_std,
            'noise_std': self.noise_std
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
        np.savez(filename + "_pso_state.npz",
                 noise_std=self.noise_std,
                 return_ema=self.return_ema,
                 prev_return_ema=self.prev_return_ema,
                 episode_count=self.episode_count)
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        try:
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth", map_location=self.device))
            self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth", map_location=self.device))
            pso_state = np.load(filename + "_pso_state.npz")
            self.noise_std = float(pso_state['noise_std'])
            self.return_ema = float(pso_state['return_ema'])
            self.prev_return_ema = float(pso_state['prev_return_ema'])
            self.episode_count = int(pso_state['episode_count'])
        except:
            print("Could not load optimizer or PSO states, using fresh optimizers")
