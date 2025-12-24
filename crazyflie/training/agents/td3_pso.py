# TD3-PSO for crazyflie altitude control
import numpy as np
import torch
from .td3 import Actor, Critic, ReplayBuffer

class TD3_PSO:
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

        # PSO tracking
        self.pso_noise_scale = config['pso_noise_scale']
        self.pso_improvements = 0
        self.pso_perturbations = 0

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

        # Update critic
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.config['policy_noise']).clamp(
                -self.config['noise_clip'], self.config['noise_clip'])
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.config['max_action'], self.config['max_action'])
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.config['gamma'] * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + \
                     torch.nn.functional.mse_loss(current_Q2, target_Q)
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

    def apply_pso(self, env, num_eval_episodes=3):
        current_params = self.get_actor_params_vector()
        best_global_params = current_params.copy()
        best_global_fitness = self.evaluate_fitness(env, num_eval_episodes)
        # Initialize swarm
        particles = []
        velocities = []
        personal_best_params = []
        personal_best_fitness = []
        for i in range(self.config['pso_particles']):
            particle = current_params + np.random.normal(0, self.pso_noise_scale, current_params.shape)
            particles.append(particle)
            velocities.append(np.zeros_like(particle))
            personal_best_params.append(particle.copy())
            self.set_actor_params_vector(particle)
            fitness = self.evaluate_fitness(env, num_eval_episodes)
            personal_best_fitness.append(fitness)
            if fitness > best_global_fitness:
                best_global_fitness = fitness
                best_global_params = particle.copy()

        for iteration in range(self.config['pso_iterations']):
            for i in range(self.config['pso_particles']):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.config['pso_w'] * velocities[i] +
                               self.config['pso_c1'] * r1 * (personal_best_params[i] - particles[i]) +
                               self.config['pso_c2'] * r2 * (best_global_params - particles[i]))
                # Update position
                particles[i] = particles[i] + velocities[i]
                # Evaluate
                self.set_actor_params_vector(particles[i])
                fitness = self.evaluate_fitness(env, num_eval_episodes)
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_params[i] = particles[i].copy()
                # Update global best
                if fitness > best_global_fitness:
                    best_global_fitness = fitness
                    best_global_params = particles[i].copy()

        # Apply best parameters
        self.set_actor_params_vector(best_global_params)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.pso_perturbations += 1
        if best_global_fitness > self.evaluate_fitness(env, num_eval_episodes):
            self.pso_improvements += 1
        self.pso_noise_scale = max(self.pso_noise_scale * self.config['pso_noise_decay'],
                                   self.config['pso_noise_min'])

        improvement_rate = (self.pso_improvements / self.pso_perturbations * 100) if self.pso_perturbations > 0 else 0
        print(f"    Best fitness: {best_global_fitness:.2f}")
        print(f"    Improvements: {self.pso_improvements}/{self.pso_perturbations} ({improvement_rate:.1f}%)")
        print(f"    PSO noise scale: {self.pso_noise_scale:.4f}")

    def get_actor_params_vector(self):
        params = []
        for param in self.actor.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_actor_params_vector(self, params_vector):
        idx = 0
        for param in self.actor.parameters():
            param_length = param.numel()
            param.data = torch.FloatTensor(
                params_vector[idx:idx+param_length].reshape(param.shape)
            ).to(self.device)
            idx += param_length

    def evaluate_fitness(self, env, num_episodes=3):
        total_reward = 0
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            for _ in range(self.config['max_steps']):
                action = self.select_action(state, add_noise=False)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward
        return total_reward / num_episodes

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
            'pso_improvements': self.pso_improvements,
            'pso_perturbations': self.pso_perturbations,
            'pso_noise_scale': self.pso_noise_scale,
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
        self.pso_improvements = checkpoint.get('pso_improvements', 0)
        self.pso_perturbations = checkpoint.get('pso_perturbations', 0)
        self.pso_noise_scale = checkpoint.get('pso_noise_scale', self.config['pso_noise_scale'])
