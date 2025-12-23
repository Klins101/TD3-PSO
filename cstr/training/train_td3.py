# Train TD3 agent for CSTR control
import os
import sys
import torch
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from environment import CSTREnv
from agents.td3 import TD3
from utils import (
    set_seed,
    create_directories,
    save_model,
    init_training_log,
    log_training_step,
    evaluate_policy,
    print_training_header,
    print_progress
)


def train_td3(seed, config):
    set_seed(seed)
    models_dir, logs_dir = create_directories(
        config['output_dir'], seed, 'td3'
    )
    log_path = init_training_log(logs_dir)

    # Create environment
    env = CSTREnv(
        dt=config['env_dt'],
        T_final=config['env_T_final'],
        Q=config['env_Q'],
        R=config['env_R'],
        r_amp=config['env_r_amp']
    )

    # Initialize TD3 agent
    agent = TD3(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.action_high,
        actor_lr=config['lr_actor'],
        critic_lr=config['lr_critic'],
        gamma=config['gamma'],
        tau=config['tau'],
        policy_noise=config['policy_noise'],
        noise_clip=config['noise_clip'],
        policy_freq=config['policy_freq']
    )

    state = env.reset()
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    rewards_history = deque(maxlen=100)
    best_eval_reward = -np.inf

    for t in range(config['max_timesteps']):
        episode_timesteps += 1
        if t < config['start_timesteps']:
            action = np.random.uniform(env.action_low, env.action_high)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.select_action(state_tensor)
            action += np.random.normal(0, config['expl_noise'])
            action = np.clip(action, env.action_low, env.action_high)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state
        episode_reward += reward

        if t >= config['start_timesteps']:
            agent.update(
                batch_size=config['batch_size']
            )

        if done:
            rewards_history.append(episode_reward)
            avg_reward = np.mean(rewards_history) if rewards_history else episode_reward
            log_training_step(log_path, t + 1, episode_num + 1, episode_reward, avg_reward)

            # Print progress every 10 episodes
            if (episode_num + 1) % 10 == 0:
                print_progress(t + 1, episode_num + 1, episode_reward, avg_reward)

            state = env.reset()
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % config['eval_freq'] == 0:
            eval_reward = evaluate_policy(agent, env, n_episodes=5)
            print_progress(t + 1, episode_num, episode_reward, avg_reward, eval_reward)

            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_model(agent, models_dir)

def main():
    config = get_config()
    for seed in config['seeds']:
        print(f"Training TD3 with seed {seed}")
        train_td3(seed, config)


if __name__ == "__main__":
    main()
