import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from environment import CrazyflieEnv
from agents import TD3
from utils import (
    set_seed,
    create_directories,
    save_model,
    init_training_log,
    log_training_step,
    print_training_header,
    print_progress,
    print_training_complete
)


def plot_training_curves(rewards, lengths, save_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Smooth rewards
    window = 50
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        episodes_smooth = np.arange(window-1, len(rewards))
    else:
        rewards_smooth = rewards
        episodes_smooth = np.arange(len(rewards))

    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    axes[0].plot(episodes_smooth, rewards_smooth, color='blue', linewidth=2, label='Smoothed (50ep)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('TD3 Training - Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot episode lengths
    axes[1].plot(lengths, color='green', alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Lengths')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"Training curves saved to: {save_dir}/training_curves.png")
    plt.close()

def train():
    config = get_config()
    # Set random seed
    set_seed(config['seed'])
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_training_header(config, algorithm='TD3')
    save_dir = create_directories(config['save_dir_td3'])
    env = CrazyflieEnv(config)
    agent = TD3(config, device)
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    log_file = init_training_log(save_dir, algorithm='td3')

    for episode in range(config['max_episodes']):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        for step in range(config['max_steps']):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            if len(agent.replay_buffer) >= config['warmup_steps']:
                critic_loss, actor_loss = agent.update()
            state = next_state
            episode_reward += reward
            episode_length += 1
            if done:
                break
        agent.decay_exploration()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        if avg_reward_100 > best_reward:
            best_reward = avg_reward_100
            save_model(agent, os.path.join(save_dir, 'best_model.pth'))
        if (episode + 1) % config['save_freq'] == 0:
            save_model(agent, os.path.join(save_dir, f'checkpoint_ep{episode+1}.pth'))
        log_training_step(log_file, episode+1, episode_reward, episode_length,
                         avg_reward_100, agent.expl_noise, best_reward)
        if (episode + 1) % 10 == 0:
            print_progress(episode+1, config['max_episodes'], episode_reward, avg_reward_100,
                          episode_length, agent.expl_noise, best_reward)
    save_model(agent, os.path.join(save_dir, 'final_model.pth'))
    print_training_complete(best_reward, save_dir)
    plot_training_curves(episode_rewards, episode_lengths, save_dir)
    return agent

if __name__ == "__main__":
    agent = train()
