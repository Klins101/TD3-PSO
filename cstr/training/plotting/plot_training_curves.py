import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_utils import setup_plot_style, save_plot, get_algorithm_color, get_algorithm_label


def load_training_log(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")
    df = pd.read_csv(log_path)
    return df


def plot_training_curves(log_path, algorithm='td3', output_dir=None, show=False):
    setup_plot_style()
    # Load data
    df = load_training_log(log_path)
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    color = get_algorithm_color(algorithm)
    label = get_algorithm_label(algorithm)

    # Reward
    ax1.plot(df['Episode'], df['Reward'], alpha=0.3, color=color, linewidth=1)
    ax1.plot(df['Episode'], df['AvgReward100'], color=color, linewidth=2, label=f'{label} (Avg 100)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'{label} Training Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # timestep Reward
    ax2.plot(df['Timestep'], df['Reward'], alpha=0.3, color=color, linewidth=1)
    ax2.plot(df['Timestep'], df['AvgReward100'], color=color, linewidth=2, label=f'{label} (Avg 100)')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'{label} Training Progress - Timestep')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.join(output_dir, f'{algorithm}_training_curves')
        save_plot(fig, base_name)
    if show:
        plt.show()

    return fig
       

def plot_comparison(td3_log_path, td3pso_log_path, output_dir=None, show=False):
    setup_plot_style()
    df_td3 = load_training_log(td3_log_path)
    df_td3pso = load_training_log(td3pso_log_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot TD3
    color_td3 = get_algorithm_color('td3')
    ax.plot(df_td3['Timestep'], df_td3['AvgReward100'],
            color=color_td3, linewidth=2, label='TD3')

    # Plot TD3-PSO
    color_td3pso = get_algorithm_color('td3_pso')
    ax.plot(df_td3pso['Timestep'], df_td3pso['AvgReward100'],
            color=color_td3pso, linewidth=2, label='TD3-PSO')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Reward (100 episodes)')
    ax.set_title('TD3 vs TD3-PSO Training Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.join(output_dir, 'training_comparison')
        save_plot(fig, base_name)
    if show:
        plt.show()
    return fig


def plot_multiseed_comparison(log_paths_dict, output_dir=None, show=False):
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for algorithm, log_paths in log_paths_dict.items():
        color = get_algorithm_color(algorithm)
        label = get_algorithm_label(algorithm)
        all_rewards = []
        max_timesteps = 0

        for log_path in log_paths:
            df = load_training_log(log_path)
            all_rewards.append(df['AvgReward100'].values)
            max_timesteps = max(max_timesteps, len(df))
        padded_rewards = []
        for rewards in all_rewards:
            padded = np.full(max_timesteps, np.nan)
            padded[:len(rewards)] = rewards
            padded_rewards.append(padded)
        rewards_array = np.array(padded_rewards)
        mean_reward = np.nanmean(rewards_array, axis=0)
        std_reward = np.nanstd(rewards_array, axis=0)
        df_first = load_training_log(log_paths[0])
        timesteps = df_first['Timestep'].values
        if len(timesteps) < max_timesteps:
            timesteps = np.arange(max_timesteps) * (timesteps[1] - timesteps[0])
        ax.plot(timesteps, mean_reward, color=color, linewidth=2, label=f'{label} (n={len(log_paths)})')
        ax.fill_between(timesteps,
                        mean_reward - std_reward,
                        mean_reward + std_reward,
                        color=color, alpha=0.2)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.join(output_dir, 'comparison')
        save_plot(fig, base_name)

    if show:
        plt.show()

    return fig
