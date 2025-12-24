import os
import csv
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_directories(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_model(agent, filepath):
    agent.save(filepath)

def init_training_log(log_dir, algorithm='td3'):
    log_file = os.path.join(log_dir, 'training_log.csv')
    if algorithm == 'td3':
        headers = ['Episode', 'Reward', 'Length', 'AvgReward100', 'ExplNoise', 'BestReward']
    elif algorithm == 'td3pso':
        headers = ['Episode', 'Reward', 'Length', 'AvgReward100', 'ExplNoise',
                  'PSOImprovements', 'PSOPerturbations', 'BestReward']
    else:
        headers = ['Episode', 'Reward', 'Length', 'AvgReward100', 'BestReward']
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    return log_file

def log_training_step(log_file, episode, reward, length, avg_reward, expl_noise, best_reward,
                      pso_improvements=None, pso_perturbations=None):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if pso_improvements is not None and pso_perturbations is not None:
            writer.writerow([episode, reward, length, avg_reward, expl_noise,
                           pso_improvements, pso_perturbations, best_reward])
        else:
            writer.writerow([episode, reward, length, avg_reward, expl_noise, best_reward])

def evaluate_policy(agent, env, n_episodes=5):
    total_reward = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, add_noise=False)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / n_episodes


def print_training_header(config, algorithm='TD3'):
    print(f"System: Crazyflie vertical dynamics")
    if algorithm == 'TD3-PSO':
        print(f"PSO frequency: every {config['pso_freq']} episodes")


def print_progress(episode, max_episodes, reward, avg_reward, length,
                  expl_noise, best_reward, pso_info=None):
    if pso_info:
        pso_improvements, pso_perturbations = pso_info
        improvement_rate = (pso_improvements / pso_perturbations * 100) if pso_perturbations > 0 else 0
        print(f"Episode {episode}/{max_episodes} | "
              f"Reward: {reward:.2f} | "
              f"Avg100: {avg_reward:.2f} | "
              f"Length: {length} | "
              f"Noise: {expl_noise:.4f} | "
              f"PSO: {pso_improvements}/{pso_perturbations} ({improvement_rate:.0f}%) | "
              f"Best: {best_reward:.2f}")
    else:
        print(f"Episode {episode}/{max_episodes} | "
              f"Reward: {reward:.2f} | "
              f"Avg100: {avg_reward:.2f} | "
              f"Length: {length} | "
              f"Noise: {expl_noise:.4f} | "
              f"Best: {best_reward:.2f}")


def print_training_complete(best_reward, save_dir, pso_info=None):
    if pso_info:
        pso_improvements, pso_perturbations = pso_info
        improvement_rate = (pso_improvements / pso_perturbations * 100) if pso_perturbations > 0 else 0
    else:
        print(f"Best average reward (100 episodes): {best_reward:.2f}")
