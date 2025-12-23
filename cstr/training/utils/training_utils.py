import os
import csv
import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_directories(base_dir, seed, algorithm):
    seed_dir = os.path.join(base_dir, f"seed_{seed}", algorithm)
    models_dir = os.path.join(seed_dir, "models")
    logs_dir = os.path.join(seed_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return models_dir, logs_dir


def save_model(agent, models_dir, filename="best_model_actor.pth"):
    save_path = os.path.join(models_dir, filename)
    torch.save(agent.actor.state_dict(), save_path)
    print(f"Model saved: {save_path}")


def init_training_log(logs_dir, filename="training_log.csv"):
    log_path = os.path.join(logs_dir, filename)
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestep', 'Episode', 'Reward', 'AvgReward100'])
    return log_path


def log_training_step(log_path, timestep, episode, reward, avg_reward):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestep, episode, reward, avg_reward])


def evaluate_policy(agent, env, n_episodes=5):
    agent.actor.eval()
    total_reward = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Select action without exploration noise
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.actor(state_tensor).cpu().numpy().flatten()
            state, reward, done, _ = env.step(action[0])
            episode_reward += reward
        total_reward += episode_reward
    avg_reward = total_reward / n_episodes
    agent.actor.train()

    return avg_reward


def print_training_header(algorithm, seed, max_timesteps):
    print(f"CSTR TRAINING - {algorithm.upper()}")
    print(f"Seed: {seed}")
    print(f"Algorithm: {algorithm}")


def print_progress(timestep, episode, reward, avg_reward, eval_reward=None):
    msg = f"Timestep {timestep:7d} | Episode {episode:4d} | " \
          f"Reward: {reward:8.2f} | Avg100: {avg_reward:8.2f}"
    if eval_reward is not None:
        msg += f" | Eval: {eval_reward:8.2f}"

    print(msg)
