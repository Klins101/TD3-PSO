import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment import CSTREnv
from agents import TD3, TD3_PSO
from .plot_utils import setup_plot_style, save_plot, get_algorithm_color, get_algorithm_label


def load_agent(model_path, algorithm, env):
    if algorithm.lower() == 'td3':
        agent = TD3(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            max_action=env.action_high
        )
    elif algorithm.lower() == 'td3_pso':
        agent = TD3_PSO(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            max_action=env.action_high
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    # Load actor weights
    actor_path = os.path.join(model_path, "best_model_actor.pth")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Model not found: {actor_path}")

    agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
    agent.actor.eval()
    return agent


def run_step_response(agent, env, n_steps=None):
    state = env.reset()
    if n_steps is None:
        n_steps = int(env.T_final / env.dt)
    times = []
    outputs = []
    references = []
    controls = []
    errors = []

    for step in range(n_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy().flatten()[0]
        next_state, reward, done, info = env.step(action)
        times.append(env.t)
        outputs.append(info['output'])
        references.append(info['reference'])
        controls.append(action)
        errors.append(info['error'])
        state = next_state
        if done:
            break

    return {
        'time': np.array(times),
        'output': np.array(outputs),
        'reference': np.array(references),
        'control': np.array(controls),
        'error': np.array(errors)
    }


def plot_step_response(model_path, algorithm, output_dir=None, show=False, env_config=None):
    setup_plot_style()

    if env_config is None:
        env_config = {}
    env = CSTREnv(**env_config)
    agent = load_agent(model_path, algorithm, env)
    results = run_step_response(agent, env)
    fig, (ax1, ax2, ax3) = plt.subplots(2, 1, figsize=(10, 10))

    color = get_algorithm_color(algorithm)
    label = get_algorithm_label(algorithm)

    # Plot 1: y vs r
    ax1.plot(results['time'], results['reference'], 'k--', linewidth=2, label='Reference')
    ax1.plot(results['time'], results['output'], color=color, linewidth=2, label=f'{label} Output')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output')
    ax1.set_title(f'{label} Step Response - Output Tracking')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(results['time'], results['control'], color=color, linewidth=2)
    ax2.axhline(y=env.action_high, color='r', linestyle='--', linewidth=1, alpha=0.5, label='r')
    ax2.axhline(y=env.action_low, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Signal')
    ax2.set_title(f'{label} Step Response - Control Signal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.join(output_dir, f'{algorithm}_step_response')
        save_plot(fig, base_name)
    if show:
        plt.show()
    return fig


def plot_step_response_comparison(td3_model_path, td3pso_model_path, output_dir=None, show=False, env_config=None):
    setup_plot_style()
    # environment
    if env_config is None:
        env_config = {}
    env = CSTREnv(**env_config)
    # agents
    td3_agent = load_agent(td3_model_path, 'td3', env)
    td3pso_agent = load_agent(td3pso_model_path, 'td3_pso', env)
    td3_results = run_step_response(td3_agent, env)
    env = CSTREnv(**env_config)
    td3pso_results = run_step_response(td3pso_agent, env)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    color_td3 = get_algorithm_color('td3')
    color_td3pso = get_algorithm_color('td3_pso')

    # Plot 
    ax1.plot(td3_results['time'], td3_results['reference'], 'k--', linewidth=2, label='Reference')
    ax1.plot(td3_results['time'], td3_results['output'], color=color_td3, linewidth=2, label='TD3')
    ax1.plot(td3pso_results['time'], td3pso_results['output'], color=color_td3pso, linewidth=2, label='TD3-PSO')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output')
    ax1.set_title('Step Response Comparison - Output Tracking')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Tracking Error
    ax2.plot(td3_results['time'], td3_results['error'], color=color_td3, linewidth=2, label='TD3')
    ax2.plot(td3pso_results['time'], td3pso_results['error'], color=color_td3pso, linewidth=2, label='TD3-PSO')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tracking Error')
    ax2.set_title('Step Response Comparison - Tracking Error')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Control Signal
    ax3.plot(td3_results['time'], td3_results['control'], color=color_td3, linewidth=2, label='TD3')
    ax3.plot(td3pso_results['time'], td3pso_results['control'], color=color_td3pso, linewidth=2, label='TD3-PSO')
    ax3.axhline(y=env.action_high, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Limits')
    ax3.axhline(y=env.action_low, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Signal')
    ax3.set_title('Step Response Comparison - Control Signal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.join(output_dir, 'step_response_comparison')
        save_plot(fig, base_name)
    if show:
        plt.show()
    return fig
