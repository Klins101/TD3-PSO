import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from plotting import (plot_training_curves, plot_comparison, plot_multiseed_comparison,
                      plot_step_response, plot_step_response_comparison)
from config import get_config


def plot_single_seed(seed, algorithm, output_dir='./plots'):
    config = get_config()
    base_dir = config['output_dir']
    log_path = os.path.join(base_dir, f"seed_{seed}", algorithm, "logs", "training_log.csv")

    if not os.path.exists(log_path):
        print(f"Error: Training log not found at {log_path}")
        print("Make sure you've trained the model first.")
        return
    fig = plot_training_curves(log_path, algorithm, output_dir, show=True)


def compare_algorithms(seed, output_dir='./plots'):
    config = get_config()
    base_dir = config['output_dir']
    td3_log = os.path.join(base_dir, f"seed_{seed}", "td3", "logs", "training_log.csv")
    td3pso_log = os.path.join(base_dir, f"seed_{seed}", "td3_pso", "logs", "training_log.csv")
    if not os.path.exists(td3_log):
        print(f"Error: TD3 log not found at {td3_log}")
        return
    if not os.path.exists(td3pso_log):
        print(f"Error: TD3-PSO log not found at {td3pso_log}")
        return

    print(f"Comparing TD3 vs TD3-PSO for seed {seed}...")
    fig = plot_comparison(td3_log, td3pso_log, output_dir, show=True)

def compare_multiseed(output_dir='./plots'):
    config = get_config()
    base_dir = config['output_dir']
    seeds = config['seeds']
    log_paths_dict = {
        'td3': [],
        'td3_pso': []
    }

    # Collect all log paths
    for seed in seeds:
        td3_log = os.path.join(base_dir, f"seed_{seed}", "td3", "logs", "training_log.csv")
        td3pso_log = os.path.join(base_dir, f"seed_{seed}", "td3_pso", "logs", "training_log.csv")

    setup_plot_style()

    # Load data
    df_td3 = load_training_log(td3_log_path)
    df_td3pso = load_training_log(td3pso_log_path)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot TD3
    col
        if os.path.exists(td3_log):
            log_paths_dict['td3'].append(td3_log)
        if os.path.exists(td3pso_log):
            log_paths_dict['td3_pso'].append(td3pso_log)

    if not log_paths_dict['td3'] and not log_paths_dict['td3_pso']:

        return


    fig = plot_multiseed_comparison(log_paths_dict, output_dir, show=True)



def plot_step_response_single(seed, algorithm, output_dir='./plots'):
    config = get_config()
    base_dir = config['output_dir']

    model_path = os.path.join(base_dir, f"seed_{seed}", algorithm, "models")

    if not os.path.exists(model_path):
        return


    # Get environment config from main config
    env_config = {
        'dt': config['env_dt'],
        'T_final': config['env_T_final'],
        'Q': config['env_Q'],
        'R': config['env_R'],
        'r_amp': config['env_r_amp']
    }

    fig = plot_step_response(model_path, algorithm, output_dir, show=True, env_config=env_config)



def compare_step_responses(seed, output_dir='./plots'):
    config = get_config()
    base_dir = config['output_dir']

    td3_model = os.path.join(base_dir, f"seed_{seed}", "td3", "models")
    td3pso_model = os.path.join(base_dir, f"seed_{seed}", "td3_pso", "models")

    if not os.path.exists(td3_model):
        print(f"Error: TD3 model not found at {td3_model}")
        return
    if not os.path.exists(td3pso_model):
        print(f"Error: TD3-PSO model not found at {td3pso_model}")
        return

    print(f"Comparing step responses for seed {seed}...")

    # Get environment config from main config
    env_config = {
        'dt': config['env_dt'],
        'T_final': config['env_T_final'],
        'Q': config['env_Q'],
        'R': config['env_R'],
        'r_amp': config['env_r_amp']
    }

    fig = plot_step_response_comparison(td3_model, td3pso_model, output_dir, show=True, env_config=env_config)


def main():
    parser = argparse.ArgumentParser(description='Visualize CSTR training results')

    parser.add_argument('--seed', type=int, help='Seed number')
    parser.add_argument('--algorithm', type=str, choices=['td3', 'td3_pso'],
                       help='Algorithm to plot')
    parser.add_argument('--compare', action='store_true',
                       help='Compare TD3 vs TD3-PSO for given seed')
    parser.add_argument('--multiseed', action='store_true',
                       help='Compare across all seeds')
    parser.add_argument('--step-response', action='store_true',
                       help='Plot step response (requires trained model)')
    parser.add_argument('--output', type=str, default='./plots',
                       help='Output directory for plots (default: ./plots)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    if args.multiseed:
        # Multi-seed comparison
        compare_multiseed(args.output)

    elif args.step_response and args.compare:
        # Step response comparison
        if args.seed is None:
            parser.error("--step-response --compare requires --seed")
        compare_step_responses(args.seed, args.output)

    elif args.step_response and args.seed and args.algorithm:
        # Single step response
        plot_step_response_single(args.seed, args.algorithm, args.output)

    elif args.compare:
        # Single-seed training comparison
        if args.seed is None:
            parser.error("--compare requires --seed")
        compare_algorithms(args.seed, args.output)

    elif args.seed and args.algorithm:
        # Single algorithm, single seed
        plot_single_seed(args.seed, args.algorithm, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
