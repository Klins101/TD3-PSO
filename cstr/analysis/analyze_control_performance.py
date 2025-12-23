"""
Calculate comprehensive control performance metrics using specific seeds
TD3: training_results_20251110_212156 (seeds 6, 7, 8)
TD3-PSO: training_results_20251111_124430 (seeds 22, 24) + training_results_20251111_102200 (seed 9)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from scipy import signal

# ===============================================================
# Actor Network Definition
# ===============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * torch.tanh(self.net(x))


# ===============================================================
# Environment
# ===============================================================
class TD3EnvCustom:
    def __init__(self, dt=0.1, T_final=25.0, Q=10.0, R=1.0, r_amp=1.0):
        self.dt = dt
        self.T_final = T_final
        self.max_steps = int(T_final / dt)

        # CSTR system matrices
        self.Ap = np.array([[0.0, 1.0],
                            [-5.47, -4.719]])
        self.Bp = np.array([[0.0],
                            [1.0]])
        self.Cp = np.array([3.199, -1.135])

        self.Q = Q
        self.R = R
        self.r_amp = r_amp

        self.action_low = -5.0
        self.action_high = 5.0

        self.max_abs_eint = 5.0
        self.max_abs_state = 5.0

        self.reset()

    def reset(self):
        self.t = 0
        self.x = np.array([0.0, 0.0])
        self.e_int = 0.0
        self.current_step = 0
        self.ref = self.r_amp
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.e_int, self.x[0], self.x[1]], dtype=np.float32)

    def step(self, action):
        u = np.clip(action, self.action_low, self.action_high)
        self.ref = self.r_amp
        y = np.dot(self.Cp, self.x)
        e = self.ref - y
        self.e_int += self.dt * e
        self.e_int = np.clip(self.e_int, -self.max_abs_eint, self.max_abs_eint)

        dx = np.dot(self.Ap, self.x) + self.Bp.flatten() * u
        self.x += self.dt * dx

        cost = self.Q * (self.e_int ** 2) + self.R * (u ** 2)
        reward = -cost
        self.current_step += 1

        done = (self.current_step >= self.max_steps or np.any(np.abs(self.x) > self.max_abs_state))
        return self._get_obs(), reward, done


# ===============================================================
# Metrics Calculation
# ===============================================================
def calculate_step_response_metrics(time, output, reference, control):
    """Calculate comprehensive step response metrics"""
    metrics = {}

    # Find steady state (last 20% of response)
    steady_idx = int(0.8 * len(output))
    steady_state = np.mean(output[steady_idx:])

    # Steady-state error
    metrics['e_ss'] = abs(reference - steady_state)

    # Rise time (10% to 90%)
    target_10 = reference * 0.1
    target_90 = reference * 0.9

    idx_10 = np.where(output >= target_10)[0]
    idx_90 = np.where(output >= target_90)[0]

    if len(idx_10) > 0 and len(idx_90) > 0:
        t_10 = time[idx_10[0]]
        t_90 = time[idx_90[0]]
        metrics['t_r'] = t_90 - t_10
    else:
        metrics['t_r'] = np.nan

    # Maximum overshoot
    peak_value = np.max(output)
    if reference > 0:
        metrics['M_p'] = ((peak_value - reference) / reference) * 100
    else:
        metrics['M_p'] = 0.0

    # Settling time (2% criterion)
    tolerance = 0.02 * abs(reference)
    settled = np.abs(output - steady_state) <= tolerance

    if np.any(settled):
        first_settled = np.where(settled)[0][0]
        # Check if it stays settled
        if np.all(settled[first_settled:]):
            metrics['t_s'] = time[first_settled]
        else:
            # Find last time it enters and stays in band
            for i in range(len(output)-1, -1, -1):
                if np.all(settled[i:]):
                    metrics['t_s'] = time[i]
                    break
            else:
                metrics['t_s'] = time[-1]
    else:
        metrics['t_s'] = time[-1]

    # Error signal
    error = reference - output

    # ISE (Integral Squared Error)
    metrics['ISE'] = np.trapz(error**2, time)

    # ITAE (Integral Time Absolute Error)
    metrics['ITAE'] = np.trapz(time * np.abs(error), time)

    # IACE (Integral Absolute Control Effort)
    metrics['IACE'] = np.trapz(np.abs(control), time)

    # IACER (Integral Absolute Control Effort Rate)
    control_rate = np.diff(control) / np.diff(time)
    time_rate = time[:-1]
    metrics['IACER'] = np.trapz(np.abs(control_rate), time_rate)

    # Maximum control
    metrics['u_max'] = np.max(np.abs(control))

    return metrics


def run_test_episode(actor, env):
    """Run a test episode and collect data"""
    obs = env.reset()
    done = False

    time_data = []
    output_data = []
    control_data = []
    reference_data = []
    error_data = []

    # Record initial state at t=0 BEFORE any action
    y_initial = env.Cp.dot(env.x)
    time_data.append(0.0)
    output_data.append(y_initial)
    control_data.append(0.0)  # No control applied yet
    reference_data.append(env.ref)
    error_data.append(env.ref - y_initial)

    step = 0
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_t).cpu().numpy()[0]

        obs, reward, done = env.step(action[0] if hasattr(action, '__len__') else action)

        # Extract data AFTER stepping
        step += 1
        e_int, x1, x2 = obs
        y = env.Cp.dot(env.x)
        r = env.ref
        e = r - y

        time_data.append(step * env.dt)
        output_data.append(y)
        # Ensure action is a scalar
        control_data.append(action[0] if hasattr(action, '__len__') else action)
        reference_data.append(r)
        error_data.append(e)

        if step >= env.max_steps:
            done = True

    return (np.array(time_data), np.array(output_data),
            np.array(control_data), np.array(reference_data),
            np.array(error_data))


# ===============================================================
# Main Analysis
# ===============================================================
def main():
    print("="*80)
    print("CSTR CONTROL PERFORMANCE ANALYSIS")
    print("="*80)

    # Define seeds (5 seeds for statistical significance)
    seeds = [1, 2, 3, 4, 5]

    # Base directory for trained models (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "data", "trained_models")

    # Environment for testing
    env = TD3EnvCustom(dt=0.1, T_final=25.0, Q=10.0, R=1.0, r_amp=1.0)

    # Storage for metrics
    td3_metrics_all = []
    td3pso_metrics_all = []

    # Storage for trajectories
    td3_trajectories = []
    td3pso_trajectories = []

    # Process TD3 seeds
    print(f"\nProcessing TD3 models (seeds {seeds})...")
    for seed in seeds:
        model_path = os.path.join(base_dir, f"seed_{seed}", "td3", "models", "best_model_actor.pth")

        if not os.path.exists(model_path):
            print(f"  Warning: Model not found at {model_path}")
            continue

        print(f"  Loading TD3 seed {seed}...")
        actor = Actor(state_dim=3, action_dim=1, max_action=5.0)
        actor.load_state_dict(torch.load(model_path, map_location='cpu'))
        actor.eval()

        # Run test episode
        time_data, output_data, control_data, reference_data, error_data = run_test_episode(actor, env)

        # Calculate metrics
        metrics = calculate_step_response_metrics(time_data, output_data, reference_data[0], control_data)
        td3_metrics_all.append(metrics)
        td3_trajectories.append((time_data, output_data, control_data, reference_data, error_data))

        print(f"    ISE: {metrics['ISE']:.4f}, IACER: {metrics['IACER']:.4f}")

    # Process TD3-PSO seeds
    print(f"\nProcessing TD3-PSO models (seeds {seeds})...")
    for seed in seeds:
        model_path = os.path.join(base_dir, f"seed_{seed}", "td3_pso", "models", "best_model_actor.pth")

        if not os.path.exists(model_path):
            print(f"  Warning: Model not found at {model_path}")
            continue

        print(f"  Loading TD3-PSO seed {seed}...")
        actor = Actor(state_dim=3, action_dim=1, max_action=5.0)
        actor.load_state_dict(torch.load(model_path, map_location='cpu'))
        actor.eval()

        # Run test episode
        time_data, output_data, control_data, reference_data, error_data = run_test_episode(actor, env)

        # Calculate metrics
        metrics = calculate_step_response_metrics(time_data, output_data, reference_data[0], control_data)
        td3pso_metrics_all.append(metrics)
        td3pso_trajectories.append((time_data, output_data, control_data, reference_data, error_data))

        print(f"    ISE: {metrics['ISE']:.4f}, IACER: {metrics['IACER']:.4f}")

    # Compute statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE CONTROL PERFORMANCE METRICS")
    print("="*80)

    metric_names = ['t_r', 'M_p', 't_s', 'e_ss', 'ISE', 'ITAE', 'IACE', 'IACER', 'u_max']

    print(f"\n{'Metric':<15} {'TD3':<25} {'TD3-PSO':<25} {'Improvement':<15}")
    print("-"*80)

    comparison_data = {}

    for metric_name in metric_names:
        td3_values = [m[metric_name] for m in td3_metrics_all if not np.isnan(m[metric_name])]
        td3pso_values = [m[metric_name] for m in td3pso_metrics_all if not np.isnan(m[metric_name])]

        if len(td3_values) > 0 and len(td3pso_values) > 0:
            td3_mean = np.mean(td3_values)
            td3_std = np.std(td3_values)
            td3pso_mean = np.mean(td3pso_values)
            td3pso_std = np.std(td3pso_values)

            # Calculate improvement (negative means worse for TD3-PSO)
            if metric_name in ['e_ss', 'ISE', 'ITAE', 'IACE', 'IACER', 'u_max', 't_r', 't_s', 'M_p']:
                # Lower is better
                improvement = ((td3_mean - td3pso_mean) / td3_mean) * 100
            else:
                improvement = ((td3pso_mean - td3_mean) / td3_mean) * 100

            comparison_data[metric_name] = {
                'td3_mean': td3_mean,
                'td3_std': td3_std,
                'td3pso_mean': td3pso_mean,
                'td3pso_std': td3pso_std,
                'improvement': improvement
            }

            print(f"{metric_name:<15} {td3_mean:>8.2f} ± {td3_std:<8.2f} {td3pso_mean:>8.2f} ± {td3pso_std:<8.2f} {improvement:>10.2f}%")

    # Save results to file (in plots directory)
    output_dir = os.path.join(script_dir, "..", "..", "..", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cstr_metrics.txt")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CSTR CONTROL PERFORMANCE METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"TD3 seeds: {td3_seeds}\n")
        f.write(f"TD3-PSO seeds: {td3pso_seeds}\n")
        f.write("\n" + "="*80 + "\n\n")

        f.write(f"{'Metric':<15} {'TD3':<25} {'TD3-PSO':<25} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")

        for metric_name in metric_names:
            if metric_name in comparison_data:
                data = comparison_data[metric_name]
                f.write(f"{metric_name:<15} {data['td3_mean']:>8.2f} ± {data['td3_std']:<8.2f} "
                       f"{data['td3pso_mean']:>8.2f} ± {data['td3pso_std']:<8.2f} "
                       f"{data['improvement']:>10.2f}%\n")

    print(f"\nResults saved to: {output_file}")

    # Generate LaTeX table
    latex_file = os.path.join(output_dir, "cstr_metrics_table.tex")
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{CSTR Control Quality (Step Response, Selected Seeds)}\n")
        f.write("\\label{tab:comprehensive_metrics_selected}\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{TD3} & \\textbf{TD3-PSO} & \\textbf{\\% Change} \\\\\n")
        f.write("\\midrule\n")

        for metric_name in metric_names:
            if metric_name in comparison_data:
                data = comparison_data[metric_name]

                # Format metric name
                if metric_name == 't_r':
                    display_name = "$t_r$ (s)"
                elif metric_name == 'M_p':
                    display_name = "$M_p$ (\\%)"
                elif metric_name == 't_s':
                    display_name = "$t_s$ (s)"
                elif metric_name == 'e_ss':
                    display_name = "$e_{\\text{ss}}$"
                elif metric_name == 'u_max':
                    display_name = "$u_{\\max}$"
                else:
                    display_name = metric_name

                # Determine which is better
                td3_better = data['improvement'] < 0

                if td3_better:
                    td3_str = f"\\mathbf{{{data['td3_mean']:.2f} \\pm {data['td3_std']:.2f}}}"
                    td3pso_str = f"{data['td3pso_mean']:.2f} \\pm {data['td3pso_std']:.2f}"
                    improvement_str = f"\\textcolor{{red!60!black}}{{{data['improvement']:.1f}}}"
                else:
                    td3_str = f"{data['td3_mean']:.2f} \\pm {data['td3_std']:.2f}"
                    td3pso_str = f"\\mathbf{{{data['td3pso_mean']:.2f} \\pm {data['td3pso_std']:.2f}}}"
                    improvement_str = f"\\textcolor{{green!60!black}}{{{data['improvement']:.1f}}}"

                # Highlight IACER row
                if metric_name == 'IACER':
                    f.write("\\rowcolor{gray!15}\n")

                f.write(f"{display_name} & ${td3_str}$ & ${td3pso_str}$ & {improvement_str} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to: {latex_file}")

    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING STEP RESPONSE COMPARISON PLOTS")
    print("="*80)

    # Enable LaTeX-style rendering (without requiring full LaTeX installation)
    plt.rcParams.update({
        'text.usetex': False,  # Don't use external LaTeX
        'mathtext.fontset': 'cm',  # Use Computer Modern font for math
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 14,
        'axes.labelsize': 18,  # Increased for axis labels
        'axes.titlesize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 18,  # Increased for tick numbers
        'ytick.labelsize': 18,  # Increased for tick numbers
        'lines.linewidth': 2.0,
    })

    # Create figure with only 2 subplots (average plots only) - vertical layout
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Average responses
    td3_outputs = np.array([traj[1] for traj in td3_trajectories])
    td3_controls = np.array([traj[2] for traj in td3_trajectories])
    td3pso_outputs = np.array([traj[1] for traj in td3pso_trajectories])
    td3pso_controls = np.array([traj[2] for traj in td3pso_trajectories])

    td3_output_mean = np.mean(td3_outputs, axis=0)
    td3_output_std = np.std(td3_outputs, axis=0)
    td3_control_mean = np.mean(td3_controls, axis=0)
    td3_control_std = np.std(td3_controls, axis=0)

    td3pso_output_mean = np.mean(td3pso_outputs, axis=0)
    td3pso_output_std = np.std(td3pso_outputs, axis=0)
    td3pso_control_mean = np.mean(td3pso_controls, axis=0)
    td3pso_control_std = np.std(td3pso_controls, axis=0)

    # Get time and reference data
    time_data, _, _, reference_data, _ = td3_trajectories[0]

    # Left plot: Output tracking
    axes[0].plot(time_data, td3_output_mean, color='#ff7f0e', linewidth=2.5, label='$TD3$')
    axes[0].fill_between(time_data,
                         td3_output_mean - td3_output_std,
                         td3_output_mean + td3_output_std,
                         color='#ff7f0e', alpha=0.2)

    axes[0].plot(time_data, td3pso_output_mean, color='#2ca02c', linewidth=2.5, label='$TD3-PSO$')
    axes[0].fill_between(time_data,
                         td3pso_output_mean - td3pso_output_std,
                         td3pso_output_mean + td3pso_output_std,
                         color='#2ca02c', alpha=0.2)

    axes[0].plot(time_data, reference_data, 'k--', linewidth=1.8, alpha=0.8, label='$r$')
    axes[0].set_ylabel(r'$Output (y)$')
    axes[0].legend(loc='best', framealpha=0.95)
    axes[0].grid(False)

    # Right plot: Control signal
    axes[1].plot(time_data, td3_control_mean, color='#ff7f0e', linewidth=2.5, label='TD3')
    axes[1].fill_between(time_data,
                         td3_control_mean - td3_control_std,
                         td3_control_mean + td3_control_std,
                         color='#ff7f0e', alpha=0.2)

    axes[1].plot(time_data, td3pso_control_mean, color='#2ca02c', linewidth=2.5, label='TD3-PSO')
    axes[1].fill_between(time_data,
                         td3pso_control_mean - td3pso_control_std,
                         td3pso_control_mean + td3pso_control_std,
                         color='#2ca02c', alpha=0.2)

    axes[1].set_xlabel(r'$Time (t)$')
    axes[1].set_ylabel(r'$Cont. Sig. (u_c)$')
    axes[1].grid(False)

    plt.tight_layout()

    # Save plots
    plot_file_eps = os.path.join(output_dir, "cstr_step_response.eps")
    plot_file_pdf = os.path.join(output_dir, "cstr_step_response.pdf")
    plt.savefig(plot_file_eps, dpi=300, bbox_inches='tight')
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"\nPlots saved to:")
    print(f"  - {plot_file_eps}")
    print(f"  - {plot_file_pdf}")

    plt.show()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
