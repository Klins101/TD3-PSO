# CSTR Control Experiments

This directory contains code and data for the Continuous Stirred Tank Reactor (CSTR) control experiments.

## 📂 Directory Structure

- **training/**: Professional training framework for TD3 and TD3-PSO
  - Modular structure with agents, environment, and utilities
  - Centralized configuration
  - Complete documentation
- **analysis/**: Performance analysis and metrics calculation
- **data/**: Trained models (5 seeds × 2 algorithms = 10 models)

## 🎯 System Description

### CSTR Dynamics
State-space representation:
```
dx/dt = Ap @ x + Bp @ u
y = Cp @ x
```

**System matrices:**
```python
Ap = [[0.0,    1.0   ],
      [-5.47, -4.719]]

Bp = [[0.0],
      [1.0]]

Cp = [3.199, -1.135]
```

### Control Configuration
- **State dimension**: 3 [e_int, x1, x2]
- **Action dimension**: 1
- **Action bounds**: [-5.0, 5.0]
- **Time step**: 0.1 s
- **Simulation time**: 25.0 s
- **Reference**: 1.0

### Cost Function
```python
cost = Q × e_int² + R × u²
```
where Q = 10.0, R = 1.0

## 📊 Performance Metrics

The analysis script computes comprehensive control metrics:

### Time-Domain Metrics
- **t_r**: Rise time (10% to 90%)
- **M_p**: Maximum overshoot (%)
- **t_s**: Settling time (2% criterion)
- **e_ss**: Steady-state error

### Integral Metrics
- **ISE**: Integral Squared Error
- **ITAE**: Integral Time Absolute Error
- **IACE**: Integral Absolute Control Effort
- **IACER**: Integral Absolute Control Effort Rate

### Control Metrics
- **u_max**: Maximum control signal

## 🚀 Usage

### Train Models

```bash
cd training

# Train both TD3 and TD3-PSO for all seeds
python train_all.py

# Or train individually
python train_td3.py       # TD3 only
python train_td3_pso.py   # TD3-PSO only
```

See [training/README.md](training/README.md) for detailed training documentation.

### Calculate Metrics

```bash
cd analysis
python analyze_control_performance.py
```

**What it does:**
1. Loads trained models from specified seeds
2. Runs evaluation episodes
3. Calculates comprehensive metrics
4. Generates comparison plots
5. Creates LaTeX tables

**Outputs:**
- `specific_seeds_metrics.txt`: Detailed metrics comparison
- `specific_seeds_metrics_table.tex`: LaTeX table for paper
- `specific_seeds_step_response_comparison.eps`: Output tracking plot
- `specific_seeds_step_response_comparison.pdf`: Control signal plot

### Configuration

Edit the seed configurations in `calculate_specific_seeds_metrics.py`:

```python
# TD3 seeds
td3_configs = [
    ("training_results_20251110_212156", 1),
    ("training_results_20251110_212156", 6),
    ("training_results_20251110_212156", 2),
]

# TD3-PSO seeds
td3pso_configs = [
    ("training_results_20251111_124430", 22),
    ("training_results_20251111_124430", 24),
    ("training_results_20251111_102200", 9),
]
```

## 📈 Expected Results

### Step Response Characteristics
- **Rise time**: ~1-3 seconds
- **Overshoot**: 5-15%
- **Settling time**: 5-10 seconds
- **Steady-state error**: < 0.01

### Performance Comparison
TD3-PSO typically shows:
- Lower ISE and ITAE (better tracking)
- Significantly lower IACER (smoother control)
- Similar or slightly better settling time
- Comparable overshoot

## 📊 Plot Features

### Output Tracking Plot
- Shows mean ± std across seeds
- Compares TD3 vs TD3-PSO vs Reference
- Displays full step response (0 to 25s)

### Control Signal Plot
- Shows control effort over time
- Highlights differences in smoothness
- Illustrates IACER metric visually

### Styling
- LaTeX-style fonts (Computer Modern)
- Orange (#ff7f0e) for TD3
- Green (#2ca02c) for TD3-PSO
- Publication-ready (300 DPI)

## 🔧 Customization

### Add New Metrics

Edit the `calculate_step_response_metrics()` function:

```python
def calculate_step_response_metrics(time, output, reference, control):
    metrics = {}

    # Add your custom metric
    metrics['your_metric'] = compute_your_metric(...)

    return metrics
```

### Change Plot Style

Modify matplotlib settings:

```python
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'lines.linewidth': 2.0,
    # ... customize as needed
})
```

## 📁 Data Organization

### Model Structure
```
training_results_YYYYMMDD_HHMMSS/
├── seed_N/
│   ├── TD3/
│   │   └── models/
│   │       └── best_model_actor.pth
│   └── TD3-PSO/
│       └── models/
│           └── best_model_actor.pth
```

### Actor Network Format
```python
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
```

## 🔍 Troubleshooting

### Model Loading Errors
- Check file paths in script
- Verify model architecture matches
- Ensure PyTorch compatibility

### Metric Calculation Issues
- Verify time series starts from t=0
- Check for NaN values in data
- Ensure steady-state is reached

### Plot Generation Problems
- Install matplotlib fonts: `apt-get install fonts-dejavu`
- Disable LaTeX if unavailable: `'text.usetex': False`
- Check file write permissions

## 📝 Notes

- Initial state recording fix ensures plots start from t=0
- Multiple seeds provide statistical significance
- LaTeX tables ready for direct paper inclusion
- EPS format preferred for publications

## 🔗 Related Files

- Main README: `../README.md`
- Crazyflie experiments: `../crazyflie/`
- All plots: `../plots/`
