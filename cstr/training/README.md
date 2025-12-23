# CSTR Training Scripts

Professional training framework for TD3 and TD3-PSO control of Continuous Stirred Tank Reactor (CSTR).

## 📁 Directory Structure

```
training/
├── agents/                      # RL agent implementations
│   ├── __init__.py
│   ├── td3.py                   # TD3 algorithm
│   └── td3_pso.py               # TD3-PSO algorithm
├── environment/                 # CSTR environment
│   ├── __init__.py
│   └── cstr_env.py              # CSTR dynamics and control
├── utils/                       # Training utilities
│   ├── __init__.py
│   └── training_utils.py        # Helper functions
├── plotting/                    # Visualization tools
│   ├── __init__.py
│   ├── plot_utils.py            # Plotting utilities
│   ├── plot_training_curves.py  # Training curve plots
│   ├── visualize_training.py    # CLI visualization tool
│   └── README.md                # Plotting documentation
├── config.py                    # Configuration file
├── train_td3.py                 # Train TD3 only
├── train_td3_pso.py             # Train TD3-PSO only
├── train_all.py                 # Train both algorithms
└── README.md                    # This file
```

## 🚀 Quick Start

### Train Both Algorithms (Recommended)

```bash
python train_all.py
```

This trains both TD3 and TD3-PSO for all seeds specified in `config.py`.

### Train Individual Algorithms

```bash
# Train TD3 only
python train_td3.py

# Train TD3-PSO only
python train_td3_pso.py
```

## ⚙️ Configuration

All hyperparameters are centralized in [config.py](config.py):

```python
CONFIG = {
    # Environment
    'env_dt': 0.1,              # Time step
    'env_T_final': 25.0,        # Episode duration
    'env_Q': 10.0,              # State weight
    'env_R': 1.0,               # Control weight

    # Training
    'max_timesteps': 200000,    # Total timesteps
    'start_timesteps': 1000,    # Random exploration

    # TD3 (matched for fair comparison)
    'lr_actor': 1e-3,
    'lr_critic': 1e-3,
    'batch_size': 256,
    'gamma': 0.99,

    # TD3-PSO specific
    'pso_freq': 10,             # Apply PSO every 10 episodes
    'pso_particles': 5,
    'pso_iterations': 3,

    # Seeds
    'seeds': [1, 2, 3, 4, 5],
}
```

To modify training, edit `config.py` and rerun the training script.

## 📊 Visualize Training

After training, visualize results using the plotting module:

```bash
# Plot single training run
python plotting/visualize_training.py --seed 1 --algorithm td3

# Compare TD3 vs TD3-PSO
python plotting/visualize_training.py --seed 1 --compare

# Multi-seed statistical comparison
python plotting/visualize_training.py --multiseed
```

See [plotting/README.md](plotting/README.md) for detailed visualization documentation.

## 📊 Output Structure

Models and logs are saved to `../data/trained_models/`:

```
../data/trained_models/
├── seed_1/
│   ├── td3/
│   │   ├── models/
│   │   │   └── best_model_actor.pth
│   │   └── logs/
│   │       └── training_log.csv
│   └── td3_pso/
│       ├── models/
│       │   └── best_model_actor.pth
│       └── logs/
│           └── training_log.csv
├── seed_2/
│   ├── td3/
│   └── td3_pso/
...
```

### Training Logs

Each `training_log.csv` contains:
- `Timestep`: Total timesteps
- `Episode`: Episode number
- `Reward`: Episode reward
- `AvgReward100`: Average reward over last 100 episodes

## 🔬 Modules

### Environment ([environment/cstr_env.py](environment/cstr_env.py))

CSTR environment with:
- **State**: `[e_int, x1, x2]` (integral error + plant states)
- **Action**: Control input `u ∈ [-5, 5]`
- **Reward**: LQ formulation `r = -(Q*e² + R*u²)`
- **Dynamics**: Linear state-space model

### Agents ([agents/](agents/))

- **td3.py**: Twin Delayed DDPG implementation
  - Actor-Critic architecture
  - Target networks with soft updates
  - Delayed policy updates
  - Target policy smoothing

- **td3_pso.py**: TD3 enhanced with Particle Swarm Optimization
  - All TD3 features
  - Periodic PSO-based actor network optimization
  - Swarm-based exploration

### Utilities ([utils/training_utils.py](utils/training_utils.py))

Helper functions:
- `set_seed()`: Set random seeds
- `create_directories()`: Create output directories
- `save_model()`: Save trained models
- `init_training_log()`: Initialize CSV logs
- `log_training_step()`: Log training data
- `evaluate_policy()`: Evaluate performance
- `print_progress()`: Display training progress

## 🎯 Training Process

1. **Initialization**
   - Set random seed for reproducibility
   - Create output directories
   - Initialize environment and agent
   - Create training log

2. **Exploration Phase** (first 1000 timesteps)
   - Random actions for initial exploration
   - Build replay buffer

3. **Training Phase**
   - Select actions using policy + exploration noise
   - Store transitions in replay buffer
   - Train agent on mini-batches
   - Update target networks

4. **PSO Enhancement** (TD3-PSO only)
   - Apply PSO every `pso_freq` episodes
   - Optimize actor network parameters
   - Improve exploration

5. **Evaluation & Saving**
   - Evaluate policy every `eval_freq` timesteps
   - Save best model based on evaluation reward

## 📈 Expected Training Time

- **TD3**: ~15-20 minutes per seed (CPU)
- **TD3-PSO**: ~20-25 minutes per seed (CPU)
- **All seeds (5×2)**: ~3 hours total (CPU)

GPU training is faster but not required for CSTR.

## 🔍 Hyperparameter Matching

**Important**: TD3 and TD3-PSO use **identical** hyperparameters except for PSO-specific parameters. This ensures fair comparison:

| Parameter | TD3 | TD3-PSO | Notes |
|-----------|-----|---------|-------|
| Actor LR | 1e-3 | 1e-3 | ✓ Matched |
| Critic LR | 1e-3 | 1e-3 | ✓ Matched |
| Batch size | 256 | 256 | ✓ Matched |
| Gamma | 0.99 | 0.99 | ✓ Matched |
| Exploration noise | 0.1 | 0.1 | ✓ Matched |
| PSO frequency | N/A | 10 episodes | TD3-PSO only |

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Run scripts from the `training/` directory:
```bash
cd cstr/training
python train_all.py
```

### Issue: Out of memory
**Solution**: Reduce `batch_size` in `config.py`

### Issue: Training unstable
**Solution**:
- Reduce learning rates (`lr_actor`, `lr_critic`)
- Increase `start_timesteps` for more exploration
- Check environment reward scaling

### Issue: PSO slowing down training
**Solution**: Reduce `pso_particles` or `pso_iterations` in `config.py`

## 📝 Citation

If you use this code, please cite our paper:
```bibtex
@article{yourpaper2025,
  title={TD3-PSO: Enhancing Deep Reinforcement Learning with Particle Swarm Optimization},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Last updated**: 2025-12-06
