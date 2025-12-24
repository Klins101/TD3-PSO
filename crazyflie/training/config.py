CONFIG = {
    # Environment parameters 
    'm': 0.027,                 
    'g': 9.81,                  
    'dt': 0.01,     

    # Training parameters
    'max_episodes': 500,
    'max_steps': 1000,
    'target_altitude': 1.0,

    # TD3 hyperparameters
    'state_dim': 3,             # [position_error, velocity, integral_error]
    'action_dim': 1,
    'hidden_dim': 64,
    'max_action': 1.0,

    # Action space 
    'max_thrust': 1.0,          
    'min_thrust': 0.0,          

    # Reward weights 
    'w_e': 1.0,                 
    'w_u': 0.01,                

    # lr
    'lr_actor': 3e-5,
    'lr_critic': 3e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'policy_delay': 2,

    # Exploration
    'expl_noise': 0.1,
    'expl_noise_decay': 0.9995,
    'expl_noise_min': 0.01,

    # Replay buffer
    'buffer_size': 100000,
    'batch_size': 256,
    'warmup_steps': 1000,

    # PSO parameters 
    'pso_freq': 300,            
    'pso_particles': 10,
    'pso_iterations': 3,
    'pso_w': 0.7,               
    'pso_c1': 1.5,              
    'pso_c2': 1.5,              
    'pso_noise_scale': 0.1,
    'pso_noise_decay': 0.95,
    'pso_noise_min': 0.01,

    # Saving
    'save_dir_td3': 'model',
    'save_dir_td3pso': 'model',
    'save_freq': 50,

    'seed': 2,
}


def get_config():
    return CONFIG.copy()
