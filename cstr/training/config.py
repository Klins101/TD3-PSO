CONFIG = {
    # Environment parameters
    'env_dt': 0.1,              
    'env_T_final': 25.0,        
    'env_Q': 10.0,              
    'env_R': 1.0,               
    'env_r_amp': 1.0,           

    # Training parameters
    'max_timesteps': 5000,    
    'start_timesteps': 1000,    
    'eval_freq': 1000,          
    'save_freq': 10000,         

    # hyperparameters 
    'lr_actor': 1e-3,          
    'lr_critic': 1e-3,          
    'batch_size': 256,          
    'gamma': 0.99,              
    'tau': 0.005,               
    'policy_noise': 0.2,      
    'noise_clip': 0.5,          
    'policy_freq': 2,           
    'expl_noise': 0.1,          

    # TD3-PSO parameters
    'pso_freq': 10,             
    'pso_particles': 5,         
    'pso_iterations': 3,        
    'pso_omega': 0.5,           
    'pso_phi_p': 1.0,           
    'pso_phi_g': 1.0,           

    # Replay buffer
    'replay_buffer_size': 1000000,  

    # seeds
    'seeds': [1, 2, 3, 4, 5],  

    # Output directories
    'output_dir': '../data/trained_models',  
}


def get_config():
    return CONFIG.copy()


