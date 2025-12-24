import numpy as np

class CrazyflieEnv:
    def __init__(self, config):
        self.config = config
        self.m = config['m']
        self.g = config['g']
        self.dt = config['dt']
        self.A = np.array([[0.0, 1.0],
                          [0.0, 0.0]], dtype=np.float32)
        self.B = np.array([[0.0],
                          [1.0 / self.m]], dtype=np.float32)

        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.reset()

    def reset(self):
        self.state = np.array([0.0, 0.0], dtype=np.float32)  #[position, velocity]
        self.integral_error = 0.0
        self.steps = 0
        return self.get_observation()

    def get_observation(self):
        position_error = self.config['target_altitude'] - self.state[0]
        return np.array([
            position_error,
            self.state[1],  # velocity
            self.integral_error
        ], dtype=np.float32)

    def step(self, action):
        thrust = self.denormalize_action(action)
        # dynamics
        control_force = thrust - self.m * self.g
        state_dot = self.A @ self.state + self.B.flatten() * control_force
        self.state = self.state + state_dot * self.dt
        position_error = self.config['target_altitude'] - self.state[0]
        self.integral_error += position_error * self.dt
        self.integral_error = np.clip(self.integral_error, -2.0, 2.0) 
        reward = self.compute_reward(position_error, self.state[1], thrust)

        self.steps += 1
        done = (self.steps >= self.config['max_steps'] or
                abs(self.state[0]) > 3.0 or
                abs(self.state[1]) > 5.0)
        info = {
            'position': self.state[0],
            'velocity': self.state[1],
            'position_error': position_error,
            'integral_error': self.integral_error,
            'thrust': thrust
        }
        return self.get_observation(), reward, done, info

    def denormalize_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        thrust = (action + 1.0) * 0.5 * (self.config['max_thrust'] - self.config['min_thrust']) + self.config['min_thrust']
        return float(thrust)

    def compute_reward(self, position_error, velocity, thrust):
        w_e = self.config['w_e']
        w_u = self.config['w_u']
        reward = -(w_e * position_error**2 + w_u * thrust**2)
        return reward
