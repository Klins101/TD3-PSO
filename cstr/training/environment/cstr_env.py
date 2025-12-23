# CSTR (Continuous Stirred Tank Reactor) Environment
import numpy as np

class CSTREnv:
    """
    Continuous Stirred Tank Reactor Environment

    State: [e_int, x1, x2] where e_int is integral error
    Action: Control input u
    Reward: LQ formulation r = -(Q*e^2 + R*u^2)
    """
    def __init__(self, dt=0.1, T_final=25.0, Q=10.0, R=1.0, r_amp=1.0):
        self.dt = dt
        self.T_final = T_final
        self.max_steps = int(T_final / dt)
        self.Ap = np.array([[0.0, 1.0],
                            [-5.47, -4.719]])
        self.Bp = np.array([[0.0],
                            [1.0]])
        self.Cp = np.array([3.199, -1.135])
        self.Q = Q
        self.R = R
        self.r_amp = r_amp
        # action limits
        self.action_low = -25.0
        self.action_high = 25.0
        # State limits for normalization
        self.max_abs_eint = 5.0
        self.max_abs_state = 5.0
        self.reset()

    def reset(self):
        self.t = 0
        self.x = np.array([0.0, 0.0])  
        self.e_int = 0.0  
        self.current_step = 0
        self.ref = self.r_amp  
        state = self._get_state()
        return state

    def _get_state(self):
        return np.array([self.e_int, self.x[0], self.x[1]], dtype=np.float32)

    def step(self, action):
        u = np.clip(action, self.action_low, self.action_high)
        y = self.Cp.dot(self.x)
        e = self.ref - y
        self.e_int += e * self.dt
        self.e_int = np.clip(self.e_int, -self.max_abs_eint, self.max_abs_eint)
        # State-space dynamics: dx/dt = Ap*x + Bp*u
        x_dot = self.Ap.dot(self.x) + self.Bp.flatten() * u
        # Euler integration
        self.x = self.x + x_dot * self.dt
        self.x = np.clip(self.x, -self.max_abs_state, self.max_abs_state)
        reward = -(self.Q * e**2 + self.R * u**2)
        self.t += self.dt
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        next_state = self._get_state()
        info = {
            'output': y,
            'error': e,
            'control': u,
            'reference': self.ref
        }
        return next_state, reward, done, info

    @property
    def state_dim(self):
        """State dimension"""
        return 3

    @property
    def action_dim(self):
        """Action dimension"""
        return 1
