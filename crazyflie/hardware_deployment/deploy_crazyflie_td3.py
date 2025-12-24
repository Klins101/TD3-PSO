# deploy trained TD3 controller to real Crazyflie hardware
import numpy as np
import time
import logging
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    HAS_CFLIB = True
except ImportError:
    print("WARNING: cflib not installed. Install with: pip install cflib")
    HAS_CFLIB = False
logging.basicConfig(level=logging.INFO)

CONFIG = {
    #hardware connection
    'uri': 'radio://0/80/2M/E7E7E7E7E7',
    # Flight parameters
    'target_altitude': 1.0,
    'flight_duration': 15.0,
    'dt': 0.01,  # 100 Hz 

    # Model parameters
    'model_path': 'models/best_td3_model.pth',

    # System parameters 
    'm': 0.027,
    'g': 9.81,
    'max_thrust': 1.0,
    'min_thrust': 0.0,

    # PWM conversion 
    'hover_pwm': 42000,
    'thrust_gain': 41000,
    'pwm_min': 10000,
    'pwm_max': 66000,

    # Network parameters
    'state_dim': 3,
    'action_dim': 1,
    'hidden_dim': 64,
    'max_action': 1.0,

    # Safety params
    'max_altitude': 2.0,
    'max_velocity': 3.0,
    'stabilization_time': 3.0,
    'emergency_land_on_error': True,
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# NN
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class TD3CrazyflieController:
    def __init__(self, scf, config):
        self.scf = scf
        self.cf = scf.cf
        self.config = config
        # Control state
        self.current_altitude = 0.0
        self.current_velocity = 0.0
        self.integral_error = 0.0
        self.start_time = None
        # Data logging
        self.log_data = {
            'time': [],
            'altitude': [],
            'velocity': [],
            'reference': [],
            'position_error': [],
            'integral_error': [],
            'action': [],
            'thrust': [],
            'pwm': [],
        }
        # Load TD3 model
        self.actor = self._load_model()
        # Setup logging
        self._setup_logging()

    def _load_model(self):
        if not os.path.exists(self.config['model_path']):
            raise FileNotFoundError(f"Model not found: {self.config['model_path']}")

        actor = Actor(
            self.config['state_dim'],
            self.config['action_dim'],
            self.config['hidden_dim'],
            self.config['max_action']
        ).to(device)
        checkpoint = torch.load(self.config['model_path'], map_location=device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        return actor

    def _setup_logging(self):
        log_conf = LogConfig(name='StateEstimate', period_in_ms=10)
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stateEstimate.vz', 'float')
        try:
            self.cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(self._log_callback)
            log_conf.start()
        except Exception as e:
            print(f"Could not add log config: {e}")

    def _log_callback(self, timestamp, data, logconf):
        self.current_altitude = data['stateEstimate.z']
        self.current_velocity = data['stateEstimate.vz']

    def compute_control(self, reference):
        position_error = reference - self.current_altitude
        self.integral_error += position_error * self.config['dt']
        self.integral_error = np.clip(self.integral_error, -2.0, 2.0) 
        state = np.array([
            position_error,
            self.current_velocity,
            self.integral_error
        ], dtype=np.float32)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state_tensor).cpu().data.numpy().flatten()[0]
        return action, position_error

    def action_to_thrust(self, action):
        action = np.clip(action, -1.0, 1.0)
        thrust = (action + 1.0) * 0.5 * (self.config['max_thrust'] - self.config['min_thrust']) + self.config['min_thrust']
        return float(thrust)

    def thrust_to_pwm(self, thrust):
        #thrust to control 
        control_force = thrust - self.config['m'] * self.config['g']
        pwm = self.config['hover_pwm'] + control_force * self.config['thrust_gain']
        pwm = np.clip(pwm, self.config['pwm_min'], self.config['pwm_max'])
        return int(pwm)

    def check_safety(self):
        if abs(self.current_altitude) > self.config['max_altitude']:
            return False
        if abs(self.current_velocity) > self.config['max_velocity']:
            return False

        return True

    def run_test_flight(self):
        try:
            # Run flight
            self.start_time = time.time()
            self._altitude_control()
            # Land
            self._land()
            # done
            self._save_data()
            self._print_summary()
            self._plot_results()

        except KeyboardInterrupt:
            self._emergency_land()
            self._save_data()
            self._plot_results()

        except Exception as e:
            if self.config['emergency_land_on_error']:
                self._emergency_land()
            raise

    def _altitude_control(self):
        reference = self.config['target_altitude']
        flight_duration = self.config['flight_duration']

        for i in range(10):
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.02)
        loop_start = time.time()
        iteration = 0

        while (time.time() - loop_start) < flight_duration:
            t = time.time() - loop_start
            # Safety check
            if not self.check_safety():
                break
            # Compute control action
            action, position_error = self.compute_control(reference)
            thrust = self.action_to_thrust(action)
            pwm_cmd = self.thrust_to_pwm(thrust)
            self.cf.commander.send_setpoint(0, 0, 0, pwm_cmd)
            # save data
            self.log_data['time'].append(t)
            self.log_data['altitude'].append(self.current_altitude)
            self.log_data['velocity'].append(self.current_velocity)
            self.log_data['reference'].append(reference)
            self.log_data['position_error'].append(position_error)
            self.log_data['integral_error'].append(self.integral_error)
            self.log_data['action'].append(action)
            self.log_data['thrust'].append(thrust)
            self.log_data['pwm'].append(pwm_cmd)

            iteration += 1
            time.sleep(self.config['dt'])

    def _land(self):
        self.cf.commander.send_stop_setpoint()
        time.sleep(0.1)

    def _emergency_land(self):
        self.cf.commander.send_stop_setpoint()
        time.sleep(0.1)

    def _save_data(self):
        filename = f"crazyflie_td3_data.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (s)', 'Altitude (m)', 'Velocity (m/s)',
                           'Reference (m)', 'Position Error (m)', 'Integral Error',
                           'Action', 'Thrust (N)', 'PWM'])

            for i in range(len(self.log_data['time'])):
                writer.writerow([
                    self.log_data['time'][i],
                    self.log_data['altitude'][i],
                    self.log_data['velocity'][i],
                    self.log_data['reference'][i],
                    self.log_data['position_error'][i],
                    self.log_data['integral_error'][i],
                    self.log_data['action'][i],
                    self.log_data['thrust'][i],
                    self.log_data['pwm'][i],
                ])
        print(f"\n flight data saved to: {filename}")

    def _print_summary(self):
        if len(self.log_data['time']) == 0:
            return
        altitude_data = np.array(self.log_data['altitude'])
        print(f"  Mean altitude: {np.mean(altitude_data):.3f} m")
        print(f"  Target altitude: {self.config['target_altitude']:.3f} m")
        print(f"  Min altitude: {np.min(altitude_data):.3f} m")
        print(f"  Max altitude: {np.max(altitude_data):.3f} m")

    def _plot_results(self):
        if len(self.log_data['time']) == 0:
            return

        times = np.array(self.log_data['time'])
        altitudes = np.array(self.log_data['altitude'])
        velocities = np.array(self.log_data['velocity'])
        references = np.array(self.log_data['reference'])
        errors = np.array(self.log_data['position_error'])
        actions = np.array(self.log_data['action'])
        thrusts = np.array(self.log_data['thrust'])
        pwms = np.array(self.log_data['pwm'])

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('TD3 Crazyflie Hardware Flight Results', fontsize=16, fontweight='bold')
        # Altitude tracking
        axes[0, 0].plot(times, altitudes, 'b-', linewidth=2, label='Actual')
        axes[0, 0].plot(times, references, 'r--', linewidth=2, label='Reference')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].set_title('Altitude Tracking')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Tracking error
        axes[0, 1].plot(times, errors, 'r-', linewidth=2)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position Error (m)')
        axes[0, 1].set_title('Tracking Error')
        axes[0, 1].grid(True, alpha=0.3)

        # Velocity
        axes[0, 2].plot(times, velocities, 'g-', linewidth=2)
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Velocity (m/s)')
        axes[0, 2].set_title('Vertical Velocity')
        axes[0, 2].grid(True, alpha=0.3)

        # Control action
        axes[1, 0].plot(times, actions, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Action (normalized)')
        axes[1, 0].set_title('TD3 Control Action')
        axes[1, 0].grid(True, alpha=0.3)

        # Thrust
        axes[1, 1].plot(times, thrusts, 'orange', linewidth=2)
        hover_thrust = self.config['m'] * self.config['g']
        axes[1, 1].axhline(y=hover_thrust, color='k', linestyle='--', alpha=0.3,
                          label=f'Hover ({hover_thrust:.3f} N)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Thrust (N)')
        axes[1, 1].set_title('Applied Thrust')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # PWM command
        axes[1, 2].plot(times, pwms, 'brown', linewidth=2)
        axes[1, 2].axhline(y=self.config['hover_pwm'], color='k', linestyle='--',
                          alpha=0.3, label=f"Hover ({self.config['hover_pwm']})")
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('PWM Command')
        axes[1, 2].set_title('Motor PWM')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filename = f"crazyflie_td3_plot.png"
        plt.savefig(plot_filename, dpi=350, bbox_inches='tight')
        print(f"plot saved to: {plot_filename}")
        plt.show()


def main():
    if not HAS_CFLIB:
        return

    if not os.path.exists(CONFIG['model_path']):
        return

    cflib.crtp.init_drivers()

    try:
        with SyncCrazyflie(CONFIG['uri'], cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected to Crazyflie!")
            time.sleep(CONFIG['stabilization_time'])

            # controller
            controller = TD3CrazyflieController(scf, CONFIG)
            #confirmation
            input("\nPress ENTER to start the flight (Ctrl+C to cancel)")

            # flight
            controller.run_test_flight()

    except KeyboardInterrupt:
        print("\n Flight cancelled by user")

    except Exception as e:
        print(f"\n error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
