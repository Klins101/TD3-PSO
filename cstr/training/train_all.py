import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from train_td3 import train_td3
from train_td3_pso import train_td3_pso
from config import get_config


def main():
    config = get_config()

    for seed in config['seeds']:
        print(f"Training TD3 with seed {seed}")
        train_td3(seed, config)

    for seed in config['seeds']:
        print(f"Training TD3-PSO with seed {seed}")
        train_td3_pso(seed, config)

    print("TRAINING COMPLETED!")
 


if __name__ == "__main__":
    main()
