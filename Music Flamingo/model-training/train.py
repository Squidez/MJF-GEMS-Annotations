import argparse
from src.utils import load_config, get_device
from src.trainer import train

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config  = load_config(args.config)

    device = get_device()
    train(config, device)

if __name__ == "__main__":
    main()