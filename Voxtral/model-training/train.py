import argparse
from src.utils import load_config, get_device
from src.trainer import train

def main():
    
    # Set up the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load the configuration from the YAML file.
    config  = load_config(args.config)

    # Launch the training loop
    device = get_device()
    train(config, device)

if __name__ == "__main__":
    main()