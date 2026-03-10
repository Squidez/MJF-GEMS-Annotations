import argparse
from src.config import load_config
from src.model import build_model, build_processor
from src.dataset import build_dataloader
from src.training import train_model
from src.utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, default=None)
    args = parser.parse_args()

    config    = load_config(args.config, args.override)
    device    = get_device()

    processor = build_processor(config)
    model     = build_model(config)

    dataloader = build_dataloader(config, processor)

    train_model(model, dataloader, processor, config, device)

if __name__ == "__main__":
    main()