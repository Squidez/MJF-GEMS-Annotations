from dataclasses import dataclass
import yaml

@dataclass
class TrainingConfig:
    model_id: str = "music-flamingo"
    json_path: str = "./conversations.jsonl"
    output_dir: str = "./checkpoints"
    epochs: int = 3
    batch_size: int = 2
    lr: float = 2e-5
    num_workers: int = 0
    use_dummy_model: bool = False


def load_config(base_path: str, override_path: str = None) -> TrainingConfig:
    # Load base yaml
    with open(base_path) as f:
        data = yaml.safe_load(f)

    # Merge override yaml on top if provided
    if override_path:
        with open(override_path) as f:
            overrides = yaml.safe_load(f)
        data.update(overrides)

    return TrainingConfig(**data)