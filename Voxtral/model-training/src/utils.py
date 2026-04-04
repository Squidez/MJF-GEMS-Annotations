import torch
import yaml
from dataclasses import dataclass, field
from typing import List

# Dataclasses Configuration
# Each dataclass maps directly to a section of the YAML config file.

@dataclass
class LoraConfig:
    """LoRA fine-tuning hyperparameters."""
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]

@dataclass
class ModelConfig:
    """Model identity and adapter settings."""
    id: str
    use_dummy_model: bool
    lora: LoraConfig

@dataclass
class DataConfig:
    """Paths and constraints for the training and validation datasets."""
    train: str
    val: str
    max_seq_len: int

@dataclass
class TrainingConfig:
    """Hyperparameters and settings that control the training loop."""
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    early_stopping_patience: int
    output_dir: str
    seed: int

@dataclass
class Config:
    """Root configuration object that aggregates all sub-configs."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

def load_config(path: str) -> Config:

    # Parse the YAML config file and return config
    with open(path) as f:
        raw = yaml.safe_load(f)

    lora = LoraConfig(**raw["model"].pop("lora"))
    model = ModelConfig(**raw["model"], lora=lora)
    data = DataConfig(**raw["data"])
    training = TrainingConfig(**raw["training"])

    return Config(model=model, data=data, training=training)

def get_device() -> torch.device:

    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)
    return device