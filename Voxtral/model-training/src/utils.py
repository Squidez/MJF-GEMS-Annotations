import torch
import yaml
from dataclasses import dataclass, field
from typing import List

@dataclass
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]

@dataclass
class ModelConfig:
    id: str
    use_dummy_model: bool
    lora: LoraConfig

@dataclass
class DataConfig:
    train: str
    val: str
    max_seq_len: int

@dataclass
class TrainingConfig:
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
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    lora = LoraConfig(**raw["model"].pop("lora"))
    model = ModelConfig(**raw["model"], lora=lora)
    data = DataConfig(**raw["data"])
    training = TrainingConfig(**raw["training"])

    return Config(model=model, data=data, training=training)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)
    return device

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj