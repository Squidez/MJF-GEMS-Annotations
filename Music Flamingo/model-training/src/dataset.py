import json
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

class EmotionMusicDataset(Dataset):
    def __init__(self, json_path, sample_rate=16000):
        with open(json_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[idx]["conversation"]

        resolved = []
        for turn in conversation:
            new_turn = {"role": turn["role"], "content": []}
            for item in turn["content"]:
                if item["type"] == "audio":

                    audio, _ = librosa.load(item["path"], sr=self.sample_rate, mono=True)
                    new_turn["content"].append({
                        "type": "audio",
                        "array": audio,
                        "sampling_rate": self.sample_rate
                    })
                else:
                    new_turn["content"].append(item)
            resolved.append(new_turn)

        return resolved
    
class CollateFunction:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        inputs = self.processor.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            output_labels=True,
            padding=True,
        )
        return inputs

def build_dataloader(config, processor):
    collate_fn      = CollateFunction(processor)

    train_dataset   = EmotionMusicDataset(config.train_json_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    val_dataset     = EmotionMusicDataset(config.val_json_path)
    val_dataloader  = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    return train_dataloader, val_dataloader