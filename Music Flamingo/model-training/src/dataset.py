import json
import torch
import librosa
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
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
    
def collate_fn(batch, processor):

    inputs = processor.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            output_labels=True,
            padding=True,
        )
    
    return inputs