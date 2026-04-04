import json
import torch
import librosa
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, json_path, sample_rate=16000):
        # Load the dataset
        with open(json_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Each sample is a multi-turn conversation
        conversation = self.data[idx]["conversation"]
        resolved = []
        
        # Conversation with decoded audio
        for turn in conversation:
            new_turn = {"role": turn["role"], "content": []}
            for item in turn["content"]:
                if item["type"] == "audio":
                    # Decode the audio file
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

    # Apply chat template and format labels for training
    inputs = processor.apply_chat_template(
            batch,
            tokenize=True,                  # Tokenizes text and extracts audio features
            add_generation_prompt=False,
            return_dict=True,
            output_labels=True,             # Automatically creates labels for training
            padding=True,
        )
    
    return inputs