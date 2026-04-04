import json
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, path):
        self.data = []

        # Load the dataset
        with open(path) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        turns = ex["conversation"]

        user_content = turns[0]["content"]
        assistant_content = turns[1]["content"]

        # Extract the audio file path and the text prompt from the user turn
        audio_path = next(c["path"] for c in user_content if c["type"] == "audio")
        prompt_text = next(c["text"] for c in user_content if c["type"] == "text")

        # The ground-truth response
        assistant_text = assistant_content[0]["text"] 

        return {
            "audio": audio_path,
            "prompt": prompt_text,
            "target": assistant_text
        }
    
def collate_fn(batch, processor):

    conversations = []
    targets = []

    for ex in batch:
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": ex["audio"]},
                    {"type": "text",  "text": ex["prompt"]}
                ]
            }
        ])
        targets.append(ex["target"])

    # Tokenize the user turns
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,  # appends [/INST] to prompt the assistant
        tokenize= True,
        return_dict=True
    )

    # Tokenize the target responses separately
    target_encodings = processor.tokenizer(
        targets,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    )

    # Concatenate input_ids and target_ids
    input_ids = torch.cat([inputs["input_ids"], target_encodings["input_ids"]], dim=1)
    attention_mask = torch.cat([inputs["attention_mask"], target_encodings["attention_mask"]], dim=1)

    # Mask the prompt portion from the loss
    labels = input_ids.clone()
    labels[:, :inputs["input_ids"].shape[1]] = -100

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["labels"] = labels

    return dict(inputs)
