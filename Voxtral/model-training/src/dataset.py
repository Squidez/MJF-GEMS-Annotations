import json
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, path):
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        turns = ex["conversation"]

        user_content = turns[0]["content"]
        assistant_content = turns[1]["content"]

        audio_path = next(c["path"] for c in user_content if c["type"] == "audio")
        prompt_text = next(c["text"] for c in user_content if c["type"] == "text")
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
                    {"type": "text", "text": ex["prompt"]}
                ]
            }
        ])
        targets.append(ex["target"])

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
    )

    labels = inputs["input_ids"].clone()

    inst_end_ids = processor.tokenizer.encode("[/INST]", add_special_tokens=False)
    inst_end_len = len(inst_end_ids)

    for i in range(labels.shape[0]):
        ids = inputs["input_ids"][i].tolist()
        for j in range(len(ids) - inst_end_len + 1):
            if ids[j:j + inst_end_len] == inst_end_ids:
                labels[i, :j + inst_end_len] = -100
                break

    labels[inputs["input_ids"] == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    return dict(inputs)