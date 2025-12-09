import librosa
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, TrainingArguments, Trainer
import pandas as pd
from tqdm import tqdm

model_id = "nvidia/music-flamingo-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    offload_buffers=True)
df = pd.read_csv('test_30.csv')

for i, row in tqdm(df.iterrows(), total=30):

    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": """For each of the following emotions, rate the intensity, with a score ranging from 0 to 100, you percieve in this music excerpt.
- Wonder (Filled with wonder, Dazzled, Allured, Moved)
- Transcendence (Fascinated, Overwhelmed, Feelings of transcendence and spirituality)
- Nostalgia (Nostalgic, Dreamy, Sentimental, Melancholic)
- Tenderness (Tender, Affectionate, In love, Mellowed)
- Peacefulness (Serene, Calm, Soothed, Relaxed)
- Joy (Joyful, Amused, Animated, Bouncy)
- Sadness (Sad, Sorrowful)
- Power (Strong, Triumphant, Energetic, Fiery)
- Tension (Tense, Agitated, Nervous, Irritated)
                    """},
                    {"type": "audio",
                     "path": row['path']},
                ],
            },
        ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256)

    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    with open("flamingo_results.txt", "a+", encoding="utf-8") as f:
        f.write(str(decoded_outputs) + '\n')