from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from peft import PeftModel
from tqdm import tqdm
import os

# Define device and model
device = "cuda"
model_id = "Voxtral/model-training/voxtral-mini"

# Load base model and processor
processor = AutoProcessor.from_pretrained(model_id)
base_model = VoxtralForConditionalGeneration.from_pretrained(model_id,
                                                             dtype=torch.bfloat16,
                                                             device_map=device)

# Load fine-tuned version of the model
model = PeftModel.from_pretrained(base_model,
                                  "results/Voxtral-fine-tuned")

track_folder = os.listdir('mjf_tracks')

for i, track in tqdm(enumerate(track_folder), total=len(track_folder)):

    # Prompt definition
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio",
                 "path": f'mjf_tracks/{track}'},
                {"type": "text",
                 "text": """Please rate the intensity with wich you felt each of the following feelings in this music excerpt, on a scale ranging from 1 (not at all) to 5 (very much).
Just give the ratings, without justifying.
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
            ],
        }
        ]

    # Convert conversation into model inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    )
    inputs = inputs.to(device, dtype=torch.bfloat16)

    # Generate model response
    outputs = model.generate(**inputs, max_new_tokens=1024)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Save predictions to text file
    with open("voxtral_mjf_results.txt", "a+", encoding="utf-8") as f:
        f.write(f'{decoded_outputs}\n')