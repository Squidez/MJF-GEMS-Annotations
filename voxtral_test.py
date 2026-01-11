from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

device = "cuda"
model_id = "mistralai/Voxtral-Mini-3B-2507"
processor = AutoProcessor.from_pretrained(model_id)
model = VoxtralForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio",
             "path": "tracks/0dEIca2nhcxDUV8C5QkPYb.mp3"},
            {"type": "text",
             "text": """Please rate the intenity with wich you felt each of the following feelings in this music excerpt, on a scale ranging from 1 (not at all) to 5 (very much).
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

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
