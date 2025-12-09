from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# model_id = "nvidia/audio-flamingo-3-hf"
# processor = AutoProcessor.from_pretrained(model_id)
# model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model_id = "nvidia/music-flamingo-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "What style is this music in?"},
#             {"type": "audio", "path": "tracks/0il663f3f63cvsRJtGmdO2.mp3"},
#         ],
#     }
# ]

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": f"""For each of the following emotions, rate the intensity (ranging from 0 to 100) you percieve in this music excerpt.
                - Wonder
                - Transcendence
                - Nostalgia
                - Tenderness
                - Peacefulness
                - Joy
                - Sadness
                - Power
                - Tension
            """},
            {"type": "audio", "path": "tracks/0il663f3f63cvsRJtGmdO2.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
