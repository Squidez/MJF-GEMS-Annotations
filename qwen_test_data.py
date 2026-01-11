import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, TrainingArguments, Trainer
import pandas as pd
from tqdm import tqdm



processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

df = pd.read_csv('test_30.csv')
# df = df.iloc[:12]
for i, row in tqdm(df.iterrows(), total=30):

    conversation = [
            {
                "role": "user",
                "content": [
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
                    {"type": "audio",
                     "path": row['Path']},
                ],
            },
        ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios =  []
    
    audios.append(librosa.load(row['Path'], 
                               sr=processor.feature_extractor.sampling_rate)[0])
    # for message in conversation:
    #     if isinstance(message["content"], list):
    #         for ele in message["content"]:
    #             if ele["type"] == "audio":
    #                 audios.append(
    #                     librosa.load(
    #                         ele['path'], 
    #                         sr=processor.feature_extractor.sampling_rate)[0]
    #                 )

    inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_new_tokens=500)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    with open("qwen_results.txt", "a+", encoding="utf-8") as f:
        f.write(str(response) + "\n")