from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from peft import PeftModel
from tqdm import tqdm
import os
import librosa
import soundfile as sf
import tempfile
import numpy as np

# Define device and model
device = "cuda"
model_id = "Voxtral/model-training/voxtral-mini"

# Windowing parameters
WINDOW_S   = 30   # window length in seconds
HOP_S      = 20   # hop size  = window - overlap  (30 - 10 = 20)

# Load base model and processor
processor = AutoProcessor.from_pretrained(model_id)
base_model = VoxtralForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map=device
)

# Load fine-tuned version of the model
model = PeftModel.from_pretrained(base_model, "results/Voxtral-fine-tuned")

FOLDER_PATH = "mjf_full_ex"
track_folder = os.listdir(FOLDER_PATH)

PROMPT_TEXT = """Please rate the intensity with wich you felt each of the following feelings in this music excerpt, on a scale ranging from 1 (not at all) to 5 (very much).
Just give the ratings, without justifying.
- Wonder (Filled with wonder, Dazzled, Allured, Moved)
- Transcendence (Fascinated, Overwhelmed, Feelings of transcendence and spirituality)
- Nostalgia (Nostalgic, Dreamy, Sentimental, Melancholic)
- Tenderness (Tender, Affectionate, In love, Mellowed)
- Peacefulness (Serene, Calm, Soothed, Relaxed)
- Joy (Joyful, Amused, Animated, Bouncy)
- Sadness (Sad, Sorrowful)
- Power (Strong, Triumphant, Energetic, Fiery)
- Tension (Tense, Agitated, Nervous, Irritated)"""


def iter_windows(audio: np.ndarray, sr: int, window_s: int, hop_s: int):
    """Yield (start_sec, end_sec, audio_chunk) for each sliding window."""
    window_samples = window_s * sr
    hop_samples    = hop_s    * sr
    total_samples  = len(audio)

    start = 0
    while start < total_samples:
        end   = min(start + window_samples, total_samples)
        chunk = audio[start:end]
        yield start / sr, end / sr, chunk
        if end == total_samples:   # last (possibly shorter) window — stop
            break
        start += hop_samples


def predict_window(chunk: np.ndarray, sr: int, tmp_path: str) -> str:
    """Write chunk to a temp wav file, run the model, return the decoded text."""
    sf.write(tmp_path, chunk, sr)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": tmp_path},
                {"type": "text",  "text": PROMPT_TEXT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True
    )
    inputs = inputs.to(device, dtype=torch.bfloat16)

    outputs = model.generate(**inputs, max_new_tokens=1024)
    decoded = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return decoded[0]


with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
    tmp_wav = tmp_f.name

try:
    with open("mjf_full_song.txt", "a+", encoding="utf-8") as out_f:
        for track in tqdm(track_folder, desc="Tracks"):
            track_path = os.path.join(FOLDER_PATH, track)
            audio, sr  = librosa.load(track_path, sr=None, mono=True)

            windows = list(iter_windows(audio, sr, WINDOW_S, HOP_S))
            for start_s, end_s, chunk in tqdm(
                windows, desc=f"  {track}", leave=False
            ):
                prediction = predict_window(chunk, sr, tmp_wav)
                out_f.write(
                    f"[{track}] [{start_s:.1f}s – {end_s:.1f}s]\n"
                    f"{prediction}\n\n"
                )
finally:
    os.unlink(tmp_wav)   # clean up the single reused temp file