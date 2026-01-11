import torch
from pathlib import Path
import os
from mellow import MellowWrapper

# setup cuda and device
cuda = torch.cuda.is_available()
device = 0 if cuda else "cpu"

# setup mellow
mellow = MellowWrapper(
                    config = "v0",
                    model = "v0",
                    device=device,
                    use_cuda=cuda,
                )

audio_path = "tracks/0dEIca2nhcxDUV8C5QkPYb.mp3"

# list of filepaths and prompts
examples = [
    [audio_path,
     audio_path,
     f"""For each of the following emotions, rate the intensity (ranging from 0 to 100) you percieve in this music excerpt.
            - Wonder
            - Transcendence
            - Nostalgia
            - Tenderness
            - Peacefulness
            - Joy
            - Sadness
            - Power
            - Tension
    """]
]

# generate response
response = mellow.generate(examples=examples, max_len=500, top_p=0.8, temperature=1.0)
print(f"\noutput: {response}")