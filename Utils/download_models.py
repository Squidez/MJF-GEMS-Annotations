from huggingface_hub import snapshot_download
import os

# Download Music Flamingo (~ 15.4 Gb)
if os.path.isdir("./music-flamingo"):
    print('\033[31mmusic-flamingo\033[0m folder allready exists.')
else :
    snapshot_download(repo_id="nvidia/music-flamingo-hf", local_dir="./Music Flamingo/model-training/music-flamingo")

# Download Voxtral Mini (~ 17.4 Gb)
if os.path.isdir("./voxtral-mini"):
    print('\033[31mvoxtral-mini\033[0m folder allready exists.')
else :
    snapshot_download(repo_id="mistralai/Voxtral-Mini-3B-2507", local_dir="./Voxtral/model-training/model-training/voxtral-mini")
