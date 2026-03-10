from huggingface_hub import snapshot_download

snapshot_download(repo_id="nvidia/music-flamingo-hf", local_dir="./music-flamingo")
snapshot_download(repo_id="mistralai/Voxtral-Mini-3B-2507", local_dir="./voxtral-mini")
