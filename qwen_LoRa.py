import os
import json
import torch
import librosa
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_pt_utils import LabelSmoother


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
DATA_FILE = "conversations.jsonl"
OUTPUT_DIR = "./qwen2_audio_lora_finetuned"
USE_QLORA = False  # Set False for normal LoRA


# ============================================================
# LOAD DATASET
# ============================================================
dataset = load_dataset("json", data_files=DATA_FILE)


# ============================================================
# LOAD PROCESSOR
# ============================================================
processor = AutoProcessor.from_pretrained(MODEL_NAME)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
def preprocess(batch):
    conversation = batch["conversation"]

    # Convert conversation → chat template
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=False
    )

    # Load all audio segments used by user messages
    audios = []
    for msg in conversation:
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "audio":
                    wav, _ = librosa.load(
                        part["path"],
                        sr=processor.feature_extractor.sampling_rate
                    )
                    audios.append(wav)

    # Tokenize for text + audio
    proc = processor(
        text=text,
        audio=audios if len(audios) > 0 else None,
        return_tensors="pt",
        padding=True
    )

    input_ids = proc.input_ids[0]
    labels = input_ids.clone()

    # Mask out user tokens before assistant response
    mask = True
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<assistant>")

    for i, tok in enumerate(input_ids):
        if tok == assistant_token_id:
            mask = False
        if mask:
            labels[i] = -100

    batch["input_ids"] = input_ids
    batch["labels"] = labels
    batch["attention_mask"] = proc.attention_mask[0]

    # Audio fields
    if "audio_values" in proc:
        batch["audio_values"] = proc.audio_values
        batch["audio_attention_mask"] = proc.audio_attention_mask
    
    return batch


processed = dataset.map(preprocess)


# ============================================================
# LOAD MODEL
# ============================================================
if USE_QLORA:
    print("\n### Using QLoRA 4-bit ###\n")
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )
else:
    print("\n### Using standard LoRA ###\n")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


# ============================================================
# LoRA CONFIGURATION
# ============================================================
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "down_proj", "up_proj"
    ],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# ============================================================
# TRAINING ARGS
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_steps=50,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    # evaluation_strategy="no",
    report_to="none",
)


# ============================================================
# START TRAINING
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed["train"],
)

trainer.train()


# ============================================================
# SAVE BOTH LoRA AND PROCESSOR
# ============================================================
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\nTraining complete!")
print(f"LoRA model saved to: {OUTPUT_DIR}")
