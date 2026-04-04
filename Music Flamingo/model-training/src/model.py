import torch
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

def load_model(config):

    if config.model.use_dummy_model:

        # Load only the architecture config (no pretrained weights)
        dummy_config = AutoConfig.from_pretrained(config.model.id)

        # Small model config so the full pipeline can be tested quickly
        dummy_config.audio_config.hidden_size = 32
        dummy_config.audio_config.intermediate_size = 64
        dummy_config.audio_config.num_attention_heads = 2
        dummy_config.audio_config.num_hidden_layers = 2

        dummy_config.text_config.num_hidden_layers = 2
        dummy_config.text_config.layer_types = ["full_attention", "full_attention"]
        dummy_config.text_config.max_window_layers = 2
        dummy_config.text_config.num_attention_heads = 4
        dummy_config.text_config.num_key_value_heads = 1
        dummy_config.text_config.hidden_size = 32
        dummy_config.text_config.intermediate_size = 64

        # Instantiate a randomly-initialised model from the dummy config
        model = AudioFlamingo3ForConditionalGeneration(dummy_config)

    else:
        # Load full pretrained weights
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        config.model.id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        )

    # Initiate processor
    processor = AutoProcessor.from_pretrained(config.model.id)
    
    # Configure LoRA
    # Only these adapter weights will be updated during training, keeping the base model frozen
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,                       # Causal language modelling objective
        r=config.model.lora.r,                              # Rank of the adapter matrices (base : 16)
        lora_alpha=config.model.lora.alpha,                 # Scaling factor for the LoRA updates (base : 32)
        lora_dropout=config.model.lora.dropout,             # Dropout (base : 0.1)
        target_modules= config.model.lora.target_modules,   # Layers that receive adapters
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor
