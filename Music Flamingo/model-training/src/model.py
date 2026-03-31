import torch
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

def load_model(config):

    if config.model.use_dummy_model:
        dummy_config = AutoConfig.from_pretrained(config.model.id)

        # Small model config to test if the pipeline works
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

        model = AudioFlamingo3ForConditionalGeneration(dummy_config)

    else: 
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        config.model.id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        )

    processor = AutoProcessor.from_pretrained(config.model.id)
        
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.model.lora.r,
        lora_alpha=config.model.lora.alpha,
        lora_dropout=config.model.lora.dropout,
        target_modules= config.model.lora.target_modules,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor
