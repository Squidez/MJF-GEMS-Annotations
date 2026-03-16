import torch
import inspect
from transformers import AutoProcessor, AutoConfig, VoxtralForConditionalGeneration #AutoModelForAudioTextToText
from peft import get_peft_model, LoraConfig, TaskType
from transformers.models.voxtral.processing_voxtral import VoxtralProcessorKwargs


def load_model(config):

    if config.model.use_dummy_model:
        dummy_config = AutoConfig.from_pretrained(config.model.id)

        # Small model config to test if the pipeline works
        dummy_config.audio_config.num_hidden_layers = 2
        dummy_config.audio_config.num_attention_heads = 2
        dummy_config.audio_config.num_key_value_heads = 2
        dummy_config.audio_config.hidden_size = 32
        dummy_config.audio_config.intermediate_size = 64
        dummy_config.audio_config.head_dim = 16

        dummy_config.text_config.num_hidden_layers = 2
        dummy_config.text_config.num_attention_heads = 4
        dummy_config.text_config.num_key_value_heads = 2
        dummy_config.text_config.hidden_size = 32
        dummy_config.text_config.intermediate_size = 64
        dummy_config.text_config.head_dim = 8
        
        dummy_config.hidden_size = 32

        model = VoxtralForConditionalGeneration(dummy_config)

    else: 
        model = VoxtralForConditionalGeneration.from_pretrained(
        config.model.id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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
