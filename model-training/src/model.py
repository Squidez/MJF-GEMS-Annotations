import torch
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration, VoxtralForConditionalGeneration, AutoConfig

def build_model(config):

    if "music-flamingo" in config.model_id:

        if config.use_dummy_model:
            dummy_config = AutoConfig.from_pretrained(config.model_id)

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

            return model

        return AudioFlamingo3ForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
    
    elif 'voxtral-mini' in config.model_id:

        if config.use_dummy_model:
            dummy_config = AutoConfig.from_pretrained(config.model_id)
            
            dummy_config.audio_config.num_hidden_layers = 2
            dummy_config.audio_config.num_attention_heads = 2
            dummy_config.audio_config.num_key_value_heads = 2
            dummy_config.audio_config.hidden_size = 64
            dummy_config.audio_config.intermediate_size = 128
            dummy_config.audio_config.head_dim = 32

            dummy_config.text_config.num_hidden_layers = 2
            dummy_config.text_config.num_attention_heads = 4
            dummy_config.text_config.num_key_value_heads = 2
            dummy_config.text_config.hidden_size = 64
            dummy_config.text_config.intermediate_size = 128
            dummy_config.text_config.head_dim = 16

            dummy_config.hidden_size = 64

            model = VoxtralForConditionalGeneration.from_pretrained(dummy_config)

        return VoxtralForConditionalGeneration.from_pretrained(
            config.model_id,
            dtype=torch.bfloat16,
            device_map='auto')

def build_processor(config):
    return AutoProcessor.from_pretrained(config.model_id)
