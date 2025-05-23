from transformers import PretrainedConfig, AutoConfig
from typing import Optional

class DiVAConfig(PretrainedConfig):
    model_type = "diva"

    def __init__(
        self,
        audio_model_name_or_path: str = "openai/whisper-tiny", # Using tiny for easier testing
        llm_model_name_or_path: str = "EleutherAI/pythia-14m", # Using small LLM for easier testing
        audio_token_index: int = 32000, # Example, should be verified against actual tokenizer
        audio_bos_token: Optional[str] = "<|audio_bos|>", # Example token
        audio_eos_token: Optional[str] = "<|audio_eos|>", # Example token
        num_query_tokens: int = 64, # Matches DUMMY_NUM_QUERY_TOKENS from tests
        # Other DiVA specific parameters can be added here
        **kwargs,
    ):
        self.audio_model_name_or_path = audio_model_name_or_path
        self.llm_model_name_or_path = llm_model_name_or_path
        
        # Load actual HuggingFace configs
        # Error handling (e.g., if model_name_or_path is invalid) will be done by AutoConfig
        self.audio_config = AutoConfig.from_pretrained(audio_model_name_or_path)
        self.text_config = AutoConfig.from_pretrained(llm_model_name_or_path)
        
        self.audio_token_index = audio_token_index
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.num_query_tokens = num_query_tokens
        
        # Derive hidden sizes directly from the loaded configs
        # These replace the old whisper_hidden_size and text_hidden_size attributes
        # The DiVAModel will need to be updated to use these paths (e.g., self.config.audio_config.hidden_size)
        # For compatibility with previous DiVAModel code that might expect whisper_hidden_size directly on DiVAConfig,
        # we can set them here. However, the long-term goal should be to access them via audio_config/text_config.
        # For now, to minimize breakage in DiVAModel, I'll set them directly on DiVAConfig as well.
        # This implies that DiVAModel's __init__ will need to be updated later if we want strict adherence
        # to only audio_config.hidden_size.
        # The previous DiVAModel's WhisperConnector used config.whisper_hidden_size and config.text_hidden_size.
        
        # The test code expects:
        # diva_model.config.whisper_hidden_size
        # diva_model.config.text_hidden_size
        # So, we should provide these directly for now.
        self.whisper_hidden_size = self.audio_config.hidden_size
        self.text_hidden_size = self.text_config.hidden_size

        # The old attributes `reference_encoder` and `reference_decoder` are now implicitly
        # handled by `audio_model_name_or_path` and `llm_model_name_or_path` and their loaded configs.
        # `whisper_config` and `language_model_config` are replaced by `self.audio_config` and `self.text_config`.
        
        # Pass any remaining kwargs to the parent PretrainedConfig
        # This includes things like `architectures` if they are passed in kwargs.
        super().__init__(**kwargs)

    # If DiVAModel needs to access these frequently, properties could be an option,
    # but direct access after __init__ is also fine.
    # Example property (optional, not strictly required by prompt but good practice):
    # @property
    # def whisper_hidden_size(self) -> int:
    #     return self.audio_config.hidden_size

    # @property
    # def text_hidden_size(self) -> int:
    #     return self.text_config.hidden_size
