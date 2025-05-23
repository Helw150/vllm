from typing import Mapping, Optional, Sequence, Type, Any

import torch
from transformers import AutoProcessor, WhisperFeatureExtractor # Added WhisperFeatureExtractor

from vllm.config import InputProcessingContext
from vllm.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                         MultiModalInputs, MultiModalKwargs)
from vllm.multimodal.audio import AudioMediaIO 
from vllm.multimodal.base import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptUpdate, PromptUpdateDetails,
                                        BaseDummyInputsBuilder, PromptReplacement,
                                        MultiModalDataParser) # Added MultiModalDataParser
from vllm.multimodal.registry import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.diva import DiVAConfig


class DiVAProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> DiVAConfig:
        return self.ctx.get_hf_config(DiVAConfig)

    def get_hf_processor(self, **kwargs: object):
        # Uses audio_model_name_or_path and audio_config.trust_remote_code from DiVAConfig
        config = self.get_hf_config()
        return AutoProcessor.from_pretrained(
            config.audio_model_name_or_path, 
            trust_remote_code=config.audio_config.trust_remote_code, 
            **kwargs
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None} # Allow multiple audio inputs if needed, or 1 for single

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor
        if not isinstance(feature_extractor, WhisperFeatureExtractor):
            raise TypeError(
                f"Expected feature_extractor to be WhisperFeatureExtractor, "
                f"but got: {type(feature_extractor)}")
        return feature_extractor


class DiVADummyInputsBuilder(BaseDummyInputsBuilder[DiVAProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Uses tokens from DiVAConfig. If multiple audio inputs, repeat placeholders.
        # The placeholder in text that DiVAMultiModalProcessor will target.
        # This could be config.audio_bos_token or a more generic <audio> token.
        # Let's assume the processor targets the BOS token for replacement.
        diva_config = self.info.get_hf_config()
        audio_placeholder = diva_config.audio_bos_token or "<|audio_bos|>" # Default if None
        
        text = "Describe the following audio: "
        if mm_counts.get("audio", 0) > 0:
            text += audio_placeholder * mm_counts["audio"]
        return text

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        if mm_counts.get("audio", 0) > 0:
            # Create dummy audio data compatible with WhisperFeatureExtractor
            # WhisperFeatureExtractor expects raw audio waveform.
            # Default sampling rate for Whisper is 16000.
            sampling_rate = self.info.get_feature_extractor().sampling_rate
            # Duration for dummy audio, e.g., 1 second
            duration = 1 
            num_samples = sampling_rate * duration
            # Create a list of dummy audio waveforms
            dummy_waveforms = [
                torch.sin(torch.arange(num_samples).float() / (sampling_rate / 440.0) * 2 * torch.pi).numpy()
                for _ in range(mm_counts["audio"])
            ]
            # AudioMediaIO can take a numpy array directly if a 'data' field is specified
            # or if it's adapted to handle raw waveform data.
            # For now, assuming AudioMediaIO can handle a path or a dict with 'array' and 'sampling_rate'.
            # To simplify, we'll use the dict format if AudioMediaIO supports it.
            # Let's assume AudioMediaIO is flexible or we provide a path to a dummy file.
            # Since creating files is hard here, we'll assume AudioMediaIO or the processor can handle raw data.
            # The processor usually expects a list of raw audio arrays.
            # We will pass a list of dicts that MultiModalDataParser can understand
            return {"audio": [{"array": arr, "sampling_rate": sampling_rate} for arr in dummy_waveforms]}
        return {}


class DiVAMultiModalProcessor(BaseMultiModalProcessor[DiVAProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _get_mm_fields_config(
        self,
        data_items: MultiModalDataItems,
        processed_mm_inputs: MultiModalInputs,
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Whisper processor outputs 'input_features'. It does not typically output 'feature_attention_mask'.
        return {"input_features": MultiModalFieldConfig.batched("audio")}

    def _get_prompt_updates(
        self,
        cache: ProcessingCache,
        kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        
        tokenizer = self.info.get_tokenizer()
        diva_config = self.info.get_hf_config()

        # Get token IDs for BOS, EOS, and the main audio placeholder
        # The main audio placeholder token ID is diva_config.audio_token_index
        audio_bos_str = diva_config.audio_bos_token
        audio_eos_str = diva_config.audio_eos_token
        
        if not audio_bos_str or not audio_eos_str:
            raise ValueError("audio_bos_token and audio_eos_token must be defined in DiVAConfig.")

        audio_bos_id = tokenizer.convert_tokens_to_ids(audio_bos_str)
        audio_eos_id = tokenizer.convert_tokens_to_ids(audio_eos_str)
        # The repeating placeholder token (between BOS and EOS)
        audio_placeholder_id = diva_config.audio_token_index 
        
        # Validate token IDs
        for token_str, token_id in [
            (audio_bos_str, audio_bos_id), 
            (audio_eos_str, audio_eos_id),
            (f"audio_token_index ({audio_placeholder_id})", audio_placeholder_id) # For error message
        ]:
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"Audio token '{token_str}' (ID: {token_id}) not found in tokenizer "
                    f"vocabulary or maps to UNK token. Ensure it's added to the tokenizer."
                )

        # Get processed audio features from kwargs (output of HF processor)
        # These are expected to be batched if multiple audio inputs were provided.
        input_features = kwargs.get("input_features") # Shape: [num_audios, num_mels, seq_len_mels]
        if input_features is None:
            return [] # No audio input, no updates needed

        if not isinstance(input_features, torch.Tensor):
             raise TypeError(f"Expected input_features to be a Tensor, but got {type(input_features)}")

        num_audio_inputs = input_features.shape[0]
        
        # Determine num_dynamic_features for each audio input
        # Whisper encoder downsamples sequence length by a factor of 2.
        # input_features.shape[2] is the sequence length of mel spectrogram frames.
        # Number of features after encoder = input_features.shape[2] // 2
        # This assumes input_features is already padded/truncated to max length by processor.
        # If WhisperProcessor doesn't pad to a fixed length, this shape[2] could vary per audio.
        # However, HF processors usually pad batches to the max length in the batch.
        
        # For Whisper, the number of encoder output tokens is `n_frames // 2`, where `n_frames` is
        # the sequence length of the input features to the encoder.
        # `input_features` from the processor has shape (batch_size, num_mel_bins, num_frames_after_processing)
        # So, num_frames_after_processing = input_features.shape[2]
        
        num_features_per_audio = input_features.shape[2] // 2 # Each item in the batch

        # The textual placeholder to target for replacement.
        # This should be what `DiVADummyInputsBuilder` (and users) put in the prompt.
        # Let's assume the BOS token string is used as the placeholder in the prompt for each audio segment.
        target_placeholder_str = audio_bos_str
        target_placeholder_id = audio_bos_id
        
        # Create a list of replacement token sequences, one for each audio input
        replacements_for_all_audios: List[List[int]] = []
        for i in range(num_audio_inputs):
            # For each audio input, construct its BOS + features + EOS sequence
            # The number of features is `num_features_per_audio` for this specific audio item.
            # If `input_features` is batched and padded, `num_features_per_audio` will be the same for all items.
            # If we had per-item feature lengths (e.g. from a feature_attention_mask), we would use that here.
            # For now, assume uniform length from padded batch.
            current_audio_replacement = (
                [audio_bos_id] + 
                [audio_placeholder_id] * num_features_per_audio + 
                [audio_eos_id]
            )
            replacements_for_all_audios.append(current_audio_replacement)

        # The PromptReplacement needs to handle multiple occurrences of the target_placeholder_id
        # if there are multiple audio inputs. It will replace them sequentially.
        
        # This counter ensures that if multiple audio inputs (and thus multiple placeholders)
        # are in the prompt, each gets replaced by its corresponding processed audio data.
        replace_idx_counter = 0

        def get_replacement_for_next_audio(details: PromptUpdateDetails) -> list[int]:
            nonlocal replace_idx_counter
            if replace_idx_counter < len(replacements_for_all_audios):
                replacement_tokens = replacements_for_all_audios[replace_idx_counter]
                replace_idx_counter += 1
                return replacement_tokens
            # Should not happen if prompt and audio counts match
            raise RuntimeError("More audio placeholders in prompt than processed audio inputs.")

        return [
            PromptReplacement(
                modality="audio", 
                target=[target_placeholder_id], # The token ID in the prompt to be replaced
                replacement=get_replacement_for_next_audio, 
            )
        ]

@MULTIMODAL_REGISTRY.register_processor(
    DiVAMultiModalProcessor,
    info=DiVAProcessingInfo,
    dummy_inputs=DiVADummyInputsBuilder,
    config_class=DiVAConfig,
)
class DiVARegistrationHelper:
    pass
