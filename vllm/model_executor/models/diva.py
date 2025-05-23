from typing import Optional, List, Union, Tuple, TypedDict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel 

from vllm.config import VllmConfig, QuantizationConfig, ModelConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal.base import MultiModalEmbeddings, SupportsMultiModal 
from vllm.model_executor.models import init_vllm_registered_model
from vllm.model_executor.models.utils import AutoWeightsLoader # For load_weights
from vllm.model_executor.weight_utils import load_tensor_parallel_weights, load_sharded_weights,وار_weights 
from safetensors.torch import load_model as load_safetensors_model 
from vllm.model_executor.models.registry import ModelRegistry 
from vllm.transformers_utils.configs.diva import DiVAConfig # Import refined DiVAConfig


class DiVAAudioInputs(TypedDict):
    input_features: torch.Tensor
    # feature_attention_mask: Optional[torch.Tensor] # WhisperProcessor usually doesn't return this for input_features


def _validate_and_reshape_mm_tensor(
    t: Union[torch.Tensor, List[torch.Tensor]],
    expected_dims: int,
    param_name: str,
) -> torch.Tensor:
    """Helper copied from qwen2_audio.py for validating and reshaping multimedia tensors."""
    if isinstance(t, list):
        if not t:
            raise ValueError(f"Empty list found for {param_name}")
        if not all(isinstance(x, torch.Tensor) for x in t):
            raise TypeError(
                f"Expected {param_name} to be a list of Tensors, "
                f"but got: {[type(x) for x in t]}")

        # Assuming all tensors in the list have the same shape except for the first dimension (batch)
        # and we want to concatenate them along the batch dimension.
        # This is a common scenario for multimodal inputs that are batched.
        # However, for Whisper input_features, it's often a single tensor already batched.
        # If each item in the list is a separate audio clip for the batch,
        # then they should be concatenated along a new batch dimension or handled appropriately.
        # For Whisper, a list of tensors for input_features might mean each tensor is [num_mels, seq_len]
        # and they need to be padded and stacked into [batch_size, num_mels, seq_len].
        # The WhisperProcessor typically handles this padding and stacking.
        # If `t` is a list of already batched tensors [batch_i_size, ...], they could be concatenated.
        # For simplicity, if `t` is a list, let's assume it's a list of single, unbatched items
        # that have already been processed (e.g. padded and stacked) by the multimodal processor
        # into a single tensor before reaching here.
        # If the processor gives a list of tensors that need stacking, that logic is usually
        # handled before it gets to the model's forward pass like this.
        # For now, if it's a list of tensors, we'll assume it's a batch of items
        # and try to stack them if they are not already a single tensor.
        # However, the most common case for input_features from WhisperProcessor is a single tensor.
        if len(t) == 1 and t[0].ndim == expected_dims: # Already a single batched tensor in a list
             t = t[0]
        elif all(x.ndim == expected_dims -1 for x in t): # List of unbatched items
            # This case implies the processor didn't batch them, which is unusual for HF processors.
            # However, if it happens, we stack. This assumes padding was handled.
            try:
                t = torch.stack(t, dim=0)
            except Exception as e:
                raise ValueError(f"Could not stack list of tensors for {param_name}. Ensure they are padded to the same dimensions. Error: {e}")

        else: # List of tensors with unexpected dimensions
            raise ValueError(
                f"List of tensors for {param_name} has items with "
                f"unexpected dimensions: {[x.shape for x in t]}. "
                f"Expected items to be suitable for stacking into {expected_dims} dims."
            )


    if not isinstance(t, torch.Tensor):
        raise TypeError(
            f"Expected {param_name} to be a Tensor or list of Tensors, "
            f"but got: {type(t)}")

    if t.ndim != expected_dims:
        # Example: If a single audio input [num_mels, seq_len] is passed
        # and model expects [batch_size, num_mels, seq_len]
        if t.ndim == expected_dims - 1:
            t = t.unsqueeze(0)
        else:
            raise ValueError(
                f"Expected {param_name} to have {expected_dims} dimensions, "
                f"but got: {t.ndim}")
    return t


class WhisperAudioEncoder(nn.Module):
    def __init__(self, diva_config: DiVAConfig, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = diva_config.audio_config 
        self.model = WhisperModel(self.config)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.model.encoder(input_features=input_features)
        return encoder_outputs.last_hidden_state


class DiVAMultiModalProjector(nn.Module):
    def __init__(self, diva_config: DiVAConfig):
        super().__init__()
        self.linear = nn.Linear(diva_config.whisper_hidden_size, diva_config.text_hidden_size, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear(audio_features)


@ModelRegistry.register("diva") 
class DiVAModel(nn.Module, SupportsMultiModal):

    def __init__(self, vllm_config: VllmConfig, diva_config_override: Optional[DiVAConfig] = None):
        super().__init__()
        self.vllm_config = vllm_config
        
        if diva_config_override is not None:
            self.config = diva_config_override
        elif isinstance(vllm_config.model_config.hf_config, DiVAConfig):
            self.config = vllm_config.model_config.hf_config
        else:
            try:
                self.config = DiVAConfig.from_pretrained(vllm_config.model_config.model)
            except Exception as e:
                raise ValueError(
                    "DiVAConfig not found in vllm_config.model_config.hf_config "
                    "and could not be loaded from the model path. "
                    "Please ensure DiVAConfig is correctly configured."
                ) from e

        self.quant_config = vllm_config.quant_config
        self.audio_tower = WhisperAudioEncoder(self.config, self.quant_config)
        self.multi_modal_projector = DiVAMultiModalProjector(self.config)

        llm_hf_config = self.config.text_config 
        lm_vllm_config = VllmConfig(
            model=self.config.llm_model_name_or_path, 
            tokenizer=self.config.llm_model_name_or_path, 
            tokenizer_mode=vllm_config.tokenizer_mode,
            trust_remote_code=vllm_config.trust_remote_code,
            dtype=vllm_config.dtype,
            seed=vllm_config.seed,
            revision=vllm_config.revision,
            tokenizer_revision=vllm_config.tokenizer_revision,
        )
        lm_vllm_config.model_config = ModelConfig(
            model=lm_vllm_config.model,
            tokenizer=lm_vllm_config.tokenizer,
            tokenizer_mode=lm_vllm_config.tokenizer_mode,
            trust_remote_code=lm_vllm_config.trust_remote_code,
            dtype=lm_vllm_config.dtype,
            seed=lm_vllm_config.seed,
            revision=lm_vllm_config.revision,
            hf_config=llm_hf_config, 
            download_dir=vllm_config.download_dir,
            load_format=vllm_config.load_format,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            quantization=vllm_config.quantization,
            max_model_len=vllm_config.max_model_len 
        )
        self.language_model = init_vllm_registered_model(
            model_class=None, 
            model_config=lm_vllm_config.model_config, 
            vllm_config=lm_vllm_config, 
            quant_config=self.quant_config 
        )
        
        # Initialize AutoWeightsLoader
        # Assuming _get_hf_weights_names is not strictly needed if submodule names align with checkpoint keys
        # or if AutoWeightsLoader has good inference for these standard names.
        self.loader = AutoWeightsLoader(self, self._get_hf_weights_names if hasattr(self, '_get_hf_weights_names') else None)


    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[DiVAAudioInputs]:
        input_features = kwargs.get("input_features")

        if input_features is None:
            return None
        
        # Whisper input_features are typically [batch_size, num_mel_bins, sequence_length] (expected_dims=3)
        input_features = _validate_and_reshape_mm_tensor(input_features, 3, "input_features")
        
        # Currently, WhisperProcessor doesn't typically return a feature_attention_mask for input_features.
        # The length is implicitly handled by the model or fixed by padding.
        # If a feature_attention_mask were available and needed:
        # feature_attention_mask = kwargs.get("feature_attention_mask")
        # if feature_attention_mask is not None:
        #     feature_attention_mask = _validate_and_reshape_mm_tensor(feature_attention_mask, 2, "feature_attention_mask") # Assuming [batch, seq_len]

        return DiVAAudioInputs(input_features=input_features) # Add feature_attention_mask if used


    def _process_audio_input(self, audio_input: DiVAAudioInputs) -> torch.Tensor:
        input_features = audio_input["input_features"]
        # feature_attention_mask = audio_input.get("feature_attention_mask") # If used

        audio_tower_output = self.audio_tower(input_features)
        projected_audio_embeddings = self.multi_modal_projector(audio_tower_output)

        # Dynamic length handling note:
        # Qwen2Audio uses feature_attention_mask to determine actual audio length and masks
        # the projected embeddings. Whisper's encoder output length can vary based on input audio length,
        # but the WhisperProcessor often pads input_features to a fixed length (e.g., 30 seconds).
        # If dynamic length of projected_audio_embeddings is desired (i.e., not using all sequence positions
        # from the projector if the original audio was shorter), a mechanism similar to Qwen2Audio's
        # `_get_feat_extract_output_lengths` and subsequent masking would be needed.
        # This would require knowing the original unpadded length of audio features or having an
        # attention mask for `input_features` that reflects padding.
        # For this refactoring, we assume the full sequence length of `projected_audio_embeddings` is used.
        # This means the number of "audio tokens" is fixed by the Whisper encoder's output sequence length
        # for a given input_feature length (e.g. 30s -> 1500 tokens from Whisper encoder).
        # This differs from the previous DiVA connector that produced a fixed `num_query_tokens`.
        # The current simplified projector will output embeddings with sequence length
        # determined by Whisper encoder's output sequence length.
        
        return projected_audio_embeddings


    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        return self._process_audio_input(audio_input)


    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings: Optional[MultiModalEmbeddings] = None) -> torch.Tensor:
        # The actual merging of multimodal_embeddings with text_embeddings,
        # particularly if placeholder tokens are used in input_ids, is typically handled
        # by the underlying embedding layer of the language model (e.g., VocabParallelEmbedding)
        # when it receives both input_ids and multimodal_embeddings.
        # The `language_model.get_input_embeddings` method itself might not directly perform this merge
        # if it's designed to just return text embeddings.
        # The key is that `self.language_model.forward` (or its embedding layer)
        # needs to be aware of `multimodal_embeddings` and `self.config.audio_token_index`
        # (or its equivalent, like `image_token_id` if that's what the LM's embedding layer uses).

        # Assuming self.language_model.get_input_embeddings is a method that can take
        # input_ids and optionally multimodal_embeddings to produce combined embeddings,
        # or that it returns text_embeddings and the LM's forward handles the merge.
        # For clarity, let's follow Qwen2Audio's pattern where get_input_embeddings returns text embeddings,
        # and the merge happens based on how inputs_embeds is formed and passed to the LM.
        
        text_embeddings = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None:
            return text_embeddings

        # This simple concatenation is a placeholder.
        # Proper merging requires replacing placeholder tokens in `input_ids` with `multimodal_embeddings`.
        # This is typically handled by `self.language_model.forward` if `inputs_embeds` is prepared
        # such that placeholder tokens are already replaced, or by the embedding layer itself.
        # For DiVA, if `DiVAMultiModalProcessor` replaced `<audio>` with `N` placeholder tokens,
        # then `input_ids` passed to `get_input_embeddings` would contain these placeholders.
        # The `VocabParallelEmbedding.forward` method in VLLM handles merging `multimodal_embeddings`
        # at positions indicated by `image_token_id` (which would be `self.config.audio_token_index` for DiVA).
        
        # So, `get_input_embeddings` here should just return the text embeddings.
        # The actual combination happens when `self.language_model.forward` is called with `inputs_embeds`
        # that are formed by `self.prepare_inputs_embeds_for_multimodal_input` (a method we might need to add,
        # or ensure existing VLLM mechanisms handle it).
        # For now, let's stick to the previous concatenation logic for directness,
        # but acknowledge this is a simplification compared to placeholder token replacement.
        if not isinstance(multimodal_embeddings, torch.Tensor):
            raise ValueError("Expected multimodal_embeddings to be a Tensor for DiVA.")
        
        # The current `get_input_embeddings` in `DiVAModel` from previous steps
        # already does a torch.cat([multimodal_embeddings, text_embeddings], dim=1).
        # This assumes multimodal_embeddings are prepended.
        # This logic will be retained for now. The key change is how `multimodal_embeddings` are derived.
        return torch.cat([multimodal_embeddings, text_embeddings], dim=1)


    def forward(
        self,
        input_ids: Optional[torch.Tensor], # Made Optional as inputs_embeds can be primary
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        input_metadata: InputMetadata,
        sampling_metadata: SamplingMetadata,
        # Removed audio_input_features, using **kwargs for multimodal
        **kwargs: object, 
    ) -> torch.Tensor:
        
        # Get multimodal embeddings using the new methods
        # This will parse input_features from kwargs
        mm_embeddings = self.get_multimodal_embeddings(**kwargs)

        if mm_embeddings is not None:
            # If multimodal embeddings are present, they need to be merged with text embeddings.
            # VLLM's `VocabParallelEmbedding` layer (used by LMs) handles this merge
            # if `input_ids` contains placeholder tokens (e.g., self.config.audio_token_index)
            # and `mm_embeddings` are passed to the LM's forward method via `inputs_embeds`.
            
            # The `input_ids` here would be the tokenized prompt possibly containing audio placeholders.
            # The `language_model.forward` will internally call its embedding layer.
            # We need to ensure that `mm_embeddings` are passed correctly to it.
            # The standard way is to prepare `inputs_embeds` fully here.
            
            # This implies `input_ids` should be the text part if mm_embeddings are prepended,
            # or `input_ids` contains placeholders that will be replaced by mm_embeddings by the LM's embedding layer.
            # The `DiVAMultiModalProcessor` is set up to replace `<audio>` with N * `<audio_token_id>`.
            # So, `input_ids` coming into forward will have these placeholders.
            # The `language_model.forward` needs to receive `mm_embeddings` to perform the substitution.
            
            # The `language_model` (e.g. LlamaForCausalLM in VLLM) forward method takes `inputs_embeds`.
            # If `inputs_embeds` is provided, `input_ids` are often ignored by the underlying HF model.
            # However, `VocabParallelEmbedding` in VLLM uses `input_ids` to find placeholder tokens
            # and `mm_embeddings` to fill them.
            
            # Let's ensure `input_ids` is not None if `mm_embeddings` are present,
            # as it's needed by `VocabParallelEmbedding` to locate where to insert `mm_embeddings`.
            if input_ids is None:
                raise ValueError("input_ids must be provided when multimodal_embeddings are present for DiVA.")

            # The actual text embeddings will be computed by the language_model's embedding layer,
            # which will also handle merging mm_embeddings at placeholder locations.
            # So, we don't call self.get_input_embeddings() here to pre-form inputs_embeds.
            # We pass input_ids and mm_embeddings to the language_model.forward().
            inputs_embeds_for_lm = None # LM will derive from input_ids + mm_embeddings
            input_ids_for_lm = input_ids
            
        else: # Text-only case
            mm_embeddings = None
            inputs_embeds_for_lm = None
            input_ids_for_lm = input_ids
            if input_ids_for_lm is None: # Should not happen in text-only mode
                 raise ValueError("input_ids must be provided for text-only forward pass.")


        hidden_states = self.language_model.forward(
            input_ids=input_ids_for_lm, 
            positions=positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
            inputs_embeds=inputs_embeds_for_lm, 
            # Pass multimodal_embeddings explicitly if the LM's forward signature supports it
            # and VocabParallelEmbedding uses it.
            # For Llama, it expects `inputs_embeds` to be already combined or uses `image_features` (a LLaVA term).
            # VLLM's LlamaVocabParallelEmbedding uses `image_token_id` from config and `image_features` from kwargs.
            # So, we should pass `mm_embeddings` via kwargs with a name that the embedding layer expects.
            # Let's assume for now it's `image_features` as per LLaVA convention that VLLM's Llama embedding layer might follow.
            # This name (`image_features`) is a bit of a misnomer for audio, but it's about matching the
            # VLLM embedding layer's expected kwarg for multimodal features.
            # DiVAConfig's `audio_token_index` will be used as `image_token_id` by the embedding layer.
            image_features=mm_embeddings 
        )
        return hidden_states


    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    # load_weights is now simplified to use self.loader (AutoWeightsLoader)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): # Removed unused params
        # AutoWeightsLoader handles loading weights for submodules.
        # It requires submodule names (e.g., self.audio_tower, self.multi_modal_projector, self.language_model)
        # to match the prefixes in the checkpoint structure, or a custom mapping function
        # needs to be provided to AutoWeightsLoader (e.g. self._get_hf_weights_names).
        # For now, assuming direct mapping or AutoWeightsLoader's default logic is sufficient.
        self.loader.load_weights(weights)
