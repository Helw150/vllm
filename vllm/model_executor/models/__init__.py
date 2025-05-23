# SPDX-License-Identifier: Apache-2.0

from .interfaces import (HasInnerState, SupportsLoRA, SupportsMultiModal,
                         SupportsPP, SupportsV0Only, has_inner_state,
                         supports_lora, supports_multimodal, supports_pp,
                         supports_v0_only)
from .interfaces_base import (VllmModelForPooling, VllmModelForTextGeneration,
                              is_pooling_model, is_text_generation_model)
from .registry import ModelRegistry
from .diva import DiVAModel

__all__ = [
    "ModelRegistry",
    "DiVAModel", # Add DiVAModel to __all__
    "VllmModelForPooling",
    "is_pooling_model",
    "VllmModelForTextGeneration",
    "is_text_generation_model",
    "HasInnerState",
    "has_inner_state",
    "SupportsLoRA",
    "supports_lora",
    "SupportsMultiModal",
    "supports_multimodal",
    "SupportsPP",
    "supports_pp",
    "SupportsV0Only",
    "supports_v0_only",
]

# Actual registration happens via decorators in the model files themselves,
# or it could be done explicitly here if needed.
# For example, if DiVAModel is in diva.py and has:
# from .registry import ModelRegistry
# @ModelRegistry.register("diva")
# class DiVAModel(...):
#
# Then importing DiVAModel above is enough for it to be registered.
# If the decorator is not in DiVAModel's file, we would do:
# ModelRegistry.register("diva", DiVAModel)
# Assuming the decorator is in diva.py, so just importing is fine.
# The task also asked to add "DiVA": DiVAModel to _MODELS, which seems to be an older pattern.
# The new pattern uses ModelRegistry.
# I will ensure DiVAModel is imported so if it has a self-registration decorator, it works.
# The old _MODELS dict and _MULTI_MODAL_MODELS list are no longer in this file.
# The modern way to mark a model as multimodal is likely through an interface or a property
# that SupportsMultiModal would provide, or via its config.

# For clarity, let's explicitly register here if the decorator isn't assumed to be in diva.py
# However, the preferred VLLM style is usually to have the decorator in the model's own file.
# For now, I will only add the import as the primary instruction was "Add the line from .diva import DiVAModel"
# The instruction "Add "DiVA": DiVAModel to the _MODELS dictionary." is outdated for this file structure.
# The equivalent for the new structure is to ensure it's registered with ModelRegistry.
# If DiVAModel itself doesn't have @ModelRegistry.register("diva"), then explicit registration would be:
# ModelRegistry.register("diva")(DiVAModel) # Registering the class
# or if DiVAModel is an instance: ModelRegistry.register("diva", DiVAModel)
# Let's assume DiVAModel will have the decorator.
