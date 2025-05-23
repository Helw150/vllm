# SPDX-License-Identifier: Apache-2.0
from .base import MultiModalPlaceholderMap
from .hasher import MultiModalHashDict, MultiModalHasher
from .inputs import (BatchedTensorInputs, ModalityData, MultiModalDataBuiltins,
                     MultiModalDataDict, MultiModalKwargs,
                     MultiModalPlaceholderDict, NestedTensors)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global {class}`~MultiModalRegistry` is used by model runners to
dispatch data processing according to the target model.

Info:
    [mm-processing][]
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHashDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]

# Register all multimodal models here.
# Import processors to ensure they are registered with MULTIMODAL_REGISTRY
from . import llava # noqa: F401
from . import diva_processing # noqa: F401
