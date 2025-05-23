import copy
import json
import os
from typing import Optional, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio
from safetensors.torch import load, load_model
from torch import nn
from .configuring_diva import DiVAConfig
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    WhisperModel,
)


class WhisperConnector(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.decoder = None
        self.projection = nn.Linear(1280, 4096)
        self.query_tokens = nn.Parameter(torch.randn(448, 1280))

    def forward(self, x, output_device="cuda:1"):
        bsz = x.shape[0]
        query_tokens = self.query_tokens[None, :, :].expand(bsz, -1, -1)
        virt_whisper_tokens = self.decoder(
            inputs_embeds=query_tokens, encoder_hidden_states=x
        )
        if self.projection.weight.shape[-1] == 5120:
            virtual_tokens = self.projection(virt_whisper_tokens[0].reshape(112, 5120))
        else:
            virtual_tokens = self.projection(virt_whisper_tokens[0])
        return virtual_tokens.to(output_device)


class DiVAModel(PreTrainedModel):
    config_class = DiVAConfig

    def __init__(
        self, via_path=None, config_dict={}, device_map=None, speech_encoder_device=None
    ):
        super().__init__(DiVAConfig.from_dict(config_dict))
        if speech_encoder_device is None:
            speech_encoder_device = "cuda:0"
        whisper = WhisperModel.from_pretrained(config_dict["reference_encoder"])
        connector = WhisperConnector()
        connector.decoder = copy.deepcopy(whisper.decoder)
        if via_path is not None:
            with open(via_path, "rb") as f:
                sd = load(f.read())

            with torch.no_grad():
                connector.query_tokens = nn.Parameter(sd["query_tokens"])
                connector.projection.weight = nn.Parameter(sd["projection.weight"].T)
                connector.projection.bias = nn.Parameter(sd["projection.bias"])
                wsd = {
                    key.replace("connector.", ""): sd[key]
                    for key in sd
                    if key.startswith("connector.")
                }
                connector.decoder.load_state_dict(wsd)

        if device_map == None:
            num_layers = 32
            num_gpus = 2
            device_map = dict(
                **{"model.embed_tokens": 1, "model.norm": 1, "lm_head": 2},
                **{
                    "model.layers." + str(i): 1 + (i // (num_layers // num_gpus))
                    for i in range(num_layers)
                },
            )

        self.connector = connector.to(speech_encoder_device)
        self.whisper_encoder = whisper.encoder.to(speech_encoder_device)
        self.llm_decoder = AutoModelForCausalLM.from_pretrained(
            config_dict["reference_decoder"],
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(config_dict["reference_encoder"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            config_dict["reference_decoder"], use_fast=False
        )
        if self.tokenizer.pad_token_id == None:
            override_token = list(self.tokenizer.added_tokens_decoder.items())[-1]
            self.tokenizer.pad_token_id = override_token[0]
            self.tokenizer.pad_token = str(override_token[1])
        prefix, suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "PLACEHOLDER"}],
            tokenize=False,
            add_generation_prompt=True,
        ).split("PLACEHOLDER")
        non_null = [line for line in prefix.split("\n") if line.strip()]
        prefix_tok = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tok = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.prefix = torch.tensor(prefix_tok).to(
            self.llm_decoder.model.embed_tokens.weight.device
        )

        self.pre_system = torch.tensor(
            self.tokenizer.encode(non_null[0] + "\n", add_special_tokens=False)
        ).to(self.llm_decoder.model.embed_tokens.weight.device)
        self.post_system = torch.tensor(
            self.tokenizer.encode("\n" + non_null[-1] + "\n", add_special_tokens=False)
        ).to(self.llm_decoder.model.embed_tokens.weight.device)
        self.final_header = torch.tensor(suffix_tok).to(
            self.llm_decoder.model.embed_tokens.weight.device
        )
        self.speech_encoder_device = speech_encoder_device

    def can_generate(cls):
        return False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config=None,
        cache_dir=None,
        **kwargs,
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            via_path = pretrained_model_name_or_path + "/model.safetensors"
            config_path = pretrained_model_name_or_path + "/config.json"
        else:
            # Loading from huggingface repo
            from huggingface_hub import hf_hub_download

            via_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model.safetensors",
                token=kwargs.get("token", None),
            )
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
                token=kwargs.get("token", None),
            )
        with open(config_path, "r") as f:
            config_dict = json.loads(f.read())
        return cls(
            via_path,
            config_dict,
            kwargs["device_map"] if "device_map" in kwargs else "auto",
            (
                kwargs["speech_encoder_device"]
                if "speech_encoder_device" in kwargs
                else None
            ),
        )

    def forward(self, audio, prefix_text_tokens, suffix_text_tokens):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.speech_encoder_device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.connector(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        ).squeeze()

        prefix_embed = self.llm_decoder.model.embed_tokens(prefix_text_tokens)
        suffix_embed = self.llm_decoder.model.embed_tokens(suffix_text_tokens)
        inputs_embeds = torch.cat(
            [prefix_embed, virt_tokens, suffix_embed], axis=0
        ).unsqueeze(0)

        outputs = self.llm_decoder(
            inputs_embeds=inputs_embeds.to(
                self.llm_decoder.model.embed_tokens.weight.device
            ).half(),
            return_dict=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        audio,
        text_prompt=None,
        do_sample=False,
        logits_processor=None,
        max_new_tokens=128,
    ):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.speech_encoder_device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.connector(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        )
        bsz = virt_tokens.shape[0]

        if text_prompt != None and text_prompt != "":
            user_prompt_text = torch.tensor(
                self.tokenizer(
                    text_prompt,
                    add_special_tokens=False,
                    padding=True,
                    padding_side="right",
                )["input_ids"],
                device=self.pre_system.device,
            )
            prefix = torch.cat(
                [
                    self.pre_system.expand(
                        bsz,
                        -1,
                    ),
                    user_prompt_text,
                    self.post_system.expand(
                        bsz,
                        -1,
                    ),
                ],
                axis=1,
            )
        else:
            prefix = self.prefix
        prefix_embed = self.llm_decoder.model.embed_tokens(prefix).expand(bsz, -1, -1)
        suffix = self.final_header
        suffix_embed = self.llm_decoder.model.embed_tokens(suffix).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prefix_embed, virt_tokens, suffix_embed], axis=1)
        outs = [[] for i in range(bsz)]
        complete = [False] * bsz
        outputs = None
        greedy = 1
        i = 0
        while not all(complete) and len(outs[0]) < max_new_tokens:
            past_key_values = outputs.past_key_values if outputs else None
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds.to(
                    self.llm_decoder.model.embed_tokens.weight.device
                ).half(),
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if logits_processor:
                local_outs = torch.tensor(outs) if outs != [] else suffix
                local_outs = local_outs.reshape(1, -1)
                next_token_logits = logits_processor(
                    local_outs,
                    next_token_logits.reshape(1, -1),
                )
                next_token_logits = next_token_logits.flatten()
            if do_sample:
                logits = next_token_logits / temperature
                probs = F.softmax(logits, dim=-1)
                greedy = torch.multinomial(probs, num_samples=1)[0]
            else:
                greedy = next_token_logits.argmax(dim=-1)
            for token_index, out in enumerate(greedy.flatten().tolist()):
                if not complete[token_index]:
                    outs[token_index].append(out)
                if out == 128009:
                    complete[token_index] = True

            next_embed = self.llm_decoder.model.embed_tokens(greedy.reshape(-1, 1))
            inputs_embeds = next_embed
        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)

    def generate_stream(
        self,
        audio,
        text_prompt,
        do_sample=False,
        logits_processor=None,
        max_new_tokens=128,
        return_outputs=False,
        init_outputs=None,
    ):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to(self.whisper_encoder.device)
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.connector(
            hidden_states,
            output_device=self.llm_decoder.model.embed_tokens.weight.device,
        ).squeeze()

        if text_prompt != None and text_prompt != "":
            user_prompt_text = torch.tensor(
                self.tokenizer(
                    text_prompt,
                    add_special_tokens=False,
                    padding=True,
                    padding_side="right",
                )["input_ids"],
                device=self.pre_system.device,
            )
            prefix = torch.cat(
                [self.pre_system, user_prompt_text, self.post_system],
                axis=0,
            )
        else:
            prefix = self.prefix
        prefix_embed = self.llm_decoder.model.embed_tokens(prefix)
        suffix = self.final_header
        suffix_embed = self.llm_decoder.model.embed_tokens(suffix)
        inputs_embeds = torch.cat(
            [prefix_embed, virt_tokens, suffix_embed], axis=0
        ).unsqueeze(0)
        outs = []
        outputs = init_outputs
        greedy = 1
        i = 0
        while greedy != 128009 and len(outs) < max_new_tokens:
            past_key_values = outputs.past_key_values if outputs else None
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds.to(
                    self.llm_decoder.model.embed_tokens.weight.device
                ).half(),
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[-1, -1, :]

            if logits_processor:
                local_outs = torch.tensor(outs) if outs != [] else suffix
                local_outs = local_outs.reshape(1, -1)
                next_token_logits = logits_processor(
                    local_outs,
                    next_token_logits.reshape(1, -1),
                )
                next_token_logits = next_token_logits.flatten()
            if do_sample:
                logits = next_token_logits / temperature
                probs = F.softmax(logits, dim=-1)
                greedy = torch.multinomial(probs, num_samples=1)[0]
            else:
                greedy = next_token_logits.argmax()
            outs.append(greedy)
            next_embed = self.llm_decoder.model.embed_tokens(greedy.reshape(1, 1))
            inputs_embeds = next_embed
            if not return_outputs:
                yield self.tokenizer.decode(outs, skip_special_tokens=True).replace(
                    "<|eot_id|>", ""
                )
            else:
                yield (
                    self.tokenizer.decode(outs, skip_special_tokens=True).replace(
                        "<|eot_id|>", ""
                    ),
                    outputs,
                )
        if not return_outputs:
            return self.tokenizer.decode(outs, skip_special_tokens=True).replace(
                "<|eot_id|>", ""
            )
        else:
            return (
                self.tokenizer.decode(outs, skip_special_tokens=True).replace(
                    "<|eot_id|>", ""
                ),
                outputs,
            )
