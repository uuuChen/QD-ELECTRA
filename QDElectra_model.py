# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.electra import (
    ElectraForPreTraining,
    ElectraModel,
    ElectraConfig,
)

from huggingface_electra import (
    ElectraEmbeddings,
    ElectraAttention,
    ElectraEncoder,
    ElectraIntermediate,
    ElectraLayer,
    ElectraOutput,
    ElectraSelfAttention,
    ElectraSelfOutput
)


class ElectraModelConfig(NamedTuple):
    "Configuration for original Electra model"
    vocab_size: int = None # Size of Vocabulary
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 4 # Numher of Heads in Multi-Headed Attention Layers
    hidden_size: int = 256 # Dimension of Feed-Forward Hidden Layer of the Model
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 128 # Maximum Length for Positional Embeddings

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


class DistillElectraModelConfig(NamedTuple):
    "Configuration for Distill-Electra model"
    vocab_size: int = None # Size of Vocabulary
    n_layers: int = 12 # Numher of Hidden Layers
    t_n_heads: int = 12 # Numher of Teacher Heads in Multi-Headed Attention Layers
    s_n_heads: int = 4 # Numher of Student Heads in Multi-Headed Attention Layers
    t_hidden_size: int = 768 # Dimension of Feed-Forward Hidden Layer of Teacher Model
    s_hidden_size: int = 256 # Dimension of Feed-Forward Hidden Layer of Student Model
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 128 # Maximum Length for Positional Embeddings

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


class ELECTRA(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, masked_input_ids, attention_mask, token_type_ids, labels, original_input_ids):
        # Generator
        g_outputs = self.generator(input_ids=masked_input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   labels=labels,
                                   output_attentions=True,
                                   output_hidden_states=True)
        g_outputs_ids = torch.argmax(g_outputs.logits, dim=2)  # g_outputs.logits shape: (batch_size, max_seq_len,
        # vocab_size)

        # Discriminator
        d_labels = (original_input_ids != g_outputs_ids)
        d_outputs = self.discriminator(g_outputs_ids,
                                       labels=d_labels,
                                       output_attentions=True,
                                       output_hidden_states=True)

        return g_outputs, d_outputs


class DistillELECTRA(nn.Module):
    def __init__(self, generator, t_discriminator, s_discriminator, t_hidden_size, s_hidden_size):
        super().__init__()
        self.generator = generator
        self.t_discriminator = t_discriminator
        self.s_discriminator = s_discriminator
        self.t_hidden_size = t_hidden_size
        self.s_hidden_size = s_hidden_size

        self.fit_hidden_dense = nn.Linear(self.s_hidden_size, self.t_hidden_size)

    def forward(self, masked_input_ids, attention_mask, token_type_ids, labels, original_input_ids):
        # Generator
        g_outputs = self.generator(input_ids=masked_input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   labels=labels,
                                   output_attentions=True,
                                   output_hidden_states=True)
        g_outputs_ids = torch.argmax(g_outputs.logits, dim=2)  # g_outputs.logits shape: (batch_size, max_seq_len,
        # vocab_size)

        # Discriminator
        d_labels = (original_input_ids != g_outputs_ids)
        t_d_outputs = self.t_discriminator(g_outputs_ids,
                                           labels=d_labels,
                                           output_attentions=True,
                                           output_hidden_states=True)
        s_d_outputs = self.s_discriminator(g_outputs_ids,
                                           labels=d_labels,
                                           output_attentions=True,
                                           output_hidden_states=True)

        # Map student hidden states to teacher hidden states and return
        s2t_hidden_states = list()
        for i, hidden_state in enumerate(s_d_outputs.hidden_states):
            s2t_hidden_states.append(self.fit_hidden_dense(hidden_state))

        return g_outputs, t_d_outputs, s_d_outputs, s2t_hidden_states


class QuantizedLinear(nn.Module):
    def __init__(self):
        super().__init__()


class QuantizedElectraEmbeddings(ElectraEmbeddings):
    def __init__(self, config):
        super().__init__(config)


class QuantizedElectraModel(ElectraModel):
    def __init__(self, config):
        super().__init__(config)
        # self.embeddings = QuantizedElectraEmbeddings(config)
        #
        # if config.embedding_size != config.hidden_size:
        #     self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        #
        # self.encoder = ElectraEncoder(config)
        # self.config = config


class QuantizedElectraForPreTraining(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.electra = QuantizedElectraModel(config)


class QuantizedDistillELECTRA(nn.Module):
    def __init__(self, generator, t_discriminator, s_discriminator, t_hidden_size, s_hidden_size):
        super().__init__()
        self.generator = generator
        self.t_discriminator = t_discriminator
        self.s_discriminator = s_discriminator
        self.t_hidden_size = t_hidden_size
        self.s_hidden_size = s_hidden_size

        self.fit_hidden_dense = nn.Linear(self.s_hidden_size, self.t_hidden_size)

    def _quantized(self):
        pass

    def forward(self, masked_input_ids, attention_mask, token_type_ids, labels, original_input_ids):
        # Generator
        g_outputs = self.generator(input_ids=masked_input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   labels=labels,
                                   output_attentions=True,
                                   output_hidden_states=True)
        g_outputs_ids = torch.argmax(g_outputs.logits, dim=2)  # g_outputs.logits shape: (batch_size, max_seq_len,
        # vocab_size)

        # Discriminator
        d_labels = (original_input_ids != g_outputs_ids)
        t_d_outputs = self.t_discriminator(g_outputs_ids,
                                           labels=d_labels,
                                           output_attentions=True,
                                           output_hidden_states=True)
        s_d_outputs = self.s_discriminator(g_outputs_ids,
                                           labels=d_labels,
                                           output_attentions=True,
                                           output_hidden_states=True)

        # Map student hidden states to teacher hidden states and return
        s2t_hidden_states = list()
        for i, hidden_state in enumerate(s_d_outputs.hidden_states):
            s2t_hidden_states.append(self.fit_hidden_dense(hidden_state))

        return g_outputs, t_d_outputs, s_d_outputs, s2t_hidden_states

    def backward(self):
        pass


if __name__ == '__main__':
    model_cfg = ElectraConfig().from_json_file('config/QDElectra_base.json')
    model = QuantizedElectraForPreTraining(model_cfg).from_pretrained('google/electra-small-discriminator')

