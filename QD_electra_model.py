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


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    t_n_heads: int = 12 # Numher of Teacher Heads in Multi-Headed Attention Layers
    s_n_heads: int = 4 # Numher of Student Heads in Multi-Headed Attention Layers
    t_hidden_size: int = 768 # Dimension of Feed-Forward Hidden Layer of Teacher Model
    s_hidden_size: int = 256 # Dimension of Feed-Forward Hidden Layer of Student Model
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


class DistillELECTRA(nn.Module):
    def __init__(self,
                 generator,
                 t_discriminator,
                 s_discriminator,
                 t_hidden_size,
                 s_hidden_size):

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
        g_outputs_ids = torch.argmax(g_outputs.logits, axis=2)  # g_outputs.logits shape: (batch_size, max_seq_len,
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


class QuantizedDistillELECTRA(nn.Module):
    def __init__(self, generator, t_discriminator, s_discriminator):
        super().__init__()
        self.generator = generator
        self.t_discriminator = t_discriminator
        self.s_discriminator = s_discriminator

    def _quantized(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
