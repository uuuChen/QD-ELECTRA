# Copyright 2021 Chen-Fa-You, Tseng-Chih-Ying, NTHU
# (Strongly inspired by original Google BERT code, Hugging Face's code, Dong-Hyun Lee's code and Intel Q8Bert code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.electra.modeling_electra import (
    ElectraForPreTraining,
    ElectraDiscriminatorPredictions,
    ElectraForSequenceClassification,
    ElectraClassificationHead,
    ElectraModel,
    ElectraConfig,
    ElectraEmbeddings,
    ElectraAttention,
    ElectraEncoder,
    ElectraIntermediate,
    ElectraLayer,
    ElectraOutput,
    ElectraSelfAttention,
    ElectraSelfOutput,
)


class Electra(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self,
                masked_input_ids,
                attention_mask,
                token_type_ids,
                labels,
                original_input_ids,
                original_attention_mask):

        # Generator
        g_outputs = self.generator(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )
        g_outputs_ids = torch.argmax(g_outputs.logits, dim=2)  # g_outputs.logits shape: (batch_size, max_seq_len,
        # vocab_size)

        # Discriminator
        d_labels = (original_input_ids != g_outputs_ids)
        d_outputs = self.discriminator(
            g_outputs_ids,
            labels=d_labels,
            output_attentions=True,
            output_hidden_states=True
        )

        return g_outputs, d_outputs


class DistillElectraForPreTraining(nn.Module):
    def __init__(self, generator, t_discriminator, s_discriminator, config):
        super().__init__()
        self.generator = generator
        self.config = config
        self.t_discriminator = t_discriminator
        self.s_discriminator = s_discriminator

        self.fit_hidden_dense = nn.Linear(config.s_hidden_size, config.t_hidden_size)

    def forward(self,
                masked_input_ids,
                attention_mask,
                token_type_ids,
                g_labels,
                original_input_ids,
                original_attention_mask):

        # Generator
        g_outputs = self.generator(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=g_labels,
            output_attentions=True,
            output_hidden_states=True
        )

        g_outputs_ids = torch.argmax(g_outputs.logits, dim=2)  # g_outputs.logits shape: (batch_size, max_seq_len,
        # vocab_size)

        # Discriminator
        d_labels = (original_input_ids != g_outputs_ids)

        t_d_outputs = self.t_discriminator(
            g_outputs_ids,
            attention_mask=original_attention_mask,
            labels=d_labels,
            output_attentions=True,
            output_hidden_states=True
        )
        s_d_outputs = self.s_discriminator(
            g_outputs_ids,
            attention_mask=original_attention_mask,
            labels=d_labels,
            output_attentions=True,
            output_hidden_states=True
        )

        # Map student hidden states to teacher hidden states and return
        s2t_hidden_states = list()
        for i, hidden_state in enumerate(s_d_outputs.hidden_states):
            s2t_hidden_states.append(self.fit_hidden_dense(hidden_state))

        return g_outputs, t_d_outputs, s_d_outputs, s2t_hidden_states


class DistillElectraForSequenceClassification(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, t_discriminator, s_discriminator, config):
        super().__init__()
        self.t_discriminator = t_discriminator
        self.s_discriminator = s_discriminator

        self.fit_hidden_dense = nn.Linear(config.s_hidden_size, config.t_hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        t_outputs = self.t_discriminator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        s_outputs = self.s_discriminator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        # Map student hidden states to teacher hidden states and return
        s2t_hidden_states = list()
        for i, hidden_state in enumerate(s_outputs.hidden_states):
            s2t_hidden_states.append(self.fit_hidden_dense(hidden_state))

        return t_outputs, s_outputs, s2t_hidden_states


# -----------------------
# Slightly modified from IntelLabs/nlp-architect
# https://github.com/IntelLabs/nlp-architect/blob/master/nlp_architect/nn/torch/quantization.py
class FakeLinearQuantizationWithSTE(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, scale, bits=8):
        """fake quantize input according to scale and number of bits, dequantize
        quantize(input))"""
        return QuantizedLayer.dequantize(QuantizedLayer.quantize(input, scale, bits), scale)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate estimated gradients for fake quantization using
        Straight-Through Estimator (STE) according to:
        https://openreview.net/pdf?id=B1ae1lZRb"""
        return grad_output, None, None


class QuantizationMode(Enum):
    NONE = auto()
    DYNAMIC = auto()
    EMA = auto()


class QuantizedLayer(ABC):

    CONFIG_ATTRIBUTES = ["weight_bits", "start_step", "mode"]

    def __init__(self, *args, weight_bits=8, start_step=0, mode="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.mode = QuantizationMode[mode.upper()]
        self.start_step = start_step
        self._step = 0
        self._quantized_weight_for_eval = None
        self._weight_scale_for_eval = None

        self._fake_quantize = FakeLinearQuantizationWithSTE.apply

    def forward(self, input):
        if self.mode == QuantizationMode.NONE:
            return super().forward(input)
        if self.training:
            if self._step >= self.start_step:
                out = self.training_quantized_forward(input)
            else:
                out = super().forward(input)
            self._step += 1
        else:
            out = self.inference_quantized_forward(input)
        return out

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self._eval()

    def _eval(self):
        weight_scale = self._get_dynamic_scale(self.weight, self.weight_bits)
        self._weight_scale_for_eval = weight_scale
        self._quantized_weight_for_eval = self.quantize(self.weight, weight_scale, self.weight_bits)
        # print(QuantizedLayer.calc_num_unique_values(self._quantized_weight_for_eval))

    def _get_dynamic_scale(self, x, bits, with_grad=False):
        """Calculate dynamic scale for quantization from input by taking the
        maximum absolute value from x and number of bits"""
        with torch.set_grad_enabled(with_grad):
            threshold = x.abs().max()
        return self._get_static_scale(bits, threshold)

    def _get_static_scale(self, bits, threshold):
        """Calculate scale for quantization according to some constant and number of bits"""
        return self.calc_max_quant_value(bits) / threshold

    @staticmethod
    def calc_num_unique_values(values):
        return len(values.unique())

    @staticmethod
    def calc_max_quant_value(bits):
        """Calculate the maximum symmetric quantized value according to number of bits"""
        return 2 ** (bits - 1) - 1

    @staticmethod
    def quantize(input, scale, bits):
        """Do linear quantization to input according to a scale and number of bits"""
        thresh = QuantizedLayer.calc_max_quant_value(bits)
        return input.mul(scale).round().clamp(-thresh, thresh)

    @staticmethod
    def dequantize(input, scale):
        """linear dequantization according to some scale"""
        return input.div(scale)

    @abstractmethod
    def training_quantized_forward(self, input):
        return NotImplementedError

    @abstractmethod
    def inference_quantized_forward(self, input):
        return NotImplementedError

    @classmethod
    def from_config(cls, *args, config=None, **kwargs):
        """Initialize quantized layer from config"""
        return cls(*args, **kwargs, **{k: getattr(config, k) for k in cls.CONFIG_ATTRIBUTES})


class QuantizedLinear(QuantizedLayer, nn.Linear):

    CONFIG_ATTRIBUTES = QuantizedLayer.CONFIG_ATTRIBUTES + [
        "activation_bits",
        "requantize_output",
        "ema_decay",
    ]

    def __init__(self, *args, activation_bits=8, requantize_output=True, ema_decay=0.9999, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.requantize_output = requantize_output
        self.ema_decay = ema_decay

        self.input_ema_thresh = 0
        self.output_ema_thresh = 0

    def training_quantized_forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        # assert self.training, "should only be called when training"

        if self.mode == QuantizationMode.EMA:
            self.input_ema_thresh = self._update_ema(self.input_ema_thresh, input.detach())

        input_scale = self._get_activation_scale(input, self.input_ema_thresh)
        weight_scale = self._get_dynamic_scale(self.weight, self.weight_bits)

        fake_quantized_input = self._fake_quantize(input, input_scale, self.activation_bits)
        fake_quantized_weight = self._fake_quantize(self.weight, weight_scale, self.weight_bits)

        out = F.linear(fake_quantized_input, fake_quantized_weight, self.bias)

        if self.requantize_output:
            if self.mode == QuantizationMode.EMA:
                self.output_ema_thresh = self._update_ema(self.output_ema_thresh, out.detach())
            output_scale = self._get_activation_scale(out, self.output_ema_thresh)
            out = self._fake_quantize(out, output_scale, self.activation_bits)

        return out

    def inference_quantized_forward(self, input):
        input_scale = self._get_activation_scale(input, self.input_ema_thresh)
        bias_scale = input_scale * self._weight_scale_for_eval

        quantized_input = self.quantize(input, input_scale, self.activation_bits)
        quantized_bias = self.quantize(self.bias, bias_scale, self.accumulation_bits)

        out = F.linear(quantized_input, self._quantized_weight_for_eval, quantized_bias)
        out = self.dequantize(out, bias_scale)

        if self.requantize_output:
            output_scale = self._get_activation_scale(out, self.output_ema_thresh)
            out = self.dequantize(self.quantize(out, output_scale, self.activation_bits), output_scale)

        return out

    def _get_activation_scale(self, x, threshold):
        if self.mode == QuantizationMode.DYNAMIC:
            scale = self._get_dynamic_scale(x, self.activation_bits)
        elif self.mode == QuantizationMode.EMA:
            scale = self._get_static_scale(self.activation_bits, threshold)
        else:
            raise TypeError
        return scale

    def _update_ema(self, ema, input, reduce_fn=lambda x: x.abs().max()):
        """Update exponential moving average (EMA) of activations thresholds.
        the reduce_fn calculates the current threshold from the input tensor"""
        assert self._step >= self.start_step
        if self._step == self.start_step:
            ema = reduce_fn(input)
        else:
            ema -= (1 - self.ema_decay) * (ema - reduce_fn(input))
        return ema


class QuantizedEmbedding(QuantizedLayer, nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_quantized_forward(self, input):
        weight_scale = self._get_dynamic_scale(self.weight, self.weight_bits)
        fake_quantized_weight = self._fake_quantize(self.weight, weight_scale, self.weight_bits)
        return F.embedding(
            input,
            fake_quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def inference_quantized_forward(self, input):
        q_embedding = F.embedding(
            input,
            self._quantized_weight_for_eval,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return self.dequantize(q_embedding, self._weight_scale_for_eval)


def quantized_linear_setup(config, *args, **kwargs):
    linear = QuantizedLinear.from_config(*args, **kwargs, config=config)
    return linear


def quantized_embedding_setup(config, *args, **kwargs):
    embedding = QuantizedEmbedding.from_config(*args, **kwargs, config=config)
    return embedding
# -----------------------


class QuantizedElectraEmbeddings(ElectraEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = quantized_embedding_setup(
            config, config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = quantized_embedding_setup(
            config, config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = quantized_embedding_setup(
            config, config.type_vocab_size, config.embedding_size
        )

        self.LayerNorm = QuantizedLayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = QuantizedDropout(config.hidden_dropout_prob)


class QuantizedElectraSelfAttention(ElectraSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query = quantized_linear_setup(
            config, config.hidden_size, self.all_head_size
        )
        self.key = quantized_linear_setup(
            config, config.hidden_size, self.all_head_size
        )
        self.value = quantized_linear_setup(
            config, config.hidden_size, self.all_head_size
        )

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.distance_embedding = quantized_embedding_setup(
                config, 2 * config.max_position_embeddings - 1, self.attention_head_size
            )


class QuantizedElectraSelfOutput(ElectraSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = quantized_linear_setup(
            config, config.hidden_size, config.hidden_size
        )


class QuantizedElectraAttention(ElectraAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = QuantizedElectraSelfAttention(config)
        self.output = QuantizedElectraSelfOutput(config)


class QuantizedElectraIntermediate(ElectraIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.dense = quantized_linear_setup(
            config, config.hidden_size, config.intermediate_size
        )


class QuantizedElectraOutput(ElectraOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = quantized_linear_setup(
            config, config.intermediate_size, config.hidden_size
        )


class QuantizedElectraLayer(ElectraLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = QuantizedElectraAttention(config)
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = QuantizedElectraAttention(config)
        self.intermediate = QuantizedElectraIntermediate(config)
        self.output = QuantizedElectraOutput(config)


class QuantizedElectraEncoder(ElectraEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([QuantizedElectraLayer(config) for _ in range(config.num_hidden_layers)])


class QuantizedElectraModel(ElectraModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = QuantizedElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = quantized_linear_setup(
                config, config.embedding_size, config.hidden_size
            )

        self.encoder = QuantizedElectraEncoder(config)


class QuantizedElectraDiscriminatorPredictions(ElectraDiscriminatorPredictions):
    def __init__(self, config):
        super().__init__(config)
        self.dense = quantized_linear_setup(
            config, config.hidden_size, config.hidden_size
        )
        self.dense_prediction = quantized_linear_setup(
            config, config.hidden_size, 1
        )


class QuantizedElectraForPreTraining(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.electra = QuantizedElectraModel(config)
        self.discriminator_predictions = QuantizedElectraDiscriminatorPredictions(config)


class QuantizedElectraClassificationHead(ElectraClassificationHead):
    def __init__(self, config):
        super().__init__(config)
        self.dense = quantized_linear_setup(
            config, config.hidden_size, config.hidden_size
        )
        self.out_proj = quantized_linear_setup(
            config, config.hidden_size, config.num_labels
        )


class QuantizedElectraForSequenceClassification(ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.electra = QuantizedElectraModel(config)
        self.classifier = QuantizedElectraClassificationHead(config)


# Completely same with original torch.nn.LayerNorm code
class QuantizedLayerNorm(nn.LayerNorm):
    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


# Completely same with original torch.nn.Dropout code
class QuantizedDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)


if __name__ == '__main__':
    model_cfg = ElectraConfig().from_json_file('config/QDElectra_base.json')
    # s_discriminator = QuantizedElectraForPreTraining(model_cfg).from_pretrained(
    #     'google/electra-small-discriminator', config=model_cfg
    # )
    # print(s_discriminator)
