#!/usr/bin/env python3

# Trimmed, self-contained reimplementation of the Perceiver encoder
# (transformers.models.perceiver.{configuration,modeling}_perceiver), which
# this repo used to depend on solely for PerceiverConfig/PerceiverEncoder via
# a fork (github.com/fnlandini/transformers, see the old README.md install
# block) -- pulling in the whole `transformers` package for two classes.
#
# Module hierarchy and parameter names below (cross_attention, self_attends,
# attention.self.{query,key,value,layernorm1,layernorm2},
# attention.output.dense, layernorm, mlp.{dense1,dense2}) match the original
# exactly, so checkpoints trained against the transformers-backed model still
# load with strict=True. What's trimmed is HF surface AttractorPerceiver
# never exercises: attention_mask/head_mask/inputs_mask, output_attentions/
# output_hidden_states/return_dict, prune_heads, and feed-forward chunking
# (DiaPer never sets chunk_size_feed_forward, so chunking was always a
# no-op) -- none of that affects parameter registration or forward math.
#
# Derived from the original transformers/Perceiver implementation
# (Copyright 2021 Deepmind and the HuggingFace Inc. team), licensed under
# the Apache License, Version 2.0.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class PerceiverConfig:
    num_latents: int = 256
    d_latents: int = 1280
    d_model: int = 768
    num_blocks: int = 1
    num_self_attends_per_block: int = 26
    num_self_attention_heads: int = 8
    num_cross_attention_heads: int = 8
    qk_channels: Optional[int] = None
    v_channels: Optional[int] = None
    cross_attention_shape_for_attention: str = "kv"
    self_attention_widening_factor: int = 1
    cross_attention_widening_factor: int = 1
    hidden_act: str = "gelu"
    attention_probs_dropout_prob: float = 0.1
    use_query_residual: bool = True


_ACT2FN = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(
        self,
        config: PerceiverConfig,
        is_cross_attention: bool,
        qk_channels: int,
        v_channels: int,
        num_heads: int,
        q_dim: int,
        kv_dim: int,
    ) -> None:
        super().__init__()
        assert qk_channels % num_heads == 0, \
            f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads})."
        assert v_channels % num_heads == 0, \
            f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads})."
        self.num_heads = num_heads
        self.qk_channels_per_head = qk_channels // num_heads
        self.v_channels_per_head = v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor, channels_per_head: int) -> torch.Tensor:
        # (batch, time, channels) -> (batch, num_heads, time, channels_per_head)
        new_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_cross_attention = inputs is not None
        hidden_states = self.layernorm1(hidden_states)
        kv_input = self.layernorm2(inputs) if is_cross_attention else hidden_states

        queries = self._split_heads(self.query(hidden_states), self.qk_channels_per_head)
        keys = self._split_heads(self.key(kv_input), self.qk_channels_per_head)
        values = self._split_heads(self.value(kv_input), self.v_channels_per_head)

        q_head_dim = queries.shape[-1]
        attention_scores = torch.matmul(
            queries, keys.transpose(-1, -2)) / (q_head_dim ** 0.5)

        if is_cross_attention:
            # Softmax over the latents axis instead of the usual key axis.
            attention_probs = attention_scores.softmax(dim=-2)
            attention_probs = attention_probs / \
                attention_probs.sum(dim=-1, keepdim=True)
        else:
            attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(
            *context.size()[:-2], self.num_heads * self.v_channels_per_head)
        return context


class PerceiverSelfOutput(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class PerceiverAttention(nn.Module):
    """Attention module, including a dense output projection."""

    def __init__(
        self,
        config: PerceiverConfig,
        is_cross_attention: bool,
        qk_channels: Optional[int],
        v_channels: Optional[int],
        num_heads: int,
        q_dim: int,
        kv_dim: int,
        use_query_residual: bool = True,
    ) -> None:
        super().__init__()
        if is_cross_attention and qk_channels is None:
            assert config.cross_attention_shape_for_attention in ("q", "kv"), \
                "cross_attention_shape_for_attention must be 'q' or 'kv', " \
                f"got {config.cross_attention_shape_for_attention}."
            qk_channels = q_dim if \
                config.cross_attention_shape_for_attention == "q" else kv_dim
        elif qk_channels is None:
            qk_channels = q_dim
        if v_channels is None:
            v_channels = qk_channels

        self.self = PerceiverSelfAttention(
            config, is_cross_attention, qk_channels, v_channels, num_heads,
            q_dim, kv_dim)
        output_channels = q_dim if is_cross_attention else v_channels
        self.output = PerceiverSelfOutput(
            input_channels=v_channels, output_channels=output_channels)
        self.use_query_residual = use_query_residual

    def forward(
        self, hidden_states: torch.Tensor, inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.output(self.self(hidden_states, inputs=inputs))
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states
        return attention_output


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, config: PerceiverConfig, input_size: int, widening_factor: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        self.intermediate_act_fn = _ACT2FN[config.hidden_act]
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return self.dense2(hidden_states)


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config: PerceiverConfig,
        is_cross_attention: bool,
        qk_channels: Optional[int],
        v_channels: Optional[int],
        num_heads: int,
        q_dim: int,
        kv_dim: int,
        widening_factor: int,
        use_query_residual: bool = True,
    ) -> None:
        super().__init__()
        self.attention = PerceiverAttention(
            config, is_cross_attention=is_cross_attention,
            qk_channels=qk_channels, v_channels=v_channels,
            num_heads=num_heads, q_dim=q_dim, kv_dim=kv_dim,
            use_query_residual=use_query_residual)
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self, hidden_states: torch.Tensor, inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, inputs=inputs)
        layer_output = self.mlp(self.layernorm(attention_output))
        return layer_output + attention_output  # residual connection


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(self, config: PerceiverConfig, kv_dim: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        assert config.d_latents % config.num_self_attention_heads == 0, \
            f"d_latents ({config.d_latents}) must be divisible by " \
            f"num_self_attention_heads ({config.num_self_attention_heads})."
        assert config.d_latents % config.num_cross_attention_heads == 0, \
            f"d_latents ({config.d_latents}) must be divisible by " \
            f"num_cross_attention_heads ({config.num_cross_attention_heads})."

        # Construct the cross attention layer.
        self.cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            num_heads=config.num_cross_attention_heads,
            q_dim=config.d_latents,
            kv_dim=kv_dim,
            widening_factor=config.cross_attention_widening_factor,
            use_query_residual=config.use_query_residual,
        )

        # Construct a single block of self-attention layers. We get deeper
        # architectures by applying this same block more than once
        # (config.num_blocks, see forward()) -- those repeats share weights.
        self.self_attends = nn.ModuleList([
            PerceiverLayer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            for _ in range(config.num_self_attends_per_block)
        ])

    def forward(self, hidden_states: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # Cross-attention between the latents (hidden_states) and inputs:
        hidden_states = self.cross_attention(hidden_states, inputs=inputs)
        # Apply the block of self-attention layers config.num_blocks times:
        for _ in range(self.config.num_blocks):
            for layer_module in self.self_attends:
                hidden_states = layer_module(hidden_states)
        return hidden_states
