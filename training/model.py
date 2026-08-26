"""Decoder-only Transformer with RoPE, SwiGLU, RMSNorm, and weight tying."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 4096
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes complex rotary frequencies."""
    dim_indices = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to query or key tensors (batch, seq_len, heads, head_dim)."""
    # x shape: [B, S, H, D]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Slice freqs to match current sequence length S
    freqs = freqs_complex[:x.shape[1], :].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_rotated.type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # Gate
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # Up
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, freqs_complex)
        k = apply_rope(k, freqs_complex)

        # Permute for scaled_dot_product_attention: [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=True if attention_mask is None else False
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), freqs_complex, attention_mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class SyntheticTransformer(nn.Module):
    """Synthetic Decoder-only Transformer model initialized from random weights."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE complex frequencies
        head_dim = config.hidden_size // config.num_attention_heads
        freqs_complex = precompute_rope_freqs(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

        self._init_weights()

    def _init_weights(self):
        """Standard small-variance initialization from random weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return final normalized decoder states without applying the LM head.

        Keeping this as a first-class read-only interface lets diagnostic probes
        inspect a frozen checkpoint without hooks or duplicated forward logic.
        """
        _, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        freqs = self.freqs_complex[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs, attention_mask)

        return self.norm(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.forward_hidden_states(input_ids, attention_mask)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 0.0
    ) -> torch.Tensor:
        """Greedy or temperature-based token generation."""
        self.eval()
        stop_ids = set(stop_token_ids or [])
        curr_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] >= self.config.max_position_embeddings:
                break
            logits, _ = self.forward(curr_ids)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            token_val = next_token.item()
            if token_val in stop_ids:
                break

        return curr_ids
