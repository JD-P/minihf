import torch
from torch import distributed as dist
import transformers.models.qwen2.modeling_qwen2 as modeling

from ring_attn import ring_attn


class Qwen2RingAttention(modeling.Qwen2Attention):
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings=None,
    ):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = modeling.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        x = ring_attn(q, k, v, causal=True, group=self.group)
        x = x.view(bsz, q_len, self.num_heads * self.head_dim)
        x = self.o_proj(x)
        return x, None, None


def patch_model(group=None):
    Qwen2RingAttention.group = group
    modeling.QWEN2_ATTENTION_CLASSES["flash_attention_2"] = Qwen2RingAttention
