import torch
from torch import distributed as dist
import transformers.models.mixtral.modeling_mixtral as modeling

from ring_attn import ring_attn


class MixtralRingAttention(modeling.MixtralAttention):
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
    ):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        rank = dist.get_rank(self.group)
        world_size = dist.get_world_size(self.group)
        rotary_seq_len = world_size * q_len
        cos, sin = self.rotary_emb(v, seq_len=rotary_seq_len)
        position_ids = torch.arange(q_len, device=hidden_states.device)[None] + rank * q_len
        q, k = modeling.apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2)
        x = ring_attn(q, k, v, causal=True, group=self.group)
        x = x.view(bsz, q_len, self.num_heads * self.head_dim)
        x = self.o_proj(x)
        return x, None, None


def patch_model(group=None):
    MixtralRingAttention.group = group
    modeling.MIXTRAL_ATTENTION_CLASSES["flash_attention_2"] = MixtralRingAttention
