from transformers import modeling_utils

from ring_attn import ring_attn


class RingAttentionForward:
    def __init__(self, group):
        self.group = group

    def __call__(self, module, query, key, value, *args, **kwargs):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = ring_attn(query, key, value, causal=module.is_causal, group=self.group)
        return out, None


def patch_model(group=None):
    modeling_utils.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = RingAttentionForward(group)
