"""4-bit quantization and FSDP-style parameter sharding for nn.Linear."""

import bitsandbytes as bnb
import torch
from torch import distributed as dist, nn
from torch.nn import functional as F


class Linear4bitSharded(nn.Linear):
    def __init__(
        self,
        layer,
        device,
        group=None,
    ):
        if not isinstance(layer, nn.Linear):
            raise ValueError("layer must be an instance of nn.Linear")
        with torch.device("meta"):
            super().__init__(layer.in_features, layer.out_features, bias=layer.bias is not None)
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.bias = nn.Parameter(layer.bias.to(device)) if layer.bias is not None else None
        del self.weight
        self.device = device
        self.group = group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        weight_q, state = bnb.functional.quantize_4bit(layer.weight.to(device))
        self.state = state
        self.weight_shape = weight_q.shape
        weight_q = weight_q.flatten()
        assert weight_q.shape[0] % world_size == 0
        n_per_shard = weight_q.shape[0] // world_size
        weight_q = weight_q[rank * n_per_shard : (rank + 1) * n_per_shard].clone()
        self.register_buffer("weight", weight_q)

    def forward(self, x):
        world_size = dist.get_world_size(self.group)
        weight_list = [torch.empty_like(self.weight) for _ in range(world_size)]
        dist.all_gather(weight_list, self.weight, group=self.group)
        weight_q = torch.cat(weight_list).view(self.weight_shape)
        weight = bnb.functional.dequantize_4bit(weight_q, self.state)
        return F.linear(x, weight, self.bias)


def quantize_and_shard(module, device, group=None):
    if isinstance(module, nn.Linear):
        return Linear4bitSharded(module, device, group)
    for name, child in module.named_children():
        setattr(module, name, quantize_and_shard(child, device, group))
    return module.cuda(device)
