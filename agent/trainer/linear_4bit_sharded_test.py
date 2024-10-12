#!/usr/bin/env python3

import bitsandbytes as bnb
import torch
from torch import distributed as dist, nn
from torch.distributed import nn as dnn
import torch_dist_utils as du

from linear_4bit_sharded import Linear4bitSharded


@torch.no_grad()
def quantize_layer(
    module, compute_dtype=None, blocksize=64, compress_statistics=True, quant_type="fp4"
):
    if not isinstance(module, torch.nn.Linear):
        raise ValueError("quantize_layer only supports nn.Linear")
    compute_dtype = module.weight.dtype if compute_dtype is None else compute_dtype
    q_module = bnb.nn.Linear4bit(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )
    q_module.weight = bnb.nn.Params4bit(
        module.weight,
        requires_grad=False,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )
    if module.bias is not None:
        q_module.bias = torch.nn.Parameter(module.bias, requires_grad=module.bias.requires_grad)
    if module.weight.device.type == "cuda":
        q_module.cuda(module.weight.device)
    return q_module


def main():
    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()

    layer = nn.Linear(10, 20)
    du.broadcast_tensors(layer.parameters())
    layer_q = quantize_layer(layer).to(device)
    layer_qs = Linear4bitSharded(layer, device)

    x = torch.randn(4, 10, device=device)
    y_ref = layer_q(x)
    y = layer_qs(x)
    error = torch.sqrt(torch.mean((y - y_ref) ** 2))
    with du.do_in_order():
        print(f"Rank {rank}: error = {error}")


if __name__ == "__main__":
    main()
