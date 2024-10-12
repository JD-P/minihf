#!/usr/bin/env python3

import flash_attn
import torch
from torch import distributed as dist
from torch.distributed import nn as dnn
import torch_dist_utils as du

from ring_attn import ppermute, ring_attn


def main():
    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # test ppermute
    du.print0("Testing ppermute...")
    x = torch.arange(rank * 4, (rank + 1) * 4, device=device)
    perm = [(i, (i + 1) % world_size) for i in range(world_size)]
    y = ppermute([x], perm)
    with du.do_in_order():
        print(f"Rank {rank}: x = {x}, y = {y}")

    q = torch.randn(4, 10, 8, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, 10, 4, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(4, 10, 4, 64, device=device, dtype=torch.bfloat16)
    do = torch.randn(4, 10, 8, 64, device=device, dtype=torch.bfloat16)
    q_all = torch.cat(dnn.all_gather(q), dim=1)
    k_all = torch.cat(dnn.all_gather(k), dim=1)
    v_all = torch.cat(dnn.all_gather(v), dim=1)
    do_all = torch.cat(dnn.all_gather(do), dim=1)

    # non-causal
    du.print0("Testing non-causal ring attention...")
    q_all_ = q_all.clone().requires_grad_()
    k_all_ = k_all.clone().requires_grad_()
    v_all_ = v_all.clone().requires_grad_()
    o_ref = flash_attn.flash_attn_func(q_all_, k_all_, v_all_, causal=False)
    o_ref.backward(do_all)
    q_ = q.clone().requires_grad_()
    k_ = k.clone().requires_grad_()
    v_ = v.clone().requires_grad_()
    o = ring_attn(q_, k_, v_, causal=False)
    o.backward(do)
    o_all = torch.cat(dnn.all_gather(o), dim=1)
    dq_all = torch.cat(dnn.all_gather(q_.grad), dim=1)
    dk_all = torch.cat(dnn.all_gather(k_.grad), dim=1)
    dv_all = torch.cat(dnn.all_gather(v_.grad), dim=1)
    error_o = torch.sqrt(torch.mean((o_all - o_ref) ** 2))
    error_dq = torch.sqrt(torch.mean((q_all_.grad - dq_all) ** 2))
    error_dk = torch.sqrt(torch.mean((k_all_.grad - dk_all) ** 2))
    error_dv = torch.sqrt(torch.mean((v_all_.grad - dv_all) ** 2))
    with du.do_in_order():
        print(f"Rank {rank}: error  o = {error_o}")
        print(f"Rank {rank}: error dq = {error_dq}")
        print(f"Rank {rank}: error dk = {error_dk}")
        print(f"Rank {rank}: error dv = {error_dv}")

    # causal
    du.print0("Testing causal ring attention...")
    q_all_ = q_all.clone().requires_grad_()
    k_all_ = k_all.clone().requires_grad_()
    v_all_ = v_all.clone().requires_grad_()
    o_ref = flash_attn.flash_attn_func(q_all_, k_all_, v_all_, causal=True)
    o_ref.backward(do_all)
    q_ = q.clone().requires_grad_()
    k_ = k.clone().requires_grad_()
    v_ = v.clone().requires_grad_()
    o = ring_attn(q_, k_, v_, causal=True)
    o.backward(do)
    o_all = torch.cat(dnn.all_gather(o), dim=1)
    dq_all = torch.cat(dnn.all_gather(q_.grad), dim=1)
    dk_all = torch.cat(dnn.all_gather(k_.grad), dim=1)
    dv_all = torch.cat(dnn.all_gather(v_.grad), dim=1)
    error_o = torch.sqrt(torch.mean((o_all - o_ref) ** 2))
    error_dq = torch.sqrt(torch.mean((q_all_.grad - dq_all) ** 2))
    error_dk = torch.sqrt(torch.mean((k_all_.grad - dk_all) ** 2))
    error_dv = torch.sqrt(torch.mean((v_all_.grad - dv_all) ** 2))
    with du.do_in_order():
        print(f"Rank {rank}: error  o = {error_o}")
        print(f"Rank {rank}: error dq = {error_dq}")
        print(f"Rank {rank}: error dk = {error_dk}")
        print(f"Rank {rank}: error dv = {error_dv}")


if __name__ == "__main__":
    main()
