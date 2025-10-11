"""Ring attention for PyTorch.

See https://github.com/nshepperd/flash_attn_jax/blob/main/src/flash_attn_jax/ring_attention.py.
"""

import flash_attn.flash_attn_interface as fai
import torch
from torch import distributed as dist


def ppermute(xs, perm, group=None):
    rank = dist.get_rank(group)
    ys = [torch.empty_like(x) for x in xs]
    ops = []
    for src, dst in perm:
        for x, y in zip(xs, ys):
            if src == rank:
                ops.append(dist.P2POp(dist.isend, x, dst, group))
            if dst == rank:
                ops.append(dist.P2POp(dist.irecv, y, src, group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return ys


def _flash_fwd(q, k, v, causal):
    ret = fai._flash_attn_forward(
        q=q,
        k=k,
        v=v,
        dropout_p=0.0,
        softmax_scale=k.shape[-1] ** -0.5,
        causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return ret[0], ret[1]  # out, lse


def _flash_bwd(do, q, k, v, o, lse, causal):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    fai._flash_attn_backward(
        dout=do,
        q=q,
        k=k,
        v=v,
        out=o,
        softmax_lse=lse,
        dq=dq,
        dk=dk,
        dv=dv,
        dropout_p=0,
        softmax_scale=k.shape[-1] ** -0.5,
        causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        rng_state=None,
    )
    return dq, dk, dv


def _ring_fwd(q, k, v, causal=False, group=None):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    perm = [(i, (i + 1) % world_size) for i in range(world_size)]
    n, s, h, d = q.shape
    q_ix = torch.tensor(rank, device=q.device)
    k_ix = torch.tensor(rank, device=q.device)
    o = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full((n, h, s), float("-inf"), device=q.device, dtype=torch.float32)
    for _ in range(world_size):
        o1, lse1 = o, lse
        if not causal:
            o2, lse2 = _flash_fwd(q, k, v, causal=False)
        else:
            if q_ix < k_ix:
                o2 = torch.zeros_like(q)
                lse2 = torch.full((n, h, s), float("-inf"), device=q.device, dtype=torch.float32)
            elif q_ix == k_ix:
                o2, lse2 = _flash_fwd(q, k, v, causal=True)
            else:
                o2, lse2 = _flash_fwd(q, k, v, causal=False)
        lse = torch.logaddexp(lse1, lse2)
        o = o1 * torch.exp(lse1 - lse).mT[..., None] + o2 * torch.exp(lse2 - lse).mT[..., None]
        k, v, k_ix = ppermute([k, v, k_ix], perm, group)
    return o.to(q.dtype), lse


def _ring_bwd(do, q, k, v, o, lse, causal=False, group=None):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    perm = [(i, (i + 1) % world_size) for i in range(world_size)]
    ix = torch.tensor(rank, device=q.device)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    k2, v2, dk2, dv2, ix2 = k, v, dk, dv, ix
    for _ in range(world_size):
        dk2_, dv2_, k2_, v2_, ix2_ = ppermute([dk2, dv2, k2, v2, ix2], perm, group)
        if not causal:
            dqa, dka, dva = _flash_bwd(do, q, k2, v2, o, lse, causal=False)
            dq += dqa
            dk2_ += dka
            dv2_ += dva
        else:
            if ix == ix2:
                dqa, dka, dva = _flash_bwd(do, q, k2, v2, o, lse, causal=True)
            elif ix > ix2:
                dqa, dka, dva = _flash_bwd(do, q, k2, v2, o, lse, causal=False)
            if ix >= ix2:
                dq += dqa
                dk2_ += dka
                dv2_ += dva
        k2, v2, dk2, dv2, ix2 = k2_, v2_, dk2_, dv2_, ix2_
    dk2, dv2 = ppermute([dk2, dv2], perm)
    return dq.to(q.dtype), dk2.to(k.dtype), dv2.to(v.dtype)


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, causal, group = inputs
        o, lse = output
        ctx.causal = causal
        ctx.group = group
        ctx.save_for_backward(q, k, v, o, lse)

    @staticmethod
    def forward(q, k, v, causal, group):
        return _ring_fwd(q, k, v, causal=causal, group=group)

    @staticmethod
    def backward(ctx, do, _):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _ring_bwd(do, q, k, v, o, lse, causal=ctx.causal, group=ctx.group)
        return dq, dk, dv, None, None


def ring_attn(q, k, v, causal=False, group=None):
    o, lse = _RingAttention.apply(q, k, v, causal, group)
    return o
