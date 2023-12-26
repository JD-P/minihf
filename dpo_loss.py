"""Direct Preference Optimization loss. (https://arxiv.org/abs/2305.18290)"""

import torch
from torch import nn
from torch.nn import functional as F


def logp_completion(logits, tokens, mask):
    """Compute the log probabilities of completions given their prompts.

    Args:
        tokens: The tokens input to the model. Shape: (..., T).
        logits: The logits output from the model. Shape: (..., T, V).
        mask: A mask indicating which tokens should be included in the log probabilities. It should
            exclude prompt tokens and padding tokens. Shape: (..., T).
    """
    logits = F.log_softmax(logits, dim=-1)
    logp_tokens = logits[..., :-1, :].gather(-1, tokens[..., 1:, None])[..., 0]
    return torch.sum(logp_tokens * mask[..., 1:], dim=-1)


def reduction(x, reduction):
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    elif reduction == "none":
        return x
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss. (https://arxiv.org/abs/2305.18290)

    The DPO loss takes as input pairs of log probabilities of completions given the
    same prompt for each completion in a pair, under the model and a reference model, and a win
    rate indicating how often the first completion is preferred over the second. It optimizes the
    model to maximize the implied reward, regularized by the KL divergence between the model and
    the reference model.

    Conservative DPO (https://ericmitchell.ai/cdpo.pdf) is supported using the `eps` parameter
    and/or the `win_rate` argument.

    Args:
        beta (float): The KL penalty coefficient.
        eps (float): The label smoothing amount.
        reduction (str): The reduction to apply to the loss.
    """

    def __init__(self, beta, eps=0.0, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def extra_repr(self):
        return f"beta={self.beta:g}, eps={self.eps:g}, reduction={self.reduction!r}"

    def forward(self, logp_1, logp_ref_1, logp_2, logp_ref_2, win_rate=None):
        """Compute the Direct Preference Optimization loss.

        Args:
            logp_1: Log probabilities of the first completions given their prompts under the
                model. Should be differentiable w.r.t. the model parameters. Shape: (N).
            logp_ref_1: Log probabilities of the first completions given their prompts under the
                reference model. Shape: (N).
            logp_2: Log probabilities of the second completions given their prompts, under the
                model. Should be differentiable w.r.t. the model parameters. Shape: (N).
            logp_ref_2: Log probabilities of the second completions given their prompts under the
                reference model. Shape: (N).
            win_rate: 0-1, indicating how often the first completion is preferred over the second.
                Shape: (N). Default: 1 (the first completion is always preferred).
        """
        win_rate = torch.ones_like(logp_1) if win_rate is None else win_rate
        win_rate = win_rate * (1 - 2 * self.eps) + self.eps
        ratio_1 = logp_1 - logp_ref_1
        ratio_2 = logp_2 - logp_ref_2
        losses_1 = -F.logsigmoid(self.beta * (ratio_1 - ratio_2))
        losses_2 = -F.logsigmoid(self.beta * (ratio_2 - ratio_1))
        losses = torch.lerp(losses_2, losses_1, win_rate)
        return reduction(losses, self.reduction)
