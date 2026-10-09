import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        ignore_index=-100,
        *,
        class_balanced=True,
        binary=False,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.class_balanced = class_balanced
        self.binary = binary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = (
            logits.reshape(-1)
            if self.binary
            else logits.reshape(-1, logits.shape[-1])
        )
        targets = targets.reshape(-1)

        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        if self.binary:
            log_pt = -F.binary_cross_entropy_with_logits(
                logits, targets.to(dtype=logits.dtype), reduction="none"
            )
            pt = log_pt.exp()
        else:
            log_probs = F.log_softmax(
                logits.float() if self.class_balanced else logits, dim=-1
            )
            probs = log_probs.exp()
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = (
            torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            if self.class_balanced
            else self.alpha
        )
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)
        loss = -(focal_weight * log_pt)
        return loss.mean()
