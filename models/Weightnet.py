import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightPolicy(nn.Module):
    """
    Neural policy for computing normalized weights over samples
    based on historical observation errors.
    """

    def __init__(self, z_dim=15):
        super().__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # --------------------------------------------------
    def forward(self, z_sample, z_his, mask):
        """
        Args:
            z_sample : (N, z_dim)  predicted observations for samples
            z_his    : (z_dim,)    observation history
            mask     : (z_dim,)    1 = valid, 0 = padding

        Returns:
            w        : (N,)        normalized weights (softmax)
        """
        # Absolute prediction error
        error = torch.abs(z_sample - z_his.unsqueeze(0))  # (N, z_dim)

        # Mask out padded history entries
        error = error * mask.unsqueeze(0)

        # Compute logits and normalize
        logits = self.net(error).squeeze(-1)  # (N,)
        w = F.softmax(logits, dim=0)

        return w
