import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2, mode="nearest")
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states
