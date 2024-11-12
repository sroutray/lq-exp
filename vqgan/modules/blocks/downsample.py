import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv

        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, hidden_states):
        if self.with_conv:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)

        return hidden_states
