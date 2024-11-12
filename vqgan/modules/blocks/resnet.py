import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv_shortcut: bool = False,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )

        if self.out_channels != in_channels:
            if self.use_conv_shortcut:
                self.shortcut = nn.Conv2d(
                    in_channels, self.out_channels, kernel_size=3, padding=1
                )
            else:
                self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.out_channels != residual.shape[1]:
            residual = self.shortcut(residual)

        return hidden_states + residual
