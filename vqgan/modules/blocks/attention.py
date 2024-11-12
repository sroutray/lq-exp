import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, hidden_states, condition=None):
        residual = hidden_states
        hidden_states = self.group_norm(hidden_states)

        if condition is None:
            # Self attention
            query = self.query(hidden_states)
            key = self.key(hidden_states)
        else:
            # Cross attention
            key = self.key(condition)
            value = self.value(condition)
        value = self.value(hidden_states)

        # Reshape and permute
        batch, channels, height, width = hidden_states.shape
        query = query.reshape(batch, channels, -1).permute(0, 2, 1)
        key = key.reshape(batch, channels, -1)
        value = value.reshape(batch, channels, -1).permute(0, 2, 1)

        # Compute attention
        attn_weights = torch.bmm(query, key)
        attn_weights = attn_weights * (channels**-0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to value
        hidden_states = torch.bmm(attn_weights, value)

        # Reshape and project
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, channels, height, width
        )
        hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual
