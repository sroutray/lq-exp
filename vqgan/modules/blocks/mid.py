import torch.nn as nn

from vqgan.modules.blocks.resnet import ResnetBlock
from vqgan.modules.blocks.attention import AttnBlock


class MidBlock(nn.Module):
    def __init__(self, in_channels: int, no_attn: bool, dropout: float):
        super().__init__()
        self.in_channels = in_channels
        self.no_attn = no_attn
        self.dropout = dropout

        # First ResnetBlock
        self.resnet_block1 = ResnetBlock(
            in_channels=in_channels, dropout_prob=self.dropout
        )

        # Attention block (if used)
        if not self.no_attn:
            self.attn_block = AttnBlock(in_channels)

        # Second ResnetBlock
        self.resnet_block2 = ResnetBlock(
            in_channels=in_channels, dropout_prob=self.dropout
        )

    def forward(self, hidden_states):
        hidden_states = self.resnet_block1(hidden_states)

        if not self.no_attn:
            hidden_states = self.attn_block(hidden_states)

        hidden_states = self.resnet_block2(hidden_states)

        return hidden_states
