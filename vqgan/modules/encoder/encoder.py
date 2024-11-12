import torch.nn as nn
import torch.nn.functional as F

from vqgan.modules.blocks.resnet import ResnetBlock
from vqgan.modules.blocks.attention import AttnBlock
from vqgan.modules.blocks.downsample import Downsample
from vqgan.modules.blocks.mid import MidBlock


class DownsamplingBlock(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.block_idx = block_idx
        if block_idx == 0:
            self.block_in = config.hidden_channels
        else:
            self.block_in = config.hidden_channels * config.channel_mult[block_idx - 1]
        self.block_out = config.hidden_channels * config.channel_mult[block_idx]
        curr_resoution = config.resolution // (2 ** block_idx)

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_res_blocks):
            self.layers.append(ResnetBlock(self.block_in if layer_idx == 0 else self.block_out, self.block_out))
            if curr_resoution in config.attn_resolutions:
                self.layers.append(AttnBlock(self.block_out))
        
        self.if_downsample = block_idx != config.num_resolutions - 1
        if self.if_downsample:
            self.downsample = Downsample(self.block_out, with_conv=config.resample_with_conv)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if self.if_downsample:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels

        self.conv_in = nn.Conv2d(
            self.in_channels, self.hidden_channels, kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList(
            [
                DownsamplingBlock(config, i_level)
                for i_level in range(config.num_resolutions)
            ]
        )

        self.mid_block = MidBlock(self.hidden_channels * config.channel_mult[-1] , config.no_attn_mid_block, config.dropout)

        self.norm_out = nn.GroupNorm(
            32, config.hidden_channels * config.channel_mult[-1]
        )
        self.conv_out = nn.Conv2d(
            config.hidden_channels * config.channel_mult[-1],
            config.z_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, pixel_values):
        assert (
            pixel_values.shape[2] == pixel_values.shape[3] == self.config.resolution
        ), pixel_values.shape

        hidden_states = self.conv_in(pixel_values)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states
