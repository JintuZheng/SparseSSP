import torch
import torch.nn as nn
from sspengine.engine import BUILDER
from sspengine.modules.MoDE.commons import MoDESubNet2Conv

class MoDELayers(nn.Module):
    def __init__(self, block, num_blocks, num_experts, num_tasks, in_channels:int, out_channels_list: list, **kargs):
        super().__init__()
        assert num_blocks == len(out_channels_list), "The `num_blocks` must be equal to the length of `out_channels`"
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.blocks = []
        for block_id, out_c in enumerate(out_channels_list):
            in_c = in_channels if block_id == 0 else out_channels_list[block_id - 1]
            self.blocks.append(block(num_experts, num_tasks, in_c, out_c, **kargs))
        self.blocks = nn.ModuleList(self.blocks)

class MoDEEncoderLayers(MoDELayers):
    def forward(self, x, task_emb):
        skips = []
        for block in self.blocks:
            x, skip = block(x, task_emb)
            skips.append(skip)
        return x, skips

class MoDEDecoderLayers(MoDELayers):
    def forward(self, x, skips, task_emb):
        for block in self.blocks:
            skip = skips.pop()
            x = block(x, skip, task_emb)
        return x

class MoDEEncoderBlock(nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, **kargs):
        super().__init__()
        self.using_2d_ops = True if 'using_2d_ops' in kargs.keys() and kargs['using_2d_ops'] == True else False    
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv_more = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan, **kargs)
        self.conv_down = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace = True),
        ) if self.using_2d_ops else nn.Sequential(
            nn.Conv3d(out_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(out_chan),
            nn.ReLU(inplace = True),
        )

    def forward(self, x, t):
        x_skip = self.conv_more(x, t)
        x = self.conv_down(x_skip)
        return x, x_skip

class MoDEDecoderBlock(nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, **kargs):
        super().__init__()
        self.using_2d_ops = True if 'using_2d_ops' in kargs.keys() and kargs['using_2d_ops'] == True else False    
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        ) if self.using_2d_ops else nn.Sequential(
            nn.ConvTranspose3d(in_chan, out_chan, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(out_chan),
            nn.ReLU(inplace=True),
        )
        self.conv_less = MoDESubNet2Conv(num_experts, num_tasks, in_chan, out_chan, **kargs)

    def forward(self, x, x_skip, t):
        x = self.convt(x)
        x_cat = torch.cat((x_skip, x), 1)
        x_cat = self.conv_less(x_cat, t)
        return x_cat
