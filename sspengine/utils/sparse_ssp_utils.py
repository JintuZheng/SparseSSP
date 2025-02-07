from sspengine.engine import BUILDER
import torch.nn.functional as F
import torch
import torch.nn as nn

class PrefixInterpolation(object):
    def __init__(self, target_voxel_size) -> None:
        self.target_voxel_size = target_voxel_size

    def __call__(self, *args):
        img = F.interpolate(args[0], size = self.target_voxel_size, mode = 'nearest')
        return img

class DepthChannelSwitcher(nn.Module):
    def __init__(self, target_voxel_size, mode, gate_in_2d_channels = None, gate_out_2d_channels_list = None, out_2d_gate = None):
        super().__init__()
        self.target_voxel_size = target_voxel_size
        if mode == 'chans_to_depths':
            self.switch_func = channels_to_depths
            self.out_gates = [None]
        elif mode == 'depths_to_chans':
            self.switch_func = depths_to_channels
            assert gate_in_2d_channels is not None, "`depths_to_chans` mode must give a gate impl."
            assert gate_out_2d_channels_list is not None, "`depths_to_chans` mode must give a gate impl."
            assert out_2d_gate is not None, "`depths_to_chans` mode must give a gate impl."
            out_gates = []
            out_2d_gate.update(dict(in_chan = gate_in_2d_channels, out_chan = gate_in_2d_channels))
            out_gates.append(BUILDER.build(out_2d_gate))
            for out_c in gate_out_2d_channels_list:
                out_2d_gate.update(dict(in_chan = gate_in_2d_channels, out_chan = out_c))
                out_gates.append(BUILDER.build(out_2d_gate))
            self.out_gates = nn.ModuleList(out_gates)
        else:
            raise NotImplementedError
        
    def forward(self, *args):
        # <tensor>
        x = self.switch_func(args[0], self.target_voxel_size, self.out_gates[0], args[-1])
        if len(args) > 2:
            new_li = []
            for fidx, _i in enumerate(args[1]): # <list>
                _i = self.switch_func(_i, self.target_voxel_size, self.out_gates[fidx + 1], args[-1])
                new_li.append(_i)
            return (x, new_li)
        else:
            return x

def depths_to_channels(feat, *args):
    B, C, D, H, W = feat.shape
    feat = feat.view(B, C*D, H, W)
    gate = args[-2]
    feat = gate(feat, args[-1])
    return feat

def channels_to_depths(feat, *args):
    D = args[0][0]
    B, C, H, W = feat.shape
    feat = feat.reshape(B, C//D, D, H, W)
    return feat