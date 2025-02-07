import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MoDESubNet2Conv(nn.Module):
    def __init__(self, num_experts, num_tasks, n_in, n_out, **kargs):
        super().__init__()
        self.conv1 = MoDEConv(num_experts, num_tasks, n_in, n_out, kernel_size=5, padding='same', **kargs)
        self.conv2 = MoDEConv(num_experts, num_tasks, n_out, n_out, kernel_size=5, padding='same', **kargs)

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        return x

class MoDEConv(nn.Module):
    def __init__(self, num_experts, num_tasks, in_chan, out_chan, kernel_size=5, stride=1, padding='same', conv_type='normal', **kargs):
        super().__init__()
        self.using_2d_ops = True if 'using_2d_ops' in kargs.keys() and kargs['using_2d_ops'] == True else False    
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.stride = stride
        self.padding = padding

        self.expert_conv5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 5)
        self.expert_conv3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 3)
        self.expert_conv1x1_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg3x3_pool', self.gen_avgpool_kernel(3))
        self.expert_avg3x3_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)
        self.register_buffer('expert_avg5x5_pool', self.gen_avgpool_kernel(5))
        self.expert_avg5x5_conv = self.gen_conv_kernel(self.out_chan, self.in_chan, 1)

        assert self.conv_type in ['normal', 'final']
        if self.conv_type == 'normal':
            self.subsequent_layer = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(out_chan),
                    torch.nn.ReLU(inplace=True),
                ) if self.using_2d_ops else torch.nn.Sequential(
                    torch.nn.BatchNorm3d(out_chan),
                    torch.nn.ReLU(inplace=True),
                )
        else:
            self.subsequent_layer = torch.nn.Identity()

        self.gate = torch.nn.Linear(num_tasks, num_experts * self.out_chan, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def gen_conv_kernel(self, Co, Ci, K):
        weight = torch.nn.Parameter(torch.empty(Co, Ci, K, K)) if self.using_2d_ops else \
            torch.nn.Parameter(torch.empty(Co, Ci, K, K, K))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight

    def gen_avgpool_kernel(self, K):
        weight = torch.ones(K, K).mul(1.0 / K ** 3) if self.using_2d_ops else \
            torch.ones(K, K, K).mul(1.0 / K ** 3)
        return weight

    def trans_kernel(self, kernel, target_size):
        if self.using_2d_ops == False:
            Dp = (target_size - kernel.shape[-3]) // 2
        Hp = (target_size - kernel.shape[-2]) // 2
        Wp = (target_size - kernel.shape[-1]) // 2
        if self.using_2d_ops == False:
            return F.pad(kernel, [Wp, Wp, Hp, Hp, Dp, Dp])
        else:
            return F.pad(kernel, [Wp, Wp, Hp, Hp])

    def routing(self, g, N):
        magic_strs = ['oihw', 'hw'] if self.using_2d_ops else ['oidhw', 'dhw']

        expert_conv5x5 = self.expert_conv5x5_conv
        expert_conv3x3 = self.trans_kernel(self.expert_conv3x3_conv, self.kernel_size)
        expert_conv1x1 = self.trans_kernel(self.expert_conv1x1_conv, self.kernel_size)
        expert_avg3x3 = self.trans_kernel(
            torch.einsum(f'{magic_strs[0]},{magic_strs[1]}->{magic_strs[0]}', self.expert_avg3x3_conv, self.expert_avg3x3_pool),
            self.kernel_size,
        )
        expert_avg5x5 = torch.einsum(f'{magic_strs[0]},{magic_strs[1]}->{magic_strs[0]}', self.expert_avg5x5_conv, self.expert_avg5x5_pool)

        weights = list()
        for n in range(N):
            weight_nth_sample = torch.einsum(f'{magic_strs[0]},o->{magic_strs[0]}', expert_conv5x5, g[n, 0, :]) + \
                                torch.einsum(f'{magic_strs[0]},o->{magic_strs[0]}', expert_conv3x3, g[n, 1, :]) + \
                                torch.einsum(f'{magic_strs[0]},o->{magic_strs[0]}', expert_conv1x1, g[n, 2, :]) + \
                                torch.einsum(f'{magic_strs[0]},o->{magic_strs[0]}', expert_avg3x3, g[n, 3, :]) + \
                                torch.einsum(f'{magic_strs[0]},o->{magic_strs[0]}', expert_avg5x5, g[n, 4, :])
            weights.append(weight_nth_sample)
        weights = torch.stack(weights)

        return weights

    def forward(self, x, t):
        N = x.shape[0]

        g = self.gate(t)
        g = g.view((N, self.num_experts, self.out_chan))
        g = self.softmax(g)

        w = self.routing(g, N)

        if self.training:
            y = list()
            for i in range(N):
                fusion = F.conv2d(x[i].unsqueeze(0), w[i], bias=None, stride=1, padding='same') if self.using_2d_ops else \
                    F.conv3d(x[i].unsqueeze(0), w[i], bias=None, stride=1, padding='same')
                y.append(fusion)
            y = torch.cat(y, dim=0)
        else:
            y = F.conv2d(x, w[0], bias=None, stride=1, padding='same') if self.using_2d_ops else \
                F.conv3d(x, w[0], bias=None, stride=1, padding='same')

        y = self.subsequent_layer(y)

        return y
