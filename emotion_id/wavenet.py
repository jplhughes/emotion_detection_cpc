from math import ceil
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        padding = ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.cut_last_element is True:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class Conv1dMasked(nn.Conv1d):
    def __init__(self, *args, mask_present=False, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

        """ Pad so receptive field sees only frames in past, optionally including present frame """
        if mask_present is True:
            left_padding = ((self.kernel_size[0] - 1) * self.dilation[0]) + 1
        else:
            left_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        self.pad = nn.ConstantPad1d((left_padding, 0), 0)

    def forward(self, x):
        assert x.shape[2] % self.stride[0] == 0
        desired_out_length = x.shape[2] // self.stride[0]
        x = self.pad(x)
        x = super().forward(x)
        return x[:, :, :desired_out_length]


class ResidualStack(nn.Module):
    def __init__(
        self,
        n_residual,
        n_skip,
        dilations,
        kernel_size=3,
        groups=4,
        conv_module=Conv1dSamePadding,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for dilation in dilations:
            self.resblocks.append(
                ResidualBlock(
                    n_residual,
                    n_skip,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    conv_module=conv_module,
                )
            )

    def forward(self, x):
        skip_connections = []
        for resblock in self.resblocks:
            x, skip = resblock(x)
            skip_connections.append(skip)
        return F.relu(sum(skip_connections))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_residual,
        n_skip,
        kernel_size=3,
        dilation=1,
        groups=4,
        conv_module=Conv1dSamePadding,
    ):
        super().__init__()
        self.conv_tanh = conv_module(
            n_residual, n_residual, kernel_size=kernel_size, dilation=dilation
        )
        self.conv_sigmoid = conv_module(
            n_residual, n_residual, kernel_size=kernel_size, dilation=dilation
        )
        self.gated_activation_unit = GatedActivation(self.conv_tanh, self.conv_sigmoid)

        self.skip_connection = conv_module(n_residual, n_skip, kernel_size=1, dilation=dilation)
        self.residual_connection = conv_module(
            n_residual, n_residual, kernel_size=1, dilation=dilation
        )

    def forward(self, inp):
        x = self.gated_activation_unit(inp)
        skip = self.skip_connection(x)
        residual = self.residual_connection(x)
        output = residual + inp
        return output, skip


class GatedActivation(nn.Module):
    def __init__(self, conv_tanh, conv_sigmoid):
        super().__init__()
        self.conv_tanh = conv_tanh
        self.conv_sigmoid = conv_sigmoid
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.tanh(self.conv_tanh(x))
        s = self.sigmoid(self.conv_sigmoid(x))
        return t * s
