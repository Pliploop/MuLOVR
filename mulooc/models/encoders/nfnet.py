import torch
import torchvision.ops.stochastic_depth as sd_ops

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)



def _scaled_activation(activation_name):
    """
    Apply a scaled activation function according to [1, 2].

    Args:
        activation_name: str - The name of the scaled activation function to apply.

        input: torch.Tensor - The tensor to apply the activation to.

    Return:
        torch.Tensor - The input with the activation applied.

    References:
    
        [1] Arpit, Devansh, Yingbo Zhou, Bhargava Kota, and Venu Govindaraju. "Normalization propagation: A 
            parametric technique for removing internal covariate shift in deep networks." In International 
            Conference on Machine Learning, pp. 1168-1176. PMLR, 2016.

        [2] Brock, A., De, S., & Smith, S. L. (2021). Characterizing signal propagation to close the performance 
            gap in unnormalized resnets. arXiv preprint arXiv:2101.08692.
    """
    activations = {
        'gelu': lambda x: torch.nn.functional.gelu(x) * 1.7015043497085571,
        'relu': lambda x: torch.nn.functional.relu(x) * 1.7139588594436646 
    }
    return activations[activation_name]


class StemModule(nn.Module):
    """Create the stem module. This is a series of convolutional layers that are applied on
    the input, prior to any residual stages."""

    def __init__(self, kernels, channels, strides, activation=F.relu):
        super(StemModule, self).__init__()
        self.layers = self._make_stem_module(kernels, channels, strides, activation)

    def _make_stem_module(self, kernels, channels, strides, activation):
        """Constructs the layers for the stem module."""
        layers = []
        for c, k, s in zip(channels, kernels, strides):
            layers.append(nn.Conv2d(c, c, k, stride=s))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        """Applies the stem module to an input."""
        return self.layers(x)
    
    
class WSConv2D(nn.Module):
    """Creates the variance preserving weight standardized convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=F.relu):
        super(WSConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=False)
        self.activation = activation
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        weight = self.conv.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight_var = weight.var(dim=(1, 2, 3), keepdim=True)
        fan_in = np.prod(weight.shape[1:])
        scale = torch.rsqrt(torch.clamp(weight_var * fan_in, min=1e-4))
        shift = weight_mean * scale
        weight = weight * scale - shift
        x = F.conv2d(x, weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    
class SqueezeExcite(nn.Module):
    """Create a squeeze and excite module."""

    def __init__(self, output_channels):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channels, output_channels // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels // 2, output_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    

class FastToSlowFusion(nn.Module):
    """Make layers that comprise the operations in order to fuse the fast path of NFNet stages to the slow path."""

    def __init__(self, time_kernel_length, time_stride, input_channels, output_channels):
        super(FastToSlowFusion, self).__init__()
        self.conv1 = WSConv2D(input_channels, input_channels, kernel_size=(1, time_kernel_length), stride=(1, time_stride))
        self.conv2 = WSConv2D(input_channels, output_channels, kernel_size=1, stride=1)

    def forward(self, slow, fast):
        fast = self.conv1(fast)
        fast = self.conv2(fast)
        return torch.cat([slow, fast], dim=1)