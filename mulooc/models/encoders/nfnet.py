import torch
import torchvision.ops.stochastic_depth as sd_ops

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy

def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)



def _scaled_activation(activation_name):
    activations = {
        'gelu': lambda x: torch.nn.functional.gelu(x) * 1.7015043497085571,
        'relu': lambda x: torch.nn.functional.relu(x) * 1.7139588594436646 
    }
    return activations[activation_name]


class StemModule(nn.Module):
    """Create the stem module. This is a series of convolutional layers that are applied on
    the input, prior to any residual stages."""

    def __init__(self, kernels, in_channels, out_channels, strides, padding,activation=F.relu):
        super(StemModule, self).__init__()
        self.layers = self._make_stem_module(kernels, in_channels, out_channels, strides, padding, activation)
        
        print(f"StemModule: {in_channels} -> {out_channels}")

    def _make_stem_module(self, kernels, in_channels,out_channels, strides,padding,activation):
        """Constructs the layers for the stem module."""
        layers = []
        for i,c, k, s,p in zip(in_channels, out_channels, kernels, strides,padding):
            layers.append(nn.Conv2d(i, c, k, stride=s, padding=p))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        """Applies the stem module to an input."""
        x = self.layers(x)
        return x
    
    
class WSConv2D(nn.Module):
    """Creates the variance preserving weight standardized convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=F.relu, padding=(0,0)):
        super(WSConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=False, padding=padding)
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
            nn.ReLU(),
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
        return torch.cat([slow, fast], dim=-1)
    
class NFNetBlock(nn.Module):
    def __init__(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, beta, stoch_depth, padding,is_transition=False):
        super(NFNetBlock, self).__init__()
        self.is_transition_block = (freq_downsample > 1) or (input_channels != output_channels) or is_transition

        
        if self.is_transition_block:
            self.input_layers = nn.Sequential(
                nn.ReLU(),
                ScalarMultiply(beta)
            )
            self.residual_path = nn.ModuleList()
        else:
            self.input_layers = nn.Identity()
            self.residual_path = nn.Sequential(
                nn.ReLU(),
                ScalarMultiply(beta)
            )

        print(f"NFNetBlock: {input_channels} -> {output_channels} | freq_downsample: {freq_downsample} | is_transition: {self.is_transition_block}")

        strides = [[1, 1], [freq_downsample, 1], [1, 1], [1, 1]]
        per_layer_out_chans = [output_channels // 2] * 3 + [output_channels]
        per_layer_in_chans = [input_channels] + per_layer_out_chans[:-1]
        groups = [1] + [output_channels // 2 // group_size] * 2 + [1]
        activations = [nn.ReLU()] * (len(kernels) - 1) + [None]

        for i,c, k, s, g, a,p in zip(per_layer_in_chans,per_layer_out_chans, kernels, strides, groups, activations, padding):
            self.residual_path.extend([
                WSConv2D(i,c, kernel_size=k, stride=s, groups=g, activation=a, padding = p),
            ])

        self.residual_path.extend([
            SqueezeExcite(per_layer_out_chans[-1]),
            ScalarMultiply(0.0, learnable=True),
            ScalarMultiply(alpha)
        ])
        
        self.residual_path = nn.Sequential(*self.residual_path)

        self.skip_path = nn.Identity()
        if freq_downsample > 1:
            self.skip_path = nn.AvgPool2d(kernel_size=[freq_downsample, 1], stride=[freq_downsample, 1], padding=[freq_downsample//2-1, 0])

        if self.is_transition_block:
            self.skip_path = nn.Sequential(
                self.skip_path,
                WSConv2D(input_channels, output_channels, kernel_size=[1, 1], stride=[1, 1], activation=None, padding = 'same')
            )

        self.output_layers = nn.Sequential(
            StochDepth(survival_probability=1 - stoch_depth, scale_during_test=False)
        )

    def forward(self, x):
        x = self.input_layers(x)
        residual = self.residual_path(x)
        skip = self.skip_path(x)
        output = self.output_layers([skip, residual])
        return output


class StochDepth(nn.Module):
    def __init__(self, survival_probability=0.5, scale_during_test=False):
        super(StochDepth, self).__init__()
        self.survival_probability = survival_probability
        self.scale_during_test = scale_during_test

    def forward(self, x):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x

        # Random bernoulli variable indicating whether the branch should be kept or not
        b_l = torch.bernoulli(torch.tensor([self.survival_probability])).to(residual.device)
        b_l = b_l.view(-1, 1, 1, 1)

        if self.training:
            return shortcut + b_l * residual
        else:
            if self.scale_during_test:
                return shortcut + self.survival_probability * residual
            else:
                return shortcut + residual

class ScalarMultiply(nn.Module):
    def __init__(self, scalar, learnable=False):
        super(ScalarMultiply, self).__init__()
        self.scalar = nn.Parameter(torch.tensor(scalar), requires_grad=learnable)

    def forward(self, x):
        return x * self.scalar




class ParallelModule(nn.Module):
    
    def __init__(self, module, num_parallel = None):
        super(ParallelModule, self).__init__()
        if isinstance(module, nn.Module):
            self.parallels = nn.ModuleList([copy.deepcopy(module) for _ in range(num_parallel)])
        elif isinstance(module, list):
            self.parallels = nn.ModuleList(module)
        
    def forward(self,x):
        
        outputs = []
        
        if isinstance(x, torch.Tensor):
            x = [x for _ in range(len(self.parallels))]
        
        for i, module in enumerate(self.parallels):
            outputs.append(module(x[i]))
            
            
        print(f'ParallelModule: {len(self.parallels)} branches with shapes {[o.shape for o in outputs]}')
        return outputs
            

class NFNetStage(nn.Module):
    def __init__(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, input_expected_var, stoch_depths, num_blocks, padding):
        super(NFNetStage, self).__init__()
        self.blocks = nn.ModuleList()

        # NFNet transition block first
        self.blocks.append(NFNetBlock(
            kernels, 
            freq_downsample, 
            input_channels, 
            output_channels,
            group_size, 
            alpha,
            1.0/input_expected_var,
            float(stoch_depths[0]),
            padding,
            is_transition=True
        ))

        # NFNet non-transition blocks
        expected_std = (input_expected_var**2.0 + alpha**2.0)**0.5
        for idx in range(1,num_blocks):
            self.blocks.append(NFNetBlock(
                kernels,
                1, 
                output_channels, 
                output_channels,
                group_size, 
                alpha, 
                1.0/expected_std,
                float(stoch_depths[idx]),
                padding
            ))
            expected_std = (expected_std**2.0 + alpha**2.0)**0.5

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ParallelNFNetStage(nn.Module):
    def __init__(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, input_expected_var, stoch_depths, num_blocks, padding, from_parallel = 0, num_parallel = 3):
        super(ParallelNFNetStage, self).__init__()
        self.blocks = nn.ModuleList()

        
        block = NFNetBlock(
                kernels,
                freq_downsample,
                input_channels,
                output_channels,
                group_size,
                alpha,
                1.0/input_expected_var,
                float(stoch_depths[0]),
                padding,
                is_transition=True
            )
        if from_parallel == 0:
            block = ParallelModule(block, num_parallel=num_parallel)
        
        self.blocks.append(block)
            
        expected_std = (input_expected_var**2.0 + alpha**2.0)**0.5
        for idx in range(1, num_blocks):
            block = NFNetBlock(
                    kernels,
                    1,
                    output_channels,
                    output_channels,
                    group_size,
                    alpha,
                    1.0/expected_std,
                    float(stoch_depths[idx]),
                    padding
                )
            
            if idx >= from_parallel:
                big_block = ParallelModule(block, num_parallel=num_parallel)
                self.blocks.append(big_block)
            else:
                self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class NFNet(nn.Module):
    def __init__(self, frontend = None, f_value = 0, alpha = 0.2, scaled_activation_type = 'gelu'):
        super(NFNet, self).__init__()

        # Initialize parameters for NFNet Blocks
        self.nfnet_stage_depths = [x*(f_value+1) for x in (1,2,6,3)]
        cumulative_stage_depths = np.concatenate(([0],np.cumsum(self.nfnet_stage_depths)))
        self.stoch_depth_survival_probs = 0.1*np.arange(cumulative_stage_depths[-1])/(cumulative_stage_depths[-1])
        self.stoch_depth_survival_probs = [
            self.stoch_depth_survival_probs[st:end] for st, end in zip(cumulative_stage_depths[:-1], cumulative_stage_depths[1:])
        ]
        self.stage_expected_vars = [1.0] + [(1.0+alpha**2)**0.5]*3
        self.stage_downsamples = [1] + [2]*3
        self.scaled_activation = _scaled_activation(scaled_activation_type)
        self.projector_activation = torch.nn.functional.relu

        # Make Stems
        print("Making Stems")
        self.slow_stem = StemModule(
                kernels=[
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [3, 3]
                ],
                in_channels=[1,16, 32, 64],
                out_channels=[16, 32, 64, 128],
                strides=[
                    [2, 8], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ],
                # manually added padding for same shape
                padding = [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 1]
                ]
            )

        self.fast_stem = StemModule(
                kernels=[
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3]
                ],
                in_channels=[1,2, 4, 8],
                out_channels=[2, 4, 8, 16],
                strides=[
                    [2, 2], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ],
                # manually added padding for same shape
                padding = [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ]
            )

        # Construct NFNet stages for the slow path
        slow_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        slow_nfnet_padding = [
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]]
        ]
        slow_nfnet_input_sizes = [128,256, 512, 1536]
        slow_nfnet_output_sizes = [256,512, 1536, 1536]

        print("Making Slow Layers")
        self.slow_layers = nn.ModuleList([
            NFNetStage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                padding = p,
                group_size=128, 
                alpha=alpha,
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n
            ) for k, f, i, o, e, s, n,p in zip(
                slow_nfnet_kernels,
                self.stage_downsamples,
                slow_nfnet_input_sizes,
                slow_nfnet_output_sizes,
                self.stage_expected_vars,
                self.stoch_depth_survival_probs,
                self.nfnet_stage_depths,
                slow_nfnet_padding
            )
        ])

        # Construct NFNet stages for the fast path
        fast_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        fast_nfnet_padding = [
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]]
        ]
        fast_nfnet_input_sizes = [16,32, 64, 192]
        fast_nfnet_output_sizes = [32,64, 192, 192]

        print("Making Fast Layers")
        self.fast_layers = nn.ModuleList([
            NFNetStage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                group_size=16, 
                alpha=alpha, 
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n,
                padding = p
            ) for k, f, i, o, e, s, n,p in zip(
                fast_nfnet_kernels,
                self.stage_downsamples,
                fast_nfnet_input_sizes,
                fast_nfnet_output_sizes,
                self.stage_expected_vars,
                self.stoch_depth_survival_probs,
                self.nfnet_stage_depths,
                fast_nfnet_padding
            )
        ])

        print("Making Fusion Layers")
        # Construct fast-to-slow fusion layers
        self.fusion_layers = nn.ModuleList([
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=16, output_channels=128),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=32, output_channels=256),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=64, output_channels=512),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=192, output_channels=1536)
        ])

        # Construct summarization and aggregation layers at the output
        self.output_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        ])


        self.frontend = frontend
        self.embed_dim = 1728
        

    def forward(self, x):
        
        if isinstance(x, dict):
            wav = x['audio']
        else:
            wav = x
        
        spec = self.frontend(wav) if self.frontend is not None else wav
        
        
        slow = self.slow_stem(spec)
        fast = self.fast_stem(spec)

        # For each nfnet_transition module
        for fuse, slw_lyr, fst_lyr in zip(self.fusion_layers, self.slow_layers, self.fast_layers):
            slow = fuse(slow, fast)
            slow = slw_lyr(slow)
            fast = fst_lyr(fast)

        # Apply global average pool and concat
        slow_out = self.scaled_activation(self.output_layers[0](slow))
        fast_out = self.scaled_activation(self.output_layers[1](fast))
        output = torch.cat([slow_out, fast_out], dim=1)
        output = output.view(output.size(0), -1)     

        return output
    
    
class NFNetPlus(nn.Module):
    def __init__(self, frontend = None, f_value = 0, alpha = 0.2, scaled_activation_type = 'gelu', parallel_lastblock = 2, from_parallel = 1):
        super(NFNetPlus, self).__init__()

        # Initialize parameters for NFNet Blocks
        self.nfnet_stage_depths = [x*(f_value+1) for x in (1,2,6,3)]
        cumulative_stage_depths = np.concatenate(([0],np.cumsum(self.nfnet_stage_depths)))
        self.stoch_depth_survival_probs = 0.1*np.arange(cumulative_stage_depths[-1])/(cumulative_stage_depths[-1])
        self.stoch_depth_survival_probs = [
            self.stoch_depth_survival_probs[st:end] for st, end in zip(cumulative_stage_depths[:-1], cumulative_stage_depths[1:])
        ]
        self.stage_expected_vars = [1.0] + [(1.0+alpha**2)**0.5]*3
        self.stage_downsamples = [1] + [2]*3
        self.scaled_activation = _scaled_activation(scaled_activation_type)
        self.projector_activation = torch.nn.functional.relu

        # Make Stems
        print("Making Stems")
        self.slow_stem = StemModule(
                kernels=[
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [3, 3]
                ],
                in_channels=[1,16, 32, 64],
                out_channels=[16, 32, 64, 128],
                strides=[
                    [2, 8], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ],
                # manually added padding for same shape
                padding = [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 1]
                ]
            )

        self.fast_stem = StemModule(
                kernels=[
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3]
                ],
                in_channels=[1,2, 4, 8],
                out_channels=[2, 4, 8, 16],
                strides=[
                    [2, 2], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ],
                # manually added padding for same shape
                padding = [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ]
            )

        # Construct NFNet stages for the slow path
        slow_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        slow_nfnet_padding = [
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]]
        ]
        slow_nfnet_input_sizes = [128,256, 512, 1536]
        slow_nfnet_output_sizes = [256,512, 1536, 1536]

        print("Making Slow Layers")
        self.slow_layers = nn.ModuleList([
            NFNetStage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                padding = p,
                group_size=128, 
                alpha=alpha,
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n
            ) for k, f, i, o, e, s, n,p in zip(
                slow_nfnet_kernels,
                self.stage_downsamples,
                slow_nfnet_input_sizes,
                slow_nfnet_output_sizes,
                self.stage_expected_vars,
                self.stoch_depth_survival_probs,
                self.nfnet_stage_depths,
                slow_nfnet_padding
            )
        ])
        
        self.parallel_lastblock = parallel_lastblock
        
        self.slow_layers[-1] = ParallelNFNetStage(

                kernels=slow_nfnet_kernels[-1],
                freq_downsample=self.stage_downsamples[-1],
                input_channels=slow_nfnet_input_sizes[-1],
                output_channels=slow_nfnet_output_sizes[-1],
                group_size=128,
                alpha=alpha,
                input_expected_var=self.stage_expected_vars[-1],
                stoch_depths=self.stoch_depth_survival_probs[-1],
                num_blocks=self.nfnet_stage_depths[-1],
                padding = slow_nfnet_padding[-1],
                from_parallel = from_parallel,
                num_parallel = parallel_lastblock
            )

        # Construct NFNet stages for the fast path
        fast_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        fast_nfnet_padding = [
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]],
            [[0, 0],[0, 1],[1, 0],[0, 0]]
        ]
        fast_nfnet_input_sizes = [16,32, 64, 192]
        fast_nfnet_output_sizes = [32,64, 192, 192]

        print("Making Fast Layers")
        self.fast_layers = nn.ModuleList([
            NFNetStage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                group_size=16, 
                alpha=alpha, 
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n,
                padding = p
            ) for k, f, i, o, e, s, n,p in zip(
                fast_nfnet_kernels,
                self.stage_downsamples,
                fast_nfnet_input_sizes,
                fast_nfnet_output_sizes,
                self.stage_expected_vars,
                self.stoch_depth_survival_probs,
                self.nfnet_stage_depths,
                fast_nfnet_padding
            )
        ])
        
        # replace the last fast layer with an nfnet parallel stage
        
        self.fast_layers[-1] = ParallelNFNetStage(
                kernels=fast_nfnet_kernels[-1],
                freq_downsample=self.stage_downsamples[-1],
                input_channels=fast_nfnet_input_sizes[-1],
                output_channels=fast_nfnet_output_sizes[-1],
                group_size=16, 
                alpha=alpha, 
                input_expected_var=self.stage_expected_vars[-1],
                stoch_depths=self.stoch_depth_survival_probs[-1],
                num_blocks=self.nfnet_stage_depths[-1],
                padding = fast_nfnet_padding[-1],
                from_parallel = from_parallel,
                num_parallel = parallel_lastblock
            )

        print("Making Fusion Layers")
        # Construct fast-to-slow fusion layers
        self.fusion_layers = nn.ModuleList([
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=16, output_channels=128),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=32, output_channels=256),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=64, output_channels=512),
            FastToSlowFusion(time_kernel_length=7, time_stride=4, input_channels=192, output_channels=1536)
        ])

        # Construct summarization and aggregation layers at the output
        self.output_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        ])


        self.frontend = frontend
        self.embed_dim = 1728
        

    def forward(self, x):
        
        if isinstance(x, dict):
            wav = x['audio']
        else:
            wav = x
        
        spec = self.frontend(wav) if self.frontend is not None else wav
        
        
        slow = self.slow_stem(spec)
        fast = self.fast_stem(spec)

        # For each nfnet_transition module
        for fuse, slw_lyr, fst_lyr in zip(self.fusion_layers, self.slow_layers, self.fast_layers):
            slow = fuse(slow, fast)
            slow = slw_lyr(slow)
            fast = fst_lyr(fast)
        
        
        slow_out = [self.scaled_activation(self.output_layers[0](s)) for s in slow]
        fast_out = [self.scaled_activation(self.output_layers[1](f)) for f in fast]
        output = [torch.cat([s, f], dim=1) for s, f in zip(slow_out, fast_out)]
        output = [o.view(o.size(0), -1) for o in output]
        output = torch.stack(output, dim=0)
        
        
        return output

        