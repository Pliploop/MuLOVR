import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, conditional=False, conditioning_size=None):
        super(MLP, self).__init__()
        self.conditional = conditional
        self.conditioning_size = conditioning_size

        if self.conditional:
            input_size += conditioning_size

        layers = nn.ModuleList()  # Use ModuleList instead of a regular list
        sizes = [input_size] + hidden_sizes
        layers = [nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
        activations = [nn.ReLU() for _ in range(len(layers) - 1)]
        layers_with_activation = list(sum(zip(layers, activations), ())) + [layers[-1]]

        self.model = nn.Sequential(*layers_with_activation)

    def forward(self, x, conditioning=None):
        if self.conditional:
            x = torch.cat((x, conditioning), dim=1)
        return self.model(x)