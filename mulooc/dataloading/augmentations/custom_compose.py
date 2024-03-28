import random
from typing import List, Union, Optional, Tuple

from torch import Tensor
import torch.nn
import warnings

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.core.composition import BaseCompose

class CustomCompose(BaseCompose):
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform_names = [tfm.__class__.__name__ for tfm in self.transforms]
    
    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        transformed = {}
        if random.random() < self.p:
            print('hello')
            transform_indexes = list(range(len(self.transforms)))
            if self.shuffle:
                random.shuffle(transform_indexes)
            for i in transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    samples = inputs.samples
                    inputs = self.transforms[i](**inputs)
                    new_samples = inputs.samples
                    
                    # retroactively check which samples in the batch was changed
                    # changed is a boolean tensor of shape (batch_size,)
                    # samples is a tensor of shape (batch_size, num_channels, num_samples)
                    changed = (new_samples.sum(dim=(1, 2)) != samples.sum(dim=(1, 2))).int()
                    transformed[self.transform_names[i]] = changed
                    
                    print(f"Transform {self.transform_names[i]} changed {changed} samples")
                    
                    

                else:
                    assert isinstance(tfm, torch.nn.Module)
                    # FIXME: do we really want to support regular nn.Module?
                    inputs.samples = self.transforms[i](inputs.samples)

        return (inputs.samples, transformed) if self.output_type == "tensor" else (inputs, transformed)

