import random
from typing import List, Union, Optional, Tuple

from torch import Tensor
import torch.nn
import warnings

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.core.composition import BaseCompose

class CustomCompose(BaseCompose):
    """
    CustomCompose class for applying a series of transformations to input samples. and returning the transformed samples + whether they were transformed by a given augmentation.

    Args:
        **kwargs: Additional keyword arguments to be passed to the BaseCompose constructor.

    Attributes:
        transform_names (list): List of transformation names.
    """

    def __init__(
        self, return_tfms = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform_names = [tfm.__class__.__name__ for tfm in self.transforms]
        self.return_tfms = return_tfms
    
    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        Apply the series of transformations to the input samples.

        Args:
            samples (Tensor): Input samples.
            sample_rate (int, optional): Sample rate of the input samples.
            targets (Tensor, optional): Target samples.
            target_rate (int, optional): Sample rate of the target samples.

        Returns:
            Tuple[Tensor, ObjectDict]: Transformed samples and a dictionary of booleans indicating whether each sample was transformed by a given augmentation.
        """
        inputs = ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        transformed = ObjectDict()
        if random.random() < self.p:
            transform_indexes = list(range(len(self.transforms)))
            if self.shuffle:
                random.shuffle(transform_indexes)
            for i in transform_indexes:
                tfm = self.transforms[i]
                if isinstance(tfm, (BaseWaveformTransform, BaseCompose)):
                    samples = inputs.samples
                    inputs = self.transforms[i](**inputs)
                    
                    # changed = (new_samples.sum(dim=(1, 2)) != samples.sum(dim=(1, 2))).int()
                    
                    transformed = {}
                    if self.return_tfms:
                        changed = self.transforms[i].transform_parameters["should_apply"].int()
                        transformed[self.transform_names[i]] = changed  
                    
                    
                    # transform parameters are stored in the transform_parameters attribute of the transform
                    # should_apply is always the length of the batch
                    # other parameters are only the length of the number of samples that were transformed
                    # make the other parameter lists the length of the batch by filling non-transformed parameters with None
                    
                    
                else:
                    assert isinstance(tfm, torch.nn.Module)
                    inputs.samples = self.transforms[i](inputs.samples)
        else:
            for i in range(len(self.transforms)):
                transformed[self.transform_names[i]] = torch.zeros(inputs.samples.shape[0], dtype=torch.int)
        return (inputs.samples, transformed) if self.output_type == "tensor" else (inputs, transformed)

