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
        self, return_tfms = False, track_params = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform_names = [tfm.__class__.__name__ for tfm in self.transforms]
        self.return_tfms = return_tfms
        self.track_params = False
        
    
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
                    
                    if self.return_tfms:
                        # each transform has a ditionary attribute trackable_params, for which the keys are the names of the parameters that are tracked and the values are the tolerance values and the default
                        # for each parameter, tolerance pair, if the parameter changes by more than the tolerance, the sample is considered transformed
                        # if the tolerance is none, parameters have to be exactly equal to be considered NOT transformed
                        if self.track_params and hasattr(self.transforms[i], "trackable_params"):
                            changed_matrix = torch.zeros(inputs.samples.shape[0], inputs.samples.shape[0], dtype=torch.int)
                            for param_name, tolerance in self.transforms[i].trackable_params.items():
                                # this tracks whether or not a sample was changed relative to other samples
                                if tolerance is not None:
                                    # a square matrix of relative parameter differences
                                    diff_matrix = (self.transforms[i].transform_parameters[param_name].unsqueeze(1) - self.transforms[i].transform_parameters[param_name])
                                    tol_changed = torch.abs(diff_matrix) > tolerance
                                    changed_matrix = changed_matrix * tol_changed
                                else:
                                    changed_matrix = changed_matrix * (self.transforms[i].transform_parameters[param_name].unsqueeze(1) != self.transforms[i].transform_parameters[param_name])
                                
                                changed = changed_matrix
                                print(changed.shape)
                                    
                        else :
                            
                            changed = self.transforms[i].transform_parameters["should_apply"].int()
                        
                        transformed[self.transform_names[i]] = changed
                    
                    
                else:
                    assert isinstance(tfm, torch.nn.Module)
                    inputs.samples = self.transforms[i](inputs.samples)
                    
            
        else:
            for i in range(len(self.transforms)):
                transformed[self.transform_names[i]] = torch.zeros(inputs.samples.shape[0], dtype=torch.int)
                
        
        if self.return_tfms:
            for transform in self.transforms:
                    for key in transform.transform_parameters:
                        new_params = transform.transform_parameters[key]
                        if key != "should_apply":
                            
                            for i in range(len(transform.transform_parameters['should_apply'])):
                                if not transform.transform_parameters['should_apply'][i]:
                                    # insert None for non-transformed samples
                                    new_params.insert(i, None)
                            transform.transform_parameters[key] = new_params
                
                
        return (inputs.samples, transformed) if self.output_type == "tensor" else (inputs, transformed)

