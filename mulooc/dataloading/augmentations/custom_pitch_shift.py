
from . import PedalBoardAudiomentation
from pedalboard import Pedalboard, PitchShift
from typing import Optional
from torch import Tensor
import numpy as np



# class DiscretePitchShiftAudiomentation(PedalBoardAudiomentation):
    
#     def __init__(self, board=None, sample_rate = 22050, mode='per_example', p=0.5, p_mode=None, output_type=None, randomize_parameters = True, *args, **kwargs):
#         try:
#             board = Pedalboard([
#                 PitchShift(**kwargs)
#             ])
#         except:
#             board = Pedalboard([
#                 PitchShift()
#             ])
        
#         super().__init__(board, mode, p, p_mode, sample_rate , output_type=output_type, randomize_parameters=randomize_parameters)
#         self.set_kwargs(**kwargs)
        
#     def randomize_parameters(self,
#         samples: Tensor = None,
#         sample_rate: Optional[int] = None,
#         targets: Optional[Tensor] = None,
#         target_rate: Optional[int] = None,):
        
        
#         batch_size, num_channels, num_samples = samples.shape
        
#         ## get lower and higher range and create discrete values
        
        
#         if self._randomize_parameters:
#             if self._mode == "per_example":
#                 for key in self.transform_ranges:
#                     transform_ranges = np.arange(self.transform_ranges[key][0], self.transform_ranges[key][1], 1)
#                     self.transform_parameters[key] = np.random.choice(transform_ranges, batch_size)
                        
#             elif self._mode == "per_batch":
#                 for key in self.transform_ranges:
#                     transform_ranges = np.arange(self.transform_ranges[key][0], self.transform_ranges[key][1], 1)
#                     self.transform_parameters[key] = np.random.choice(transform_ranges, 1)

class PitchShiftAudiomentation(PedalBoardAudiomentation):
    
    def __init__(self, board=None, sample_rate = 22050, mode='per_example', p=0.5, p_mode=None, output_type=None, randomize_parameters = True, *args, **kwargs):
        try:
            board = Pedalboard([
                PitchShift(**kwargs)
            ])
        except:
            board = Pedalboard([
                PitchShift()
            ])
        
        super().__init__(board, mode, p, p_mode, sample_rate , output_type=output_type, randomize_parameters=randomize_parameters)
        self.set_kwargs(**kwargs)
        
        print("PitchShiftAudiomentation")