
from . import PedalBoardAudiomentation
from pedalboard import Pedalboard, Reverb


class ReverbAudiomentation(PedalBoardAudiomentation):
    
    def __init__(self, sample_rate = 22050, mode='per_example', p=0.5, p_mode=None, output_type=None,randomize_parameters = True, *args, **kwargs):
        
        try:
            board  = Pedalboard([
                Reverb(**kwargs)
            ])
        except:
            board = Pedalboard([
                Reverb()
            ])
        super().__init__(board = board, mode = mode, p = p, p_mode = p_mode, sample_rate = sample_rate, output_type = output_type, randomize_parameters = randomize_parameters)
        
        self.set_kwargs(**kwargs)