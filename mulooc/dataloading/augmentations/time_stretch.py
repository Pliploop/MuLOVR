
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

from torch import Tensor
from typing import Optional
import librosa
import numpy as np
import torch



class TimeStretchAudiomentation(BaseWaveformTransform):
    
    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        max_stretch_rate: float = 1.2,
        min_stretch_rate: float = 0.8,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
        ):
        """
        :param sample_rate:
        :param min_delay_ms: Minimum delay in milliseconds (default 20.0)
        :param max_delay_ms: Maximum delay in milliseconds (default 100.0)
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param target_rate:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        if min_stretch_rate > max_stretch_rate:
            raise ValueError("max_stretch_rate must be > min_stretch_rate")
        if not sample_rate:
            raise ValueError("sample_rate is invalid.")
        self._sample_rate = sample_rate
        self._max_stretch_rate = max_stretch_rate
        self._min_stretch_rate = min_stretch_rate
        self._mode = mode

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        if self._mode == "per_example":
            # uniformsampling
            self.transform_parameters['stretch_rates'] = list(np.random.uniform(self._min_stretch_rate, self._max_stretch_rate, batch_size))
        
        elif self._mode == "per_batch":
            self.transform_parameters['stretch_rates'] = [np.random.uniform(self._min_stretch_rate, self._max_stretch_rate)] * batch_size
            
    def apply_transform(
        
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        if sample_rate is not None and sample_rate != self._sample_rate:
            raise ValueError(
                "sample_rate must match the value of sample_rate "
                + "passed into the TimeStretch constructor"
            )
        sample_rate = self.sample_rate

        if self._mode == "per_example":
            for i in range(batch_size):
                samples[i, ...] = self.stretch(
                    samples[i][None],
                    self.transform_parameters["stretch_rates"][i],
                    sample_rate,
                )[0]


        elif self._mode == "per_batch":
            samples = self.stretch(
                samples, 
                self.transform_parameters["stretch_rates"][0],
                sample_rate
            )

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        
    def stretch(self,samples: Tensor, stretch_rate = 1, sr = 22050) -> Tensor:
        
       # time stretch the signal and truncate to the original length
       
        new_audio = torch.zeros_like(samples)
        samples = librosa.effects.time_stretch(y = samples.cpu().numpy(), rate =  stretch_rate)
        if samples.shape[-1] < new_audio.shape[-1]:
            new_audio[...,:samples.shape[-1]] = torch.tensor(samples)
        else:
            new_audio = torch.tensor(samples[...,:new_audio.shape[-1]])
            
        return new_audio
       

class DiscreteTimeStretchAudioMentation(TimeStretchAudiomentation):
    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size, num_channels, num_samples = samples.shape

        if self._mode == "per_example":
            # discrete sampling
            stretch_rates = np.arange(self._min_stretch_rate, self._max_stretch_rate + 0.1, 0.1)
            self.transform_parameters['stretch_rates'] = list(np.random.choice(stretch_rates, batch_size))
        
        elif self._mode == "per_batch":
            stretch_rates = np.arange(self._min_stretch_rate, self._max_stretch_rate + 0.1, 0.1)
            self.transform_parameters['stretch_rates'] = [np.random.choice(stretch_rates)]
            
            
            
            
class BetaTimeStretchAudiomentation(TimeStretchAudiomentation):
    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size, num_channels, num_samples = samples.shape

        if self._mode == "per_example":
            # beta distribution sampling
            stretch_rates = np.random.beta(0.5, 0.5, batch_size) * (self._max_stretch_rate - self._min_stretch_rate) + self._min_stretch_rate
            self.transform_parameters['stretch_rates'] = list(stretch_rates)
        
        elif self._mode == "per_batch":
            stretch_rate = np.random.beta(0.5, 0.5) * (self._max_stretch_rate - self._min_stretch_rate) + self._min_stretch_rate
            self.transform_parameters['stretch_rates'] = [stretch_rate]
            
            
class LogUniformTimeStretchAudiomentation(TimeStretchAudiomentation):
    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size, num_channels, num_samples = samples.shape

    #log uniform sampling
    
        if self._mode == "per_example":
            stretch_rates = np.random.uniform(np.log(self._min_stretch_rate), np.log(self._max_stretch_rate), batch_size)
            stretch_rates = np.exp(stretch_rates)
            self.transform_parameters['stretch_rates'] = list(stretch_rates)
            
        elif self._mode == "per_batch":
            stretch_rate = np.random.uniform(np.log(self._min_stretch_rate), np.log(self._max_stretch_rate))
            stretch_rate = np.exp(stretch_rate)
            self.transform_parameters['stretch_rates'] = [stretch_rate]