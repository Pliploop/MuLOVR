from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

from torch import Tensor
from typing import Optional, Tuple, Union, List
from pathlib import Path
from random import choices
import torch
import numpy as np
import soundfile as sf
from mulooc.dataloading.loading_utils import *
import random
import os




class AddBackgroundNoiseAudiomentation(BaseWaveformTransform):
    """
    Add background noise to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        background_paths: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """

        :param background_paths: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        self.background_paths = []
        if isinstance(background_paths, (str, Path)):
            for root, _, files in os.walk(background_paths):
                for file in files:
                    if file.endswith(".wav") or file.endswith(".mp3"):
                        self.background_paths.append(os.path.join(root, file))
        
        self.sample_rate = sample_rate
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

    def random_background(self, target_num_samples: int) -> torch.Tensor:
        pieces = []

        # TODO: support repeat short samples instead of concatenating from different files

        background_samples=None
        
        while background_samples is None:
        
            background_path = random.choice(self.background_paths)
            info = sf.info(background_path)
            frames = info.frames
            sr = info.samplerate
            new_target_n_samples = int(target_num_samples * sr / self.sample_rate)
            if new_target_n_samples < frames:
            
                background_samples = load_audio_chunk(background_path, target_num_samples, self.sample_rate)
    
        pieces.append(background_samples)

        return rms_normalize(
            torch.cat([rms_normalize(piece) for piece in pieces], dim=1)
        )



    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """

        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # (batch_size, num_samples) RMS-normalized background noise
        self.transform_parameters["background"] = torch.stack(
            [self.random_background(num_samples) for _ in range(batch_size)]
        )

        # (batch_size, ) SNRs
        if self.min_snr_in_db == self.max_snr_in_db:
            self.transform_parameters["snr_in_db"] = torch.full(
                size=(batch_size,),
                fill_value=self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            )
        else:
            snr_distribution = torch.distributions.Uniform(
                low=torch.tensor(
                    self.min_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                high=torch.tensor(
                    self.max_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                validate_args=True,
            )
            self.transform_parameters["snr_in_db"] = snr_distribution.sample(
                sample_shape=(batch_size,)
            )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        background = self.transform_parameters["background"].to(samples.device)

        # (batch_size, num_channels)
        background_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        return ObjectDict(
            samples=samples
            + background_rms.unsqueeze(-1)
            * background.view(batch_size, 1, num_samples).expand(-1, num_channels, -1),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
        
def calculate_rms(samples):
    """
    Calculates the root mean square.

    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples), dim=-1, keepdim=False))

def rms_normalize(samples: Tensor) -> Tensor:
        """Power-normalize samples

        Parameters
        ----------
        samples : (..., time) Tensor
            Single (or multichannel) samples or batch of samples

        Returns
        -------
        samples: (..., time) Tensor
            Power-normalized samples
        """
        rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
        return samples / (rms + 1e-8)