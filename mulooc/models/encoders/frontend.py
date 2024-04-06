import torch
import torchaudio
from torch import nn

class Melgram(nn.Module):
    
    def __init__(self, n_mels = 96, n_fft = 2048, window_len = 400, hop_length = 160, sample_rate = 16000, f_min = 0, f_max = 8000, power = 2):
        super(Melgram, self).__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.window_len = window_len
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length = window_len,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        self.power = power
        
        stype = 'power' if self.power == 2 else 'magnitude'
        self.compressor = torchaudio.transforms.AmplitudeToDB(stype)
        
    def  forward(self, x):
        x = self.mel(x)
        x = self.compressor(x)
        return x