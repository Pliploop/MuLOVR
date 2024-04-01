import torchaudio
import soundfile as sf
import numpy as np
import torch

def load_audio_chunk(path, target_n_samples, target_sr, start = None):
    info = sf.info(path)
    frames = info.frames
    sr = info.samplerate
    
    
    if path.split('.')[-1] == 'mp3':
        frames = frames - 8192
    
    new_target_n_samples = int(target_n_samples * sr / target_sr)
    
    if start is None:
        # random
        start = np.random.randint(0, frames - new_target_n_samples)
        
    audio,sr = sf.read(path, start=start, stop=start+new_target_n_samples, always_2d=True, dtype='float32')
    audio = torch.tensor(audio.T)
    # resample to target sample rate
    
    # print(audio.shape)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    
    # print(audio.shape)    
    return audio

def load_full_audio(path, target_sr):
    audio, sr = sf.read(path, always_2d=True, dtype='float32')
    audio = torch.tensor(audio.T)
    # resample to target sample rate
    audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


def load_full_and_split(path, target_sr, target_n_samples, overlap = 0):
    audio = load_full_audio(path, target_sr)
    audio = audio.squeeze()    
    audio = audio.unfold(0, int(target_n_samples), int(target_n_samples) - int(target_n_samples*overlap)).unsqueeze(1)
    return audio