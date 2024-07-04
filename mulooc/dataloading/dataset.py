from torch.utils.data import Dataset
import torch
from mulooc.dataloading.loading_utils import load_audio_chunk, load_full_and_split
from pedalboard import time_stretch
import numpy as np
import pickle

def load_same(path, target_n_samples, target_sr, n_augmentations):
    audio = load_audio_chunk(path, target_n_samples, target_sr)
    return torch.stack([audio] * n_augmentations)

def load_adjacent(path, target_n_samples, target_sr, n_augmentations):
    audio = load_audio_chunk(path, target_n_samples * n_augmentations, target_sr)
    audio = audio.unfold(-1, target_n_samples, target_n_samples)
    return audio.permute(1, 0, 2)

def load_random(path, target_n_samples, target_sr, n_augmentations):
    return torch.stack([load_audio_chunk(path, target_n_samples, target_sr) for _ in range(n_augmentations)])


class AudioDataset(Dataset):
    def __init__(
        self,
        annotations,
        target_len_s,
        target_sr,
        target_n_samples=None,
        max_target_n_samples = None,
        augmentations=None,
        transform=False,
        train=True,
        return_labels=False,
        return_full=False,
        n_augmentations=2,
        strategy_probs=[1, 0, 0],
        frontend = None,
        keep_anchor = False,
        tempo_stretching = False,
        return_tfm_parameters = False,
        return_clean_audio = False,
        extract_features = False,
        mono = True,
    ):
        self.annotations = annotations
        self.target_len_s = target_len_s
        self.target_sr = target_sr
        self.target_n_samples = (
            target_n_samples
            if target_n_samples is not None
            else target_len_s * target_sr
        )
        self.transform = transform
        self.augmentations = augmentations
        self.train = train
        self.return_labels = return_labels
        self.return_full = return_full  # return full audio file for test dataloader
        self.n_augmentations = n_augmentations
        self.keep_anchor = keep_anchor
        self.return_tfm_parameters = return_tfm_parameters
        self.return_clean_audio = return_clean_audio
        self.extract_features = extract_features
        self.max_target_n_samples = max_target_n_samples
        self.mono = mono
        
        self.strategy = {
            "same": strategy_probs[0],
            "adjacent": strategy_probs[1],
            "random": strategy_probs[2],
        }
        
        self.frontend = frontend
        
        self.strategy_funcs = {
            "same": load_same,
            "adjacent": load_adjacent,
            "random": load_random,
        }
        
        self.tempo_stretching = tempo_stretching
        self.mode = "per_example"
        
        self.strategy_values = torch.tensor(list(self.strategy.values())).float()

    def __len__(self):
        return len(self.annotations)
    
    def set_aug_mode(self, mode = 'per_batch'):
        # modes per batch or per example
        for tfm in self.augmentations['var'].transforms:
            tfm._mode = mode
            print(tfm._mode)

        self.mode = mode

    def __getitem__(self, idx):
        path = self.annotations.iloc[idx]["file_path"]

        if self.return_labels:
            labels = torch.tensor(self.annotations.iloc[idx]["labels"]).float()
        try:
            if self.return_full:
                audio = load_full_and_split(path, self.target_sr, self.target_n_samples, mono = self.mono)
                audio = audio.mean(dim=1, keepdim=True)
                if self.frontend and not self.extract_features:
                    audio = audio.unsqueeze(1)
                
            else:
                
                strategy = torch.multinomial(self.strategy_values, 1).item()
                strategy = list(self.strategy.keys())[strategy]
                audio = self.strategy_funcs[strategy](path, self.target_n_samples, self.target_sr, self.n_augmentations)
                if self.mono:
                    audio = audio.mean(dim=1, keepdim=True)
        except Exception as e:
            print("Error loading file:", e)
            return self[idx + 1]
        
        clean_audio = audio.clone()
        
        if self.transform and self.train and self.augmentations is not None:
            if self.keep_anchor and self.n_augmentations > 1:
                anchor = audio[0:1,...]
            if isinstance(self.augmentations, dict):
                audio,_ = self.augmentations['base'](audio)
                audio, augs = self.augmentations['var'](audio)
            else:
                audio, augs = self.augmentations(audio)
            if self.keep_anchor and self.n_augmentations > 1:
                audio[0:1,...] = anchor

        augs = augs if self.transform and self.train and self.augmentations is not None else {
            "none": torch.tensor([0]*self.n_augmentations)
        }
        
        if self.return_labels and self.tempo_stretching and self.train: #only for tempo datasets augmentation
            audio, labels = time_stretching_module(audio, self.target_sr, self.target_n_samples, labels)
            if audio is None:
                return self[idx + 1]
            
        if self.max_target_n_samples:
            audio = audio[:,:,:self.max_target_n_samples]
            if self.return_clean_audio:
                clean_audio = clean_audio[:,:,:self.max_target_n_samples]
            
        
        if self.frontend:
            audio = self.frontend(audio)
            if self.return_clean_audio:
                clean_audio = self.frontend(clean_audio)
            if audio.dim() == 3:
                audio = audio.unsqueeze(0)
                if self.return_clean_audio:
                    clean_audio = clean_audio.unsqueeze(0)
            
    
                
        if self.augmentations and self.return_tfm_parameters:
            transform_parameters = {tfm.__class__.__name__  : tfm.transform_parameters for tfm in self.augmentations['var'].transforms}
            # recursively go through the dict and turn lists into tensors
            for key in transform_parameters:
                for key2 in transform_parameters[key]:
                    if isinstance(transform_parameters[key][key2], list):
                        transform_parameters[key][key2] = torch.tensor(transform_parameters[key][key2])
        
        #truncate audio to max_target_n_samples
        
        output_ = {
            "audio": audio,
            "augs": augs,
        }
        
        if self.return_labels:
            output_["labels"] = labels
            
        if self.return_tfm_parameters:
            output_["transform_parameters"] = transform_parameters
            
        if self.return_clean_audio:
            output_["clean_audio"] = clean_audio
        
        return output_

def time_stretching_module(input_audio, samplerate, factor, labels):
    
    factor = np.random.uniform(0.8, 1.2)
    tempo = labels.argmax()
    new_tempo = int(tempo * factor)
    labels = torch.zeros_like(labels)
    
    if new_tempo < labels.shape[0]:
        labels[new_tempo] = 1
    else:
        return None,None
    
    original_shape = input_audio.shape
    
    audio = input_audio.squeeze(0)
    audio = time_stretch(input_audio = audio.numpy(), samplerate = samplerate, stretch_factor = factor)
    audio = torch.tensor(audio).unsqueeze(0)
    
    audio = torch.nn.functional.pad(audio, (0, max(0, original_shape[2] - audio.shape[2])))
    audio = audio[:,:,:original_shape[2]]

    
    
    return audio, labels



class PrecomputedDataset(Dataset):
    def __init__(self, data, mean = False):
        #file path is a picklable object
        self.annotations = data # dictionary of paths to npy files
        if not mean:
            self.clean_embeddings = np.load(self.annotations['clean_audio_embeddings_path'], mmap_mode='r')
            self.transformed_embeddings = np.load(self.annotations['audio_embeddings_path'], mmap_mode='r')
        else:
            self.clean_embeddings = np.load(self.annotations['mean_clean_audio_embeddings_path'], mmap_mode='r')
            self.transformed_embeddings = np.load(self.annotations['mean_audio_embeddings_path'], mmap_mode='r')
        self.mean = mean
        if mean:
            self.annotations['param'] = self.annotations['mean_target_param']
        else:
            self.annotations['param'] = self.annotations['target_param']
            
        # restrict the dataset to incides where param is between -6 and 6
        # indices = np.where(np.logical_and(self.annotations['param'] >= -6, self.annotations['param'] <= 6))
        # self.annotations['param'] = self.annotations['param'][indices]
        # self.clean_embeddings = self.clean_embeddings[indices]
        # self.transformed_embeddings = self.transformed_embeddings[indices]
            
        print(f'length of dataset: {len(self)}')
    
    def __len__(self):
        
        dummy_key = list(self.annotations['param'].keys())[0]
        len_ = len(self.annotations['param'][dummy_key])
        
        return len_

    def __getitem__(self, idx):
        
        # memmap open the file in read only mode
        transform_parameters = self.annotations['param']
        
        
        clean_embeddings = self.clean_embeddings[idx,...]
        transformed_embeddings = self.transformed_embeddings[idx,...]
        transform_parameters = {
            k : torch.tensor(transform_parameters[k][idx]) for k in transform_parameters
        }
        
        clean_embeddings = torch.tensor(clean_embeddings)
        transformed_embeddings = torch.tensor(transformed_embeddings)
        # transform_parameters = torch.tensor(transform_parameters, dtype=torch.float32)
        
        
        return {
            "clean_audio": clean_embeddings,
            "audio": transformed_embeddings,
            "transform_parameters": transform_parameters,
        }