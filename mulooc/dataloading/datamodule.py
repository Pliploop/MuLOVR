from mulooc.dataloading.dataset import AudioDataset
from mulooc.dataloading.datamodule_splitter import DataModuleSplitter
import torch
import pytorch_lightning as pl

from torch_audiomentations import *
from mulooc.dataloading.augmentations import *
from mulooc.dataloading.augmentations.composition.custom_compose import CustomCompose


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task=None,
        audio_dir=None,
        target_len_s=None,
        target_sr=44100,
        target_n_samples=None,
        augmentations={
            "base" : {
                "augs" : 
                    {"gain" : 
                        {"p" : 0.9}},
                "p" : 0.75},
            "var" : {
                "augs" : {},
                "p" : 0.75
            }
            },
        batch_size=32,
        num_workers=8,
        transform=False,
        n_augmentations=2,
        strategy_probs=[1, 0, 0],
        val_split=0.1,
        frontend = None,
        keep_anchor=False,
    ):
        super().__init__()
        self.task = task
        self.audio_dir = audio_dir
        assert (
            self.audio_dir is not None or self.task is not None
        ), "task and audio_dir cannot be None at the same time"

        self.splitter = DataModuleSplitter(audio_dir, task, val_split)

        self.target_len_s = target_len_s
        self.target_sr = target_sr
        self.target_n_samples = (
            target_n_samples
            if target_n_samples is not None
            else target_len_s * target_sr
        )

        if isinstance(augmentations, list):
            augmentations = {aug: {} for aug in augmentations}

        self.base_augs = augmentations.get('base', {}).get('augs', augmentations.get('augs', {}))
        self.base_p = augmentations.get('base', {}).get('p', augmentations.get('p', 0.75))
        self.var_augs = augmentations.get('var', {}).get('augs', [])
        self.var_p = augmentations.get('var', {}).get('p', 0)
        
        defaults = {
            "gain": {"min_gain_in_db": -15.0, "max_gain_in_db": 5.0, "p": 0.5, "sample_rate": self.target_sr},
            "polarity_inversion": {"p": 0.5, "sample_rate": self.target_sr},
            "add_colored_noise": {"p": 0.5, "sample_rate": self.target_sr, "min_snr_in_db": 3, "max_snr_in_db": 30, "min_f_decay": -2, "max_f_decay": 2},
            "filtering": {
                "bandpass": {"p": 0.5, "sample_rate": self.target_sr, "min_center_frequency": 200, "max_center_frequency": 4000, "min_bandwidth_fraction": 0.5, "max_bandwidth_fraction": 1.99},
                "bandstop": {"p": 0.5, "sample_rate": self.target_sr, "min_center_frequency": 200, "max_center_frequency": 4000, "min_bandwidth_fraction": 0.5, "max_bandwidth_fraction": 1.99},
                "highpass": {"p": 0.5, "sample_rate": self.target_sr, "min_cutoff_freq": 200, "max_cutoff_freq": min(0.5 * self.target_sr, 2400)},
                "lowpass": {"p": 0.5, "sample_rate": self.target_sr, "min_cutoff_freq": 75, "max_cutoff_freq": 7500}
                },
            "pitch_shift": {"p": 0.5, "sample_rate": self.target_sr, "min_transpose_semitones": -4, "max_transpose_semitones": 4},
            "delay": {"p": 0.5, "sample_rate": self.target_sr, "min_delay_ms": 100, "max_delay_ms": 500, "volume_factor": 0.5, "repeats": 2, "attenuation": 0.5},
            "timestretch": {"p": 0.5, "sample_rate": self.target_sr, "min_stretch_rate": 0.7, "max_stretch_rate": 1.3},
            "reverb": {"p": 0.5, "sample_rate": self.target_sr, "room_size": 0.2, "wet_level": 0.5, "dry_level": 0.5},
            "chorus": {"p": 0.5, "sample_rate": self.target_sr, "mix": 1, "rate_hz": 5, "depth": 1},
            "distortion": {"p": 0.5, "sample_rate": self.target_sr, "drive_db": 9},
            "compression": {"p": 0.5, "sample_rate": self.target_sr, "threshold_db": -30, "ratio": 5},
            "reverse": {"p": 0.5, "sample_rate": self.target_sr},
            "bitcrush": {"p": 0.5, "sample_rate": self.target_sr, "bit_depth": 4},
            "mp3": {"p": 0.5, "sample_rate": self.target_sr, "vbr_quality": 9},
        }
        
        
        self.augs = {
                'gain': lambda kwargs: Gain(**kwargs) ,
                'polarity_inversion': lambda p: PolarityInversion(p=0.5, sample_rate=self.target_sr),
                'add_colored_noise': lambda kwargs : AddColoredNoise(**kwargs),
                'filtering': lambda kwargs: OneOf([
                    BandPassFilter(**kwargs['bandpass']),
                    BandStopFilter(**kwargs['bandstop']),
                    HighPassFilter(**kwargs['highpass']),
                    LowPassFilter(**kwargs['lowpass']),
                ]),
                'pitch_shift': lambda kwargs: PitchShift(**kwargs),
                # 'delay': lambda p: Delay(p=0.5, sample_rate=self.target_sr, min_delay_ms=100 , max_delay_ms=500, volume_factor=0.5 , repeats=2 , attenuation=min(1,0.5 )),
                'timestretch': lambda kwargs: TimeStretch(**kwargs),
                'reverb' : lambda kwargs : ReverbAudiomentation(**kwargs),
                # 'chorus' : lambda p: ChorusAudiomentation(p=1, sample_rate=self.target_sr, mix = 1, rate_hz = 5, depth = 1),
                'distortion' : lambda kwargs: DistortionAudiomentation(**kwargs),
                # 'compression' : lambda p: CompressorAudiomentation(p=1, sample_rate=self.target_sr, threshold_db = -30, ratio = 5),
                # 'reverse' : lambda p: Reverse(p=1, sample_rate=self.target_sr),
                # 'bitcrush' : lambda p: BitcrushAudiomentation(p=1, sample_rate=self.target_sr, bit_depth = 4),
                # 'mp3' : lambda p: MP3CompressorAudiomentation(p=1, sample_rate=self.target_sr, vbr_quality = 9)
            }
        
        
        base_transforms = [self.augs[aug]({**defaults[aug], **self.base_augs[aug]}) for aug in self.base_augs]
        self.base_aug_chain = CustomCompose(
            transforms = base_transforms,
            p=self.base_p,
        )
        
        var_transforms = [self.augs[aug]({**defaults[aug], **self.var_augs[aug]}) for aug in self.var_augs]
        self.var_aug_chain = CustomCompose(
            transforms = var_transforms,
            p=self.var_p,
            )
        
        self.aug_chain = {
            "base": self.base_aug_chain,
            "var": self.var_aug_chain,
        }
        
        self.frontend=frontend
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.return_labels = self.task is not None
        self.n_augmentations = n_augmentations
        self.strategy_probs = strategy_probs
        self.keep_anchor = keep_anchor

        self.annotations = self.splitter.annotations

        self.train_annotations = self.annotations[self.annotations["split"] == "train"]
        self.val_annotations = self.annotations[self.annotations["split"] == "val"]
        self.test_annotations = self.annotations[self.annotations["split"] == "test"]

        print("Train annotations:", len(self.train_annotations))
        print("Val annotations:", len(self.val_annotations))
        print("Test annotations:", len(self.test_annotations))

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(
            annotations=self.train_annotations,
            target_len_s=self.target_len_s,
            target_sr=self.target_sr,
            target_n_samples=self.target_n_samples,
            augmentations=self.aug_chain,
            transform=self.transform,
            train=True,
            return_labels=self.return_labels,
            n_augmentations=self.n_augmentations,
            strategy_probs=self.strategy_probs,
            frontend=self.frontend,
            keep_anchor=self.keep_anchor
        )
        self.val_dataset = AudioDataset(
            annotations=self.val_annotations,
            target_len_s=self.target_len_s,
            target_sr=self.target_sr,
            target_n_samples=self.target_n_samples,
            augmentations=self.aug_chain,
            transform=True,
            train=True,
            return_labels=self.return_labels,
            n_augmentations=self.n_augmentations,
            strategy_probs=self.strategy_probs,
            frontend=self.frontend,
            keep_anchor=self.keep_anchor
        )
        if self.return_labels:
            self.test_dataset = AudioDataset(
                annotations=self.test_annotations,
                target_len_s=self.target_len_s,
                target_sr=self.target_sr,
                target_n_samples=self.target_n_samples,
                augmentations=None,
                transform=False,
                train=False,
                return_labels=self.return_labels,
                return_full=True,
                n_augmentations=1,
                strategy_probs=self.strategy_probs,
                frontend=self.frontend,
                keep_anchor=self.keep_anchor
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers
        )

    def dummy_call(self):
        dummy = self.train_dataset[0]
        for key in dummy:
            print(key, dummy[key].shape)
