from torch.utils.data import Dataset
import torch
from mulooc.dataloading.loading_utils import load_audio_chunk, load_full_and_split


class AudioDataset(Dataset):
    def __init__(
        self,
        annotations,
        target_len_s,
        target_sr,
        target_n_samples=None,
        augmentations=None,
        transform=False,
        train=True,
        return_labels=False,
        return_full=False,
        n_augmentations=2,
        strategy_probs=[1, 0, 0],
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

        self.strategy = {
            "same": strategy_probs[0],
            "adjacent": strategy_probs[1],
            "random": strategy_probs[2],
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = self.annotations.iloc[idx]["file_path"]
        if self.return_labels:
            labels = torch.tensor(self.annotations.iloc[idx]["labels"]).float()
        try:
            if self.return_full:
                audio = load_full_and_split(path, self.target_sr, self.target_n_samples)
                audio = audio.mean(dim=1, keepdim=True)
            else:
                # random choice of
                strategy = torch.multinomial(
                    torch.tensor(list(self.strategy.values())).float(), 1
                ).item()
                strategy = list(self.strategy.keys())[strategy]
                if strategy == "same":
                    audio = load_audio_chunk(
                        path, self.target_n_samples, self.target_sr
                    )
                    audio = torch.stack([audio] * self.n_augmentations)
                elif strategy == "adjacent":
                    audio = load_audio_chunk(
                        path,
                        self.target_n_samples * self.n_augmentations,
                        self.target_sr,
                    )
                    # unfold
                    audio = audio.unfold(
                        -1, self.target_n_samples, self.target_n_samples
                    )
                elif strategy == "random":
                    audios = []
                    for i in range(self.n_augmentations):
                        audios.append(
                            load_audio_chunk(
                                path, self.target_n_samples, self.target_sr
                            )
                        )
                    audio = torch.stack(audios)
                    
                audio = audio.mean(dim=1, keepdim=True)
        except Exception as e:
            print("Error loading file:", e)
            return self[idx + 1]

        if self.transform and self.train and self.augmentations is not None:
            if isinstance(self.augmentations, dict):
                audio,_ = self.augmentations['base'](audio)
                audio, augs = self.augmentations['var'](audio)
            else:
                audio, augs = self.augmentations(audio)
            
        print(augs)

        augs = augs if self.transform and self.train and self.augmentations is not None else torch.tensor([0])

        if self.return_labels:
            return {
                "audio": audio,
                "labels": labels,
                "augs": augs
            }
        else:
            return {
                "audio": audio,
                "augs": augs
            }
