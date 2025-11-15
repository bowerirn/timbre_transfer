import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class StarnetDataset(Dataset):
    def __init__(self, path="./data", target_sr=16000, mono=True):
        self.path = path
        self.list = glob.glob(os.path.join(path, "*.wav"))
        self.mono = mono
        self.target_sr = target_sr

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        filepath = self.list[idx]
        basename = os.path.basename(filepath)
        # TODO: read the audio file and chop or pad it to the length of segment_len

        wav, sr = torchaudio.load(filepath)
        if self.mono and wav.ndim > 1:
            wav = wav.mean(dim=0)

        if sr != self.target_sr:
            x = torchaudio.functional.resample(wav, sr, self.target_sr).cpu()

        if len(wav) < self.segment_len:
            pad_len = self.segment_len - len(wav)
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        else:
            wav = wav[:self.segment_len]
        
        return wav,