import os
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

class StarnetWavDataset(Dataset):
    def __init__(self, path="./data/starnet_singles", window_secs=0.20):
        self.window_secs = window_secs

        files = glob.glob(os.path.join(path, "*.wav"))

        # Group by ID
        groups = defaultdict(dict)   # { "001": {"clar": "...", "strings": "...", ...} }

        for f in files:
            file_id, inst, _ = os.path.basename(f).split(".")  # ["001", "clar", "wav"]
            groups[file_id][inst] = f

        # Build list of (conditioning, target) pairs
        self.pairs = []

        for _, insts in groups.items():

            # clar <-> strings
            if "clar" in insts and "strings" in insts:
                self.pairs.append((insts["clar"], insts["strings"]))
                self.pairs.append((insts["strings"], insts["clar"]))

            # vibes <-> piano
            if "vibes" in insts and "piano" in insts:
                self.pairs.append((insts["vibes"], insts["piano"]))
                self.pairs.append((insts["piano"], insts["vibes"]))


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        cond_fn, target_fn = self.pairs[idx]

        cond, c_sr = torchaudio.load(cond_fn)
        target, t_sr = torchaudio.load(target_fn)

        assert c_sr == t_sr, f"Sample rates must match, but got {c_sr} for conditioning audio and {t_sr} for target audio"
        
        # take 20ms segment
        n_samples = self.window_secs * c_sr
        max_idx = min(cond.shape[0], target.shape[0]) - n_samples
        idx = torch.randint((1,), max_idx)
        
        return cond[idx:idx + n_samples], target[idx:idx + n_samples]
    






# class StarnetImageDataset(Dataset):
#     def __init__(self, path="./data/mel_specs", n_frames=128):
#         self.n_frames = n_frames

#         files = glob.glob(os.path.join(path, "*.npz"))

#         # Group by ID
#         groups = defaultdict(dict)   # { "001": {"clar": "...", "strings": "...", ...} }

#         for f in files:
#             file_id, inst, _ = os.path.basename(f).split(".")  # ["001", "clar", "npz"]
#             groups[file_id][inst] = f

#         # Build list of (conditioning, target) pairs
#         self.pairs = []

#         for _, insts in groups.items():

#             # clar <-> strings
#             if "clar" in insts and "strings" in insts:
#                 self.pairs.append((insts["clar"],   insts["strings"]))
#                 self.pairs.append((insts["strings"], insts["clar"]))

#             # vibes <-> piano
#             if "vibes" in insts and "piano" in insts:
#                 self.pairs.append((insts["vibes"], insts["piano"]))
#                 self.pairs.append((insts["piano"], insts["vibes"]))


#     def __len__(self):
#         return len(self.pairs)
    

#     def __getitem__(self, idx):
#         cond_fn, target_fn = self.pairs[idx]

#         cond = torch.from_numpy(np.load(cond_fn)["mel"]).float().T     # Shape: (T_cond, 128)
#         target = torch.from_numpy(np.load(target_fn)["mel"]).float().T     # Shape: (T_target, 128)

#         max_idx = min(cond.shape[0], target.shape[0]) - self.n_frames
#         idx = torch.randint(max_idx, size=(1,))

#         return cond[idx:idx + self.n_frames, :], target[idx:idx + self.n_frames, :]
    




class StarnetImageDataset(Dataset):
    def __init__(self, path="./data/mel_specs", n_frames=128, dtype=torch.float32):
        self.n_frames = n_frames
        self.dtype = dtype

        files = glob.glob(os.path.join(path, "*.npz"))

        # Group by ID -> instrument
        # { "001": {"clar": "...npz", "strings": "...npz", ...} }
        groups = defaultdict(dict)
        for f in files:
            file_id, inst, _ = os.path.basename(f).split(".")  # ["001", "clar", "npz"]
            groups[file_id][inst] = f

        # Simple cache so each file is only loaded once
        cache = {}

        def load_mel(path):
            if path not in cache:
                arr = np.load(path)["mel"]            # (128, T)
                tensor = torch.from_numpy(arr).to(dtype)  # (128, T)
                tensor = tensor.T                     # (T, 128)  time-first
                cache[path] = tensor
            return cache[path]

        # Build list of (A, B) mel Tensors (no flipped duplicates)
        self.pairs = []   # list of (mel_A, mel_B), each (T, 128)

        for _, insts in tqdm(groups.items(), ascii=True, desc="Reading in files"):
            # clar <-> strings
            if "clar" in insts and "strings" in insts:
                clar = load_mel(insts["clar"])
                strings = load_mel(insts["strings"])
                self.pairs.append((clar, strings))

            # vibes <-> piano
            if "vibes" in insts and "piano" in insts:
                vibes = load_mel(insts["vibes"])
                piano = load_mel(insts["piano"])
                self.pairs.append((vibes, piano))

        print("Done")

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid instrument pairs found in {path}")
        

        sum_vals = 0.0
        sum_sq_vals = 0.0
        count = 0

        for mel_a, mel_b in self.pairs:
            # mel_a, mel_b each shape: (T, n_mels)
            sum_vals += mel_a.sum()
            sum_vals += mel_b.sum()

            sum_sq_vals += (mel_a ** 2).sum()
            sum_sq_vals += (mel_b ** 2).sum()

            count += mel_a.numel()
            count += mel_b.numel()

        self.mu = (sum_vals / count).item()
        variance = (sum_sq_vals / count) - (self.mu ** 2)
        self.std = float(torch.sqrt(variance + 1e-8))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mel_a, mel_b = self.pairs[idx]   # each (T, 128)

        n_frames = self.n_frames

        min_T = min(mel_a.size(0), mel_b.size(0))
        if min_T <= self.n_frames:
            print(f"Not enough frames ({min_T}) for n_frames={self.n_frames} in pair index {idx}")
            n_frames = min_T

        # Random start index (inclusive) along time axis
        max_start = min_T - n_frames
        start = torch.randint(0, max_start + 1, (1,)).item()

        mel_a = mel_a[start:start + n_frames, :]  # (n_frames, 128)
        mel_b = mel_b[start:start + n_frames, :]  # (n_frames, 128)

        # Randomly choose direction: 50% A->B, 50% B->A
        if torch.rand(1).item() < 0.5:
            cond, target = mel_a, mel_b
        else:
            cond, target = mel_b, mel_a

        cond = (cond - self.mu) / self.std
        target = (target - self.mu) / self.std

        return cond, target
    





def collate_trim(batch):
    # batch is a list of (cond, target) pairs, each (T_i, D)
    conds, targets = zip(*batch)

    # shortest length in this batch
    min_len = min(c.size(0) for c in conds)

    cond_batch = torch.stack([c[:min_len] for c in conds], dim=0)   # (B, min_len, D)
    target_batch = torch.stack([t[:min_len] for t in targets], dim=0)

    return cond_batch, target_batch



def collate_pad(batch, pad_value=-1.0):
    conds, targets = zip(*batch)  # each is (T_i, D)

    lengths = [c.size(0) for c in conds]
    max_len = max(lengths)
    B = len(batch)
    D = conds[0].size(1)

    cond_batch   = conds[0].new_full((B, max_len, D), pad_value)
    target_batch = targets[0].new_full((B, max_len, D), pad_value)

    for i, (c, t) in enumerate(zip(conds, targets)):
        L = c.size(0)
        cond_batch[i, :L]   = c
        target_batch[i, :L] = t

    return cond_batch, target_batch
