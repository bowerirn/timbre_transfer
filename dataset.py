import os
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm



PAIR_MAP = {
    "clar": "strings",
    "strings": "clar",
    "vibes": "piano",
    "piano": "vibes",
}


class StarnetImageDataset(Dataset):
    def __init__(
        self,
        format = "mel", # [mel, audio, cqt]
        npz_dir: str = "./data",
        target_inst: str = "strings",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Loads pre-cropped mel segments and builds (cond, target) pairs
        for a specific target instrument.

        Args:
            npz_path:    Path to the NPZ created by build_mel_crops.py
            target_inst: e.g., 'clar', 'strings', 'vibes', 'piano'
            dtype:       Torch dtype for tensors.
        """

        npz_path = f"{npz_dir}/{format}_crops.npz"

        if target_inst not in PAIR_MAP:
            raise ValueError(
                f"Unknown target_inst='{target_inst}'. Must be one of {list(PAIR_MAP.keys())}."
            )

        self.target_inst = target_inst
        self.cond_inst = PAIR_MAP[target_inst]
        self.dtype = dtype

        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        mels_np = data[f"{format}s"]              # (N, n_frames, n_mels)
        file_ids = data["file_ids"]         # (N,)
        insts = data["insts"]               # (N,)
        seg_ids = data["seg_ids"]           # (N,)

        # Cast to torch
        mels = torch.from_numpy(mels_np).to(dtype)  # (N, T, D)

        # Build index: (file_id, seg_id) -> {inst: mel_tensor}
        index = defaultdict(dict)
        N = mels.shape[0]
        for i in range(N):
            fid = str(file_ids[i])
            sid = int(seg_ids[i])
            inst = str(insts[i])
            key = (fid, sid)
            index[key][inst] = mels[i]  # (T, D)

        # Build all valid (cond, target) pairs for the chosen target_inst
        self.pairs = []  # list of (cond_mel, target_mel)
        for (fid, sid), inst_dict in index.items():
            if self.target_inst in inst_dict and self.cond_inst in inst_dict:
                cond = inst_dict[self.cond_inst]
                target = inst_dict[self.target_inst]
                self.pairs.append((cond, target))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid ({self.cond_inst} -> {self.target_inst}) pairs found in {npz_path}"
            )

        # Compute global mean / std over all pairs (like before)
        sum_vals = 0.0
        sum_sq_vals = 0.0
        count = 0

        for cond, target in self.pairs:
            sum_vals += cond.sum()
            sum_vals += target.sum()
            sum_sq_vals += (cond ** 2).sum()
            sum_sq_vals += (target ** 2).sum()
            count += cond.numel()
            count += target.numel()

        self.mu = (sum_vals / count).item()
        variance = (sum_sq_vals / count) - (self.mu ** 2)
        self.std = float(torch.sqrt(variance + 1e-8))

        print(
            f"Loaded {len(self.pairs)} pairs for target_inst='{self.target_inst}' "
            f"(cond_inst='{self.cond_inst}')"
        )
        print(f"Global mean: {self.mu:.4f}, std: {self.std:.4f}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cond, target = self.pairs[idx]  # each (T, D), T = n_frames

        cond = (cond - self.mu) / self.std
        target = (target - self.mu) / self.std

        # Already fixed direction: cond_inst -> target_inst
        return cond, target