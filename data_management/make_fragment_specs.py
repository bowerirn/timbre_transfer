import os
import glob
import numpy as np
import librosa
import torch
from collections import defaultdict
import sys
sys.path.append('.')

from BigVGAN.meldataset import mel_spectrogram


# ==========================
# CONFIG CONSTANTS
# ==========================

# BigVGAN / mel parameters
SR         = 24000
N_FFT      = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS     = 100
FMIN       = 0.0
FMAX       = 12000.0

# Cropping parameters
N_FRAMES           = 512    # length in frames for each crop
SEGMENTS_PER_FILE  = 25      # how many random crops per file-id group
SEED               = 9001

# I/O paths
INPUT_WAV_DIR  = "./data/wav"
OUTPUT_NPZ_PATH = "./data/mel_crops.npz"


# ==========================
# MEL COMPUTATION
# ==========================

def compute_bigvgan_mel(y_np: np.ndarray) -> np.ndarray:
    """
    Compute BigVGAN-style mel spectrogram.
    Returns array of shape (n_mels, T).
    """
    y = torch.from_numpy(y_np).float().unsqueeze(0)  # (1, T)

    with torch.no_grad():
        mel = mel_spectrogram(
            y,
            n_fft=N_FFT,
            num_mels=N_MELS,
            sampling_rate=SR,
            hop_size=HOP_LENGTH,
            win_size=WIN_LENGTH,
            fmin=FMIN,
            fmax=FMAX,
            center=False,
        )  # (1, n_mels, T)
        mel = mel.squeeze(0).cpu().numpy().astype(np.float32)  # (n_mels, T)
    return mel


# ==========================
# MAIN BUILD FUNCTION
# ==========================

def build_cropped_mels():
    """
    - Groups WAVs by file-id (e.g. '001.clar.wav' -> file_id='001', inst='clar')
    - For each group, computes mels for all instruments.
    - Randomly samples SEGMENTS_PER_FILE crops of length N_FRAMES along time
      using a shared set of start indices (aligned crops across instruments).
    - Saves all crops and labels into a single NPZ.

    Output NPZ contains:
      - mels:     float32, shape (N, N_FRAMES, N_MELS)
      - file_ids: object/string, shape (N,)
      - insts:    object/string, shape (N,)
      - seg_ids:  int32, shape (N,)
      - n_frames: int32 scalar
      - n_mels:   int32 scalar
    """
    wav_files = glob.glob(os.path.join(INPUT_WAV_DIR, "*.wav"))
    if len(wav_files) == 0:
        raise RuntimeError(f"No .wav files found in {INPUT_WAV_DIR}")

    # Group by file ID -> instrument
    # Expected filenames like "001.clar.wav"
    groups = defaultdict(dict)
    for f in wav_files:
        base = os.path.basename(f)
        stem, _ = os.path.splitext(base)  # "001.clar"
        try:
            file_id, inst = stem.split(".")
        except ValueError:
            raise RuntimeError(
                f"Expected filenames like '001.clar.wav', got '{base}'"
            )
        groups[file_id][inst] = f

    print(f"Found {len(groups)} file-id groups in {INPUT_WAV_DIR}")

    rng = np.random.RandomState(SEED)

    all_mels = []
    all_file_ids = []
    all_insts = []
    all_seg_ids = []

    for file_id, inst_dict in groups.items():
        # Compute mels for all instruments in this group
        mel_by_inst = {}
        min_T = None

        for inst, path in inst_dict.items():
            y, sr = librosa.load(path, sr=SR, mono=True)
            if sr != SR:
                print(f"[WARN] Resampled {path} from {sr} to {SR}")

            mel = compute_bigvgan_mel(y)  # (n_mels, T)
            mel = mel.T  # (T, n_mels), time-first
            mel_by_inst[inst] = mel

            T = mel.shape[0]
            min_T = T if min_T is None else min(min_T, T)

        if min_T is None or min_T < N_FRAMES:
            print(
                f"[SKIP] file_id={file_id}: min_T={min_T} < N_FRAMES={N_FRAMES}"
            )
            continue

        max_start = min_T - N_FRAMES

        # Random, but shared across instruments => aligned crops
        for seg_idx in range(SEGMENTS_PER_FILE):
            start = rng.randint(0, max_start + 1)

            for inst, mel in mel_by_inst.items():
                seg = mel[start:start + N_FRAMES, :]  # (N_FRAMES, N_MELS)
                all_mels.append(seg.astype(np.float32))
                all_file_ids.append(file_id)
                all_insts.append(inst)
                all_seg_ids.append(seg_idx)

    if len(all_mels) == 0:
        raise RuntimeError("No valid crops created; check N_FRAMES / data length")

    mels_arr = np.stack(all_mels, axis=0)  # (N, N_FRAMES, N_MELS)
    file_ids_arr = np.array(all_file_ids)
    insts_arr = np.array(all_insts)
    seg_ids_arr = np.array(all_seg_ids, dtype=np.int32)
    n_frames_arr = np.array([N_FRAMES], dtype=np.int32)
    n_mels_arr = np.array([N_MELS], dtype=np.int32)

    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        mels=mels_arr,
        file_ids=file_ids_arr,
        insts=insts_arr,
        seg_ids=seg_ids_arr,
        n_frames=n_frames_arr,
        n_mels=n_mels_arr,
    )

    print(f"Saved {mels_arr.shape[0]} crops to {OUTPUT_NPZ_PATH}")
    print(f"Each crop: (N_FRAMES={N_FRAMES}, N_MELS={N_MELS})")


if __name__ == "__main__":
    build_cropped_mels()