# ChatGPT wrote this
import os
import argparse
import numpy as np
import librosa


# based on https://arxiv.org/pdf/2307.04586
SR = 16000
N_FFT = 512
WIN_LENGTH = int(0.02 * SR)  # 20 ms -> 320 samples
HOP_LENGTH = WIN_LENGTH // 2 # 50% overlap -> 10 ms
N_MELS = 128
FMIN = 0.0
FMAX = 16000.0

input_dir = "./data/starnet_singles"        # folder containing .wav files
output_dir = "./data/mel_specs"       # folder to save .npz files
os.makedirs(output_dir, exist_ok=True)


def compute_log_mel(y, sr=SR):
    # Mel power spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    # Convert to log-mel (dB)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def normalize(S):
    """
    Min-max normalize spectrogram S to [-1, 1].
    Returns normalized S, and (min, max) so you can undo later.
    """
    S_min = S.min()
    S_max = S.max()
    if S_max == S_min:
        # Avoid division by zero: just return zeros
        return np.zeros_like(S, dtype=np.float32), float(S_min), float(S_max)

    S_norm01 = (S - S_min) / (S_max - S_min)  # [0, 1]
    S_norm = S_norm01 * 2.0 - 1.0            # [-1, 1]
    return S_norm.astype(np.float32), float(S_min), float(S_max)


def process_file(in_path, out_path):
    # Load audio as mono 16k
    y, sr = librosa.load(in_path, sr=SR, mono=True)

    # Compute log-mel
    mel_db = compute_log_mel(y, sr=sr)

    # Normalize full spectrogram to [-1, 1]
    mel_norm, s_min, s_max = normalize(mel_db)

    # Save as .npz (mel + min/max for potential denorm)
    np.savez_compressed(
        out_path,
        mel=mel_norm,
        min=s_min,
        max=s_max,
    )


wav_files = sorted(
    f for f in os.listdir(input_dir)
    if f.lower().endswith(".wav")
)

print(f"Found {len(wav_files)} wav files in {input_dir}")

for fname in wav_files:
    in_path = os.path.join(input_dir, fname)
    base, _ = os.path.splitext(fname)
    out_path = os.path.join(output_dir, base + ".npz")

    print(f"Processing {fname} -> {os.path.basename(out_path)}")
    process_file(in_path, out_path)

print("Done.")
