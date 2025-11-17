# ChatGPT wrote this
import os
import numpy as np
import librosa
import torch
import sys
sys.path.append('.')

from BigVGAN.meldataset import mel_spectrogram


# based on https://arxiv.org/pdf/2206.04658
SR = 24000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 100
FMIN = 0.0
FMAX = 12000.0
EPS = 1e-5  # for numerical stability


input_dir = "./data/wav"
output_dir = "./data/mel_specs"
os.makedirs(output_dir, exist_ok=True)

def compute_bigvgan_mel(y_np: np.ndarray) -> np.ndarray:
    # (B, T) tensor, BigVGAN expects batch dim
    y = torch.from_numpy(y_np).float().unsqueeze(0)

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
            center=False,   # matches their training code
        )
        # mel shape: [B, N_MELS, T]
        mel = mel.squeeze(0).cpu().numpy().astype(np.float32)  # (N_MELS, T)

    return mel

def process_file(in_path, out_path):
    y, sr = librosa.load(in_path, sr=SR, mono=True)
    mel = compute_bigvgan_mel(y)
    np.savez_compressed(out_path, mel=mel)



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
