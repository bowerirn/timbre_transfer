# cqt2wav_speechbrain.py

import os
import numpy as np
import torch
import soundfile as sf

from speechbrain.inference.vocoders import DiffWaveVocoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 1. Load DiffWave vocoder (SpeechBrain)
# ------------------------------------------------------------------
# NOTE: For real CQT->wav, you should train your own DiffWaveVocoder
# on CQT features. This source is mel-based (LJSpeech) by default.
diffwave = DiffWaveVocoder.from_hparams(
    source="speechbrain/tts-diffwave-ljspeech",  # replace with your CQT-trained model if/when you have one
    savedir="pretrained/diffwave",
)
diffwave = diffwave.to(device)

# These must match the training config of the vocoder.
SR_VOCODER = 22050    # 22050 for tts-diffwave-ljspeech; change if yours is 16k/24k/etc
HOP_LENGTH = 256      # hop length used for feature extraction


# ------------------------------------------------------------------
# 2. Core function: CQT -> wav (batch)
# ------------------------------------------------------------------

def cqt2wav(cqt, out_fn, n_frames=None, sr=SR_VOCODER, fast_sampling=True):
    """
    Args:
        cqt: torch.Tensor of shape (B, T, n_bins) or (B, n_bins, T)
             Must match what your DiffWave model was trained on (CQT config & normalization).
        out_fn: str or list[str] of output paths
        n_frames: optional int, crop along time dim
        sr: output sample rate (must match vocoder training)
        fast_sampling: whether to use DiffWave fast sampling (if supported)
    Returns:
        wav: torch.Tensor, shape (B, T_samples)
    """
    if isinstance(out_fn, str):
        out_fn = [out_fn]

    assert cqt.shape[0] == len(out_fn), (
        f"Need one filename per batch element, got {len(out_fn)} vs batch={cqt.shape[0]}"
    )

    # Ensure shape is (B, n_bins, T) as expected by DiffWaveVocoder.decode_batch
    if cqt.dim() != 3:
        raise ValueError(f"Expected CQT with 3 dims (B, T, BINS) or (B, BINS, T), got {cqt.shape}")

    B, A, B_ = cqt.shape
    # Heuristic: if second dim > third dim, assume time-first (B, T, BINS)
    # You can replace this with an explicit flag if you prefer.
    if A > B_:
        # (B, T, n_bins) -> (B, n_bins, T)
        cqt = cqt.permute(0, 2, 1)

    if n_frames is not None:
        cqt = cqt[:, :, :n_frames]  # crop in time dim

    cqt = cqt.to(device).float()   # [B, n_bins, T]

    # SpeechBrain DiffWaveVocoder API:
    #   waveforms = diffwave.decode_batch(mel, hop_len=HOP_LENGTH, ...)
    #   mel is [batch, mels, time]; here we treat CQT as "mel".
    with torch.no_grad():
        waveforms = diffwave.decode_batch(
            cqt,
            hop_len=HOP_LENGTH,
            fast_sampling=fast_sampling,
            # you can pass a custom noise schedule here if you want
            # fast_sampling_noise_schedule=[...]
        )
        # waveforms: [B, 1, T_samples]
        waveforms = waveforms.squeeze(1).cpu()  # [B, T_samples]

    # Save
    for i in range(waveforms.shape[0]):
        path = out_fn[i]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, waveforms[i].numpy(), sr, subtype="PCM_16")
        print("Saved:", path)

    return waveforms


# ------------------------------------------------------------------
# 3. Convenience wrapper: load CQT from NPZ and vocode
# ------------------------------------------------------------------

def cqt2wav_file(in_fn, out_fn, n_frames=None, sr=SR_VOCODER, fast_sampling=True):
    """
    Load CQT(s) from an NPZ file and convert to waveform(s).

    Expected NPZ formats:
      - 'cqt':  (T, n_bins) or (n_bins, T)
      - 'cqts': (N, T, n_bins) or (N, n_bins, T)   # like your cqt_crops.npz
    """
    data = np.load(in_fn)

    if "cqt" in data:
        cqt_np = data["cqt"]  # (T, n_bins) or (n_bins, T)
        if cqt_np.ndim == 2:
            cqt_np = cqt_np[None, ...]  # (1, T, n_bins) or (1, n_bins, T)
    elif "cqts" in data:
        cqt_np = data["cqts"]  # (N, T, n_bins) or (N, n_bins, T)
    else:
        raise KeyError(f"{in_fn} must contain 'cqt' or 'cqts' array")

    cqt = torch.from_numpy(cqt_np).float()
    return cqt2wav(cqt, out_fn, n_frames=n_frames, sr=sr, fast_sampling=fast_sampling)


# ------------------------------------------------------------------
# 4. Example usage (single file)
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Example: one CQT saved as npz
    in_fn = "data/cqt_specs/001.clar.npz"
    out_fn = ["results/cqt_diffwave/001_clar_recon.wav"]

    n_frames = None  # or e.g. 512 if you want to crop in time

    wav = cqt2wav_file(in_fn, out_fn, n_frames=n_frames, sr=SR_VOCODER, fast_sampling=True)
