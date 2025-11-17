# Modified inference script from https://github.com/NVIDIA/BigVGAN

device = 'cuda'

import torch
from BigVGAN import bigvgan
import numpy as np
import soundfile as sf

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)




def mel2wav(mel, out_fn, n_frames=2048):
    if isinstance(out_fn, str):
        out_fn = [out_fn]

    assert mel.shape[0] == len(out_fn), f"must have the same number of filenames as mels, but got {len(out_fn)} filenames and {mel.shape[0]} mels"

    print(mel.shape)
    
    with torch.inference_mode():
        wav = model(mel[..., :n_frames])  # [B, 1, T]
    wav = wav.squeeze(1).cpu()  # -> [B, T], mono float in [-1, 1]

    # Save as 16-bit PCM mono WAV
    for i in range(wav.shape[0]):
        sf.write(out_fn[i], wav[i, :].numpy(), 24000, subtype="PCM_16")
        print("Saved:", out_fn[i])

    return wav


def mel2wav_file(in_fn, out_fn, n_frames=2048):
    data = np.load(in_fn)
    mel = data["mel"]
    mel = torch.from_numpy(mel).unsqueeze(0).to(device)

    return mel2wav(mel, out_fn, n_frames=n_frames)


if __name__ == '__main__':
    in_fn = "data/mel_specs/001.clar.npz"
    out_fn = ["001_clar_recon.wav"]
    n_frames = 2048
    wav = mel2wav_file(in_fn, out_fn, n_frames=n_frames)
