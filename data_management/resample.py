# ChatGPT wrote this
import os
import torchaudio

input_dir = "./data/wav"
output_dir = input_dir  # overwrite

os.makedirs(output_dir, exist_ok=True)

target_sr = 24000 # SR for BigVGAN


print(f"Resampling all wav files to {target_sr} Hz")

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".wav"):
        continue

    path = os.path.join(input_dir, fname)

    # Load audio
    waveform, sr = torchaudio.load(path)

    # Convert to mono (average channels)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Save to output directory
    out_path = os.path.join(output_dir, fname)
    torchaudio.save(out_path, waveform, sample_rate=target_sr)

    print(f"Processed: {fname}")