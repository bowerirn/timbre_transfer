import os
import glob
import numpy as np
import librosa
from collections import defaultdict

# ==========================
# CONFIG CONSTANTS
# ==========================

# Audio / STFT geometry (must match your mel setup)
SR         = 24000
RESAMPLE_SR = 16000
WIN_LENGTH = 1024
HOP_LENGTH = 256

# Cropping in *mel frames*
N_FRAMES          = 256    # number of mel frames you used
SEGMENTS_PER_FILE = 25      # how many random crops per file-id group
SEED              = 9001

# I/O
INPUT_WAV_DIR   = "./data/wav"
OUTPUT_NPZ_PATH = "./data/audio_crops.npz"


# ==========================
# MAIN BUILD FUNCTION
# ==========================

def build_audio_crops():
    """
    Similar logic to the mel_crops script, but for raw audio segments.

    - Group WAVs by file-id (e.g. '001.clar.wav' -> file_id='001', inst='clar')
    - For each group, compute how many mel frames would be available
      given WIN_LENGTH and HOP_LENGTH, then take the min across instruments.
    - Require min_T >= N_FRAMES; otherwise skip that group.
    - Sample SEGMENTS_PER_FILE random start frames in [0, min_T - N_FRAMES],
      shared across instruments.
    - For each instrument, convert frame start -> sample start and extract
      a raw segment with the exact length corresponding to N_FRAMES frames.

    Output NPZ contains:
      - audios:   float32, shape (N, S)  where S = (N_FRAMES - 1)*HOP_LENGTH + WIN_LENGTH
      - file_ids: object/string, shape (N,)
      - insts:    object/string, shape (N,)
      - seg_ids:  int32, shape (N,)
      - n_frames: int32 scalar
      - hop_length: int32 scalar
      - win_length: int32 scalar
      - sr: int32 scalar
      - segment_length: int32 scalar  (number of samples in each audio crop)
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

    all_audios = []
    all_file_ids = []
    all_insts = []
    all_seg_ids = []

    # Length in samples for one crop corresponding to N_FRAMES mel frames
    segment_length = (N_FRAMES - 1) * HOP_LENGTH + WIN_LENGTH

    for file_id, inst_dict in groups.items():
        # Load audio for all instruments in this group
        y_by_inst = {}
        frame_counts = {}
        min_T = None

        for inst, path in inst_dict.items():
            y, sr = librosa.load(path, sr=SR, mono=True)
            if sr != SR:
                print(f"[WARN] Resampled {path} from {sr} to {SR}")

            # number of frames that STFT/mel with center=False would produce:
            # T = 1 + floor((len(y) - WIN_LENGTH) / HOP_LENGTH), for len(y) >= WIN_LENGTH
            if len(y) < WIN_LENGTH:
                print(
                    f"[SKIP inst] file_id={file_id}, inst={inst}: "
                    f"len(y)={len(y)} < WIN_LENGTH={WIN_LENGTH}"
                )
                continue

            T = 1 + (len(y) - WIN_LENGTH) // HOP_LENGTH
            if T <= 0:
                print(
                    f"[SKIP inst] file_id={file_id}, inst={inst}: "
                    f"T={T} <= 0 after frame-count calc"
                )
                continue

            y_by_inst[inst] = y.astype(np.float32)
            frame_counts[inst] = T
            min_T = T if min_T is None else min(min_T, T)

        if not y_by_inst:
            print(f"[SKIP group] file_id={file_id}: no valid instruments")
            continue

        if min_T is None or min_T < N_FRAMES:
            print(
                f"[SKIP group] file_id={file_id}: min_T={min_T} < N_FRAMES={N_FRAMES}"
            )
            continue

        max_start_frame = min_T - N_FRAMES

        # Random, but shared across instruments => aligned crops in "frame space"
        for seg_idx in range(SEGMENTS_PER_FILE):
            start_frame = rng.randint(0, max_start_frame + 1)

            # Convert frame index to sample index and crop raw audio
            sample_start = start_frame * HOP_LENGTH
            sample_end = sample_start + segment_length

            # Safety: make sure all instruments have enough samples
            # (they should, given the frame-count logic, but we'll guard anyway)
            valid = True
            for inst, y in y_by_inst.items():
                if sample_end > len(y):
                    print(
                        f"[WARN] file_id={file_id}, inst={inst}, seg_idx={seg_idx}: "
                        f"sample_end={sample_end} > len(y)={len(y)}. Skipping this segment."
                    )
                    valid = False
                    break
            if not valid:
                continue

            for inst, y in y_by_inst.items():
                seg = y[sample_start:sample_end]  # (segment_length,)
                seg = librosa.resample(seg, orig_sr=SR, target_sr=RESAMPLE_SR).astype(np.float32)
                seg = seg[:32768]
                all_audios.append(seg.astype(np.float32))
                all_file_ids.append(file_id)
                all_insts.append(inst)
                all_seg_ids.append(seg_idx)
    segment_length = 32768
    if len(all_audios) == 0:
        raise RuntimeError("No valid audio crops created; check N_FRAMES / data length")

    audios_arr = np.stack(all_audios, axis=0)  # (N, segment_length)
    file_ids_arr = np.array(all_file_ids)
    insts_arr = np.array(all_insts)
    seg_ids_arr = np.array(all_seg_ids, dtype=np.int32)

    n_frames_arr      = np.array([N_FRAMES], dtype=np.int32)
    hop_length_arr    = np.array([HOP_LENGTH], dtype=np.int32)
    win_length_arr    = np.array([WIN_LENGTH], dtype=np.int32)
    sr_arr            = np.array([SR], dtype=np.int32)
    segment_len_arr   = np.array([segment_length], dtype=np.int32)

    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        audios=audios_arr,
        file_ids=file_ids_arr,
        insts=insts_arr,
        seg_ids=seg_ids_arr,
        n_frames=n_frames_arr,
        hop_length=hop_length_arr,
        win_length=win_length_arr,
        sr=sr_arr,
        segment_length=segment_len_arr,
    )

    print(f"Saved {audios_arr.shape[0]} audio crops to {OUTPUT_NPZ_PATH}")
    print(f"Each crop: {segment_length} samples "
          f"(N_FRAMES={N_FRAMES}, HOP_LENGTH={HOP_LENGTH}, WIN_LENGTH={WIN_LENGTH})")


if __name__ == "__main__":
    build_audio_crops()
