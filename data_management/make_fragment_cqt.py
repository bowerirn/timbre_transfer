
import os
import glob
import numpy as np
import librosa
from collections import defaultdict
import sys
sys.path.append('.')



# ==========================
# CONFIG CONSTANTS
# ==========================

# BigVGAN / mel parameters
SR         = 24000
HOP_LENGTH = 256

N_BINS          = 84    # e.g. 7 octaves * 12 semitones
BINS_PER_OCTAVE = 12
FMIN            = 32.7  # ~ C1, adjust if you want

# Cropping parameters
N_FRAMES           = 512    # length in frames for each crop
SEGMENTS_PER_FILE  = 50      # how many random crops per file-id group
SEED               = 9001

# I/O paths
INPUT_WAV_DIR  = "./data/wav"
OUTPUT_NPZ_PATH = "./data/cqt_crops.npz"



# ==========================
# CQT COMPUTATION
# ==========================

def compute_cqt_logmag(y_np: np.ndarray) -> np.ndarray:
    """
    Compute log-magnitude CQT.

    Returns:
        cqt_db: np.ndarray, shape (n_bins, T)
    """
    C = librosa.cqt(
        y_np,
        sr=SR,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        center=False,
    )  # (n_bins, T), complex
    mag = np.abs(C)
    # Avoid log(0)
    mag = np.maximum(mag, 1e-7)
    cqt_db = librosa.amplitude_to_db(mag, ref=np.max).astype(np.float32)  # (n_bins, T)
    return cqt_db


# ==========================
# MAIN BUILD FUNCTION
# ==========================

def build_cqt_crops():
    """
    - Groups WAVs by file-id (e.g. '001.clar.wav' -> file_id='001', inst='clar')
    - For each group, computes CQT for all instruments.
    - Finds min_T across instruments in that group.
    - If min_T >= N_FRAMES, samples SEGMENTS_PER_FILE random start frames in
      [0, min_T - N_FRAMES] with a shared RNG, and extracts aligned crops from
      each instrument.

    Output NPZ contains:
      - cqts:           float32, shape (N, N_FRAMES, N_BINS)
      - file_ids:       object/string, shape (N,)
      - insts:          object/string, shape (N,)
      - seg_ids:        int32, shape (N,)
      - n_frames:       int32 scalar
      - n_bins:         int32 scalar
      - hop_length:     int32 scalar
      - sr:             int32 scalar
      - fmin:           float32 scalar
      - bins_per_octave:int32 scalar
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

    all_cqts = []
    all_file_ids = []
    all_insts = []
    all_seg_ids = []

    for file_id, inst_dict in groups.items():
        cqt_by_inst = {}
        min_T = None

        # Compute CQT for each instrument in this group
        for inst, path in inst_dict.items():
            y, sr = librosa.load(path, sr=SR, mono=True)
            if sr != SR:
                print(f"[WARN] Resampled {path} from {sr} to {SR}")

            cqt_db = compute_cqt_logmag(y)   # (n_bins, T)
            cqt_db = cqt_db.T                # (T, n_bins), time-first
            cqt_by_inst[inst] = cqt_db

            T = cqt_db.shape[0]
            min_T = T if min_T is None else min(min_T, T)

        if not cqt_by_inst:
            print(f"[SKIP group] file_id={file_id}: no valid instruments")
            continue

        if min_T is None or min_T < N_FRAMES:
            print(
                f"[SKIP group] file_id={file_id}: min_T={min_T} < N_FRAMES={N_FRAMES}"
            )
            continue

        max_start = min_T - N_FRAMES

        # Shared RNG -> same start frames as mel script if loop order + SEED match
        for seg_idx in range(SEGMENTS_PER_FILE):
            start = rng.randint(0, max_start + 1)

            for inst, cqt_TB in cqt_by_inst.items():
                seg = cqt_TB[start:start + N_FRAMES, :]  # (N_FRAMES, N_BINS)
                all_cqts.append(seg.astype(np.float32))
                all_file_ids.append(file_id)
                all_insts.append(inst)
                all_seg_ids.append(seg_idx)

    if len(all_cqts) == 0:
        raise RuntimeError("No valid CQT crops created; check N_FRAMES / data length")

    cqts_arr = np.stack(all_cqts, axis=0)  # (N, N_FRAMES, N_BINS)
    file_ids_arr = np.array(all_file_ids)
    insts_arr = np.array(all_insts)
    seg_ids_arr = np.array(all_seg_ids, dtype=np.int32)

    n_frames_arr       = np.array([N_FRAMES], dtype=np.int32)
    n_bins_arr         = np.array([N_BINS], dtype=np.int32)
    hop_length_arr     = np.array([HOP_LENGTH], dtype=np.int32)
    sr_arr             = np.array([SR], dtype=np.int32)
    fmin_arr           = np.array([FMIN], dtype=np.float32)
    bins_per_oct_arr   = np.array([BINS_PER_OCTAVE], dtype=np.int32)

    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        cqts=cqts_arr,
        file_ids=file_ids_arr,
        insts=insts_arr,
        seg_ids=seg_ids_arr,
        n_frames=n_frames_arr,
        n_bins=n_bins_arr,
        hop_length=hop_length_arr,
        sr=sr_arr,
        fmin=fmin_arr,
        bins_per_octave=bins_per_oct_arr,
    )

    print(f"Saved {cqts_arr.shape[0]} CQT crops to {OUTPUT_NPZ_PATH}")
    print(f"Each crop: (N_FRAMES={N_FRAMES}, N_BINS={N_BINS})")


if __name__ == "__main__":
    build_cqt_crops()
