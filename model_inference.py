import torch
from dit import DiT
from flow_matching import FlowModel
from dataset import *
from util import set_seed
from mel2wav import mel2wav

set_seed(9001)

batch_size = 200
n_frames = 2048
n_test_examples = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

dit = DiT(
    input_dim = 100, 
    num_heads = 4, 
    head_dim = 32, 
    n_blocks = 4,
    attn_pdrop = 0.1,
    causal = False,
    ff_scale = 2,
    cond_dim = None
)
model = FlowModel(dit, cfg_drop_prob=0.15).to(device)



ckpt_path = "mel_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location=device)

model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']}, step {ckpt['count']})")

model.eval()



dataset = StarnetImageDataset(n_frames=n_frames)
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - n_test_examples, n_test_examples])

test_loader = DataLoader(
    test_set, 
    batch_size=n_test_examples, 
    # collate_fn=collate_trim
)

with torch.no_grad():
    for _, (cond, target) in enumerate(test_loader):
        cond, target = cond.to(device), target.to(device)
        pred = model.sample(cond, steps=32, cfg_strength=5)# * dataset.std + dataset.mu

        pred_wav = mel2wav(pred.transpose(-2, -1), out_fn=[f"model_eval/mel/mel_pred_{i}.wav" for i in range(n_test_examples)])
        tgt_wav = mel2wav(target.transpose(-2, -1), out_fn=[f"model_eval/mel/mel_target_{i}.wav" for i in range(n_test_examples)])