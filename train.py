import torch
from dit import DiT
from flow_matching import FlowModel
from dataset import *
from util import set_seed
from tqdm import trange
from mel2wav import mel2wav



set_seed(9001)

batch_size = 200
epochs = 1000
n_frames = 2048
lr = 1e-3
n_test_examples = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

dit = DiT(
    input_dim = 100, 
    num_heads = 6, 
    head_dim = 32, 
    n_blocks = 8,
    attn_pdrop = 0.1,
    causal = False,
    ff_scale = 2,
    cond_dim = None
)
model = FlowModel(dit, cfg_drop_prob=0.15).to(device)








dataset = StarnetImageDataset(n_frames=n_frames)
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - n_test_examples, n_test_examples])

train_loader = DataLoader(
    train_set, 
    batch_size=32, 
    shuffle=True, 
    pin_memory=torch.cuda.is_available(),
    # collate_fn=collate_trim
)
test_loader = DataLoader(
    test_set, 
    batch_size=n_test_examples, 
    # collate_fn=collate_trim
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs * len(train_loader),
    eta_min=1e-6,
)


model.train()

epoch_losses = []
for epoch in range(epochs):
    
    with trange(len(train_loader), ascii=True) as t:

        loss_mavg = 0.0
        scaler = torch.amp.GradScaler('cuda')
        
        for i, (cond, target) in enumerate(train_loader):
            cond, target = cond.to(device), target.to(device)

            optimizer.zero_grad()

            loss = model(cond, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_mavg = (loss_mavg * i + loss) / (i + 1)
            # this is used to set a description in the tqdm progress bar 
            t.update(1)
            t.set_description(f"epoch: {epoch}, loss: {loss_mavg}")

    epoch_losses.append(loss_mavg.item())

torch.save({
    "epoch": epoch,
    "count": i,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss_mavg,
    "epoch_losses": epoch_losses,
}, "checkpoint.pth")


with torch.no_grad():
    for _, (cond, target) in enumerate(test_loader):
        cond, target = cond.to(device), target.to(device)
        pred = model.sample(cond, steps=50, cfg_strength=3) * dataset.std + dataset.mu

        pred_wav = mel2wav(pred.transpose(-2, -1), out_fn=[f"model_eval/mel/mel_pred_{i}.wav" for i in range(n_test_examples)])
        tgt_wav = mel2wav(target.transpose(-2, -1), out_fn=[f"model_eval/mel/mel_target_{i}.wav" for i in range(n_test_examples)])

import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
plt.plot(epoch_losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.show()