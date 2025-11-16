import torch
from dit import DiT
from flow_matching import FlowModel
from dataset import *
from utils import set_seed
from tqdm import trange


set_seed(9001)

batch_size = 200
epochs = 1000
n_frames = 2048
lr = 2e-4
n_test_examples = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

dit = DiT(
    input_dim = 128, 
    num_heads = 4, 
    head_dim = 32, 
    n_blocks = 4,
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
    batch_size=32, 
    # collate_fn=collate_trim
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


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