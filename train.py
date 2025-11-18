import torch
from dit import DiT
from flow_matching import FlowModel
from dataset import *
from util import set_seed
from tqdm import trange
import matplotlib.pyplot as plt
from model_inference import eval
import torch.nn as nn


def init_dit(model: DiT):
    """
    Initialize DiT weights for stable flow-matching training.
    Call *after* constructing the DiT instance:
        dit = DiT(...)
        init_dit(dit)
    """

    # 1) Default: Xavier for all Linear layers, bias = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # 2) Zero-init proj_out (residual flow output)
    nn.init.zeros_(model.proj_out.weight)
    nn.init.zeros_(model.proj_out.bias)

    # 3) Zero-init AdaLN output layers in each block
    for block in model.blocks:
        # block.adaln = GELU -> Linear(cond_dim, 9 * input_dim)
        last = block.adaln[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    # 4) (Optional but nice) soften t_mlp final layer
    #    If you want, you can also zero its last layer:
    last_t = model.t_mlp[-1]
    if isinstance(last_t, nn.Linear):
        nn.init.zeros_(last_t.weight)
        nn.init.zeros_(last_t.bias)



set_seed(9001)

batch_size = 10
epochs = 3000
n_frames = 512
lr = 1e-3
n_test_examples = 10
n_test_epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

dit = DiT(
    input_dim = 100, 
    num_heads = 4, 
    head_dim = 32, 
    n_blocks = 6,
    attn_pdrop = 0.0,
    causal = False,
    ff_scale = 2,
    cond_dim = None
)
# init_dit(dit)

model = FlowModel(dit, cfg_drop_prob=0.15).to(device)





dataset = StarnetImageDataset(n_frames=n_frames)
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - n_test_examples, n_test_examples])

train_loader = DataLoader(
    train_set, 
    batch_size=batch_size, 
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
fads = []
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

    if epoch % n_test_epochs == 0:
        fads.append(eval(model, test_loader, dataset))

    if (epoch + 1) % 500 == 0:
        torch.save({
            "epoch": epoch,
            "count": i,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_mavg,
            "epoch_losses": epoch_losses,
            "fads": fads,
        }, f"mel_ckpt_e{epoch}.pth")

fads.append(eval(model, test_loader, dataset))

torch.save({
    "epoch": epoch,
    "count": i,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss_mavg,
    "epoch_losses": epoch_losses,
    "fads": fads,
}, "mel_ckpt.pth")


x = np.arange(0, epoch + 2, n_test_epochs)
plt.figure(figsize=(7,4))
plt.plot(x, fads, marker='o')
plt.title("FAD through training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("mel_fad.png", dpi=200)
plt.show(block=False)


plt.figure(figsize=(7,4))
plt.plot(epoch_losses)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("mel_loss.png", dpi=200)
plt.show(block=False)