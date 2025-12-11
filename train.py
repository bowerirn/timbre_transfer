import torch
from dit import DiT, init_dit
from flow_matching import FlowModel
from dataset import *
from util import set_seed
from tqdm import trange
import matplotlib.pyplot as plt
from model_inference import eval
import torch.nn as nn





def train(
    seed = 9001,
    batch_size = 100,
    epochs = 2000,
    lr = 5e-4,
    n_test_examples = 10,
    n_test_epochs = 50,
    inst = 'piano',
):




    set_seed(seed)

    batch_size = 100
    

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





    dataset = StarnetImageDataset(target_inst=inst)
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
            fads.append(eval(model, test_loader, dataset, inst))

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

    fads.append(eval(model, test_loader, dataset, inst))

    torch.save({
        "epoch": epoch,
        "count": i,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_mavg,
        "epoch_losses": epoch_losses,
        "fads": fads,
    }, f"results/mel/mel_{inst}_ckpt.pth")


    x = np.arange(0, epoch + 2, n_test_epochs)
    plt.figure(figsize=(7,4))
    plt.plot(x, fads, marker='o')
    plt.title("FAD through training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/mel/mel_{inst}_fad.png", dpi=200)
    plt.show(block=False)


    plt.figure(figsize=(7,4))
    plt.plot(epoch_losses)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/mel/mel_{inst}_loss.png", dpi=200)
    plt.show(block=False)



if __name__ == '__main__':
    kwargs = {
        'seed': 9001,
        'batch_size': 100,
        'epochs': 2000,
        'lr': 5e-4,
        'n_test_examples': 10,
        'n_test_epochs': 50,
    }

    train(inst='piano', **kwargs)
    train(inst='vibes', **kwargs)
    train(inst='strings', **kwargs)
    train(inst='clar', **kwargs)