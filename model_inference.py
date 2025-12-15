import torch
from dit import DiT
from flow_matching import FlowModel
from dataset import *
from util import set_seed
from mel2wav import mel2wav
from frechet_audio_distance import FrechetAudioDistance
import soundfile as sf




def eval(model, test_loader, dataset, inst):
    with torch.no_grad():
        for _, (cond, target) in enumerate(test_loader):
            cond, target = cond.to(model.device), target.to(model.device)
            pred = model.sample(cond, steps=25, cfg_strength=3) * dataset.std + dataset.mu
            target = target * dataset.std + dataset.mu
            cond = cond * dataset.std + dataset.mu

            pred_wav = mel2wav(pred.transpose(-2, -1), out_fn=[f"results/mel/{inst}/pred/mel_pred_{inst}_{i}.wav" for i in range(pred.shape[0])])
            tgt_wav = mel2wav(target.transpose(-2, -1), out_fn=[f"results/mel/{inst}/target/mel_target_{inst}_{i}.wav" for i in range(target.shape[0])])
            cond_wav = mel2wav(cond.transpose(-2, -1), out_fn=[f"results/mel/{inst}/cond/mel_target_{inst}_{i}.wav" for i in range(target.shape[0])])

        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
        )

        fad_value = fad.score(f"./results/mel/{inst}/pred/", f"./results/mel/{inst}/target/")
        print(f"FAD = {fad_value}")

        return fad_value
    

def eval1D(model, test_loader, dataset, inst):
    with torch.no_grad():
        for _, (cond, target) in enumerate(test_loader):
            cond, target = cond.to(model.device), target.to(model.device)
            pred = model.sample(cond, steps=25, cfg_strength=3) * dataset.std + dataset.mu
            target = target * dataset.std + dataset.mu
            cond = cond * dataset.std + dataset.mu

            cond, target, pred = cond.detach().cpu().numpy(), target.detach().cpu().numpy(), pred.detach().cpu().numpy()

            for i in range(cond.shape[0]):
                os.makedirs(os.path.dirname(f"results/ts/{inst}/pred/ts_pred_{inst}_{i}.wav"), exist_ok=True)
                os.makedirs(os.path.dirname(f"results/ts/{inst}/target/ts_target_{inst}_{i}.wav"), exist_ok=True)
                os.makedirs(os.path.dirname(f"results/ts/{inst}/cond/ts_target_{inst}_{i}.wav"), exist_ok=True)
                sf.write(f"results/ts/{inst}/pred/ts_pred_{inst}_{i}.wav", pred[i, :], 16000, subtype="PCM_16")
                sf.write(f"results/ts/{inst}/target/ts_target_{inst}_{i}.wav", target[i, :], 16000, subtype="PCM_16")
                sf.write(f"results/ts/{inst}/cond/ts_target_{inst}_{i}.wav", cond[i, :], 16000, subtype="PCM_16")


        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
        )

        fad_value = fad.score(f"./results/ts/{inst}/pred/", f"./results/ts/{inst}/target/")
        print(f"FAD = {fad_value}")

        return fad_value


if __name__ == '__main__':
    set_seed(9001)

    n_frames = 512
    n_test_examples = 10

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
    model = FlowModel(dit, cfg_drop_prob=0.15)



    ckpt_path = "mel_strings_ckpt.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']}, step {ckpt['count']})")


    model = model.to(device)
    model.eval()



    dataset = StarnetImageDataset(n_frames=n_frames)
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - n_test_examples, n_test_examples])

    test_loader = DataLoader(
        test_set, 
        batch_size=n_test_examples, 
        # collate_fn=collate_trim
    )

    eval(model, test_loader, dataset)