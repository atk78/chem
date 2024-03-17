import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate(model, z=None, sample_size=None, deterministic=False):
    if z is None:
        z = torch.randn(sample_size, model.latent_dim).to(device)
    else:
        z = z.to(device)
    with torch.no_grad():
        model.eval()
        _, out_seq = model.decode(z, deterministic=deterministic)
        out = [model.vocab.seq2smiles(each_seq) for each_seq in out_seq]
        return out
