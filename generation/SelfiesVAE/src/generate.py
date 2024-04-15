import os
import pickle
import yaml

import torch

from .models.SmilesVAE import SmilesVAE


def generate_smiles(
    n_generate=1000,
    training_dir="",
    device="cpu"
):

    with open(
        os.path.join(training_dir, "smiles_vocab.pkl"), mode="rb"
    ) as pickle_file:
        smiles_vocab = pickle.load(pickle_file)
    with open(
        os.path.join(training_dir, "params.yaml"), mode="r"
    ) as params_file:
        parameters = yaml.safe_load(params_file)
    model = SmilesVAE(
        vocab=smiles_vocab,
        device=device,
        **parameters
    )
    model.load_state_dict(
        torch.load(
            os.path.join(training_dir, "best_weights.pth"),
            map_location=device
        )
    )
    model = model.to(device)

    return model.generate(sample_size=n_generate)
