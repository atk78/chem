import os
import pickle
import yaml

import torch
from rdkit import Chem, RDLogger

from .model import SmilesVAE


RDLogger.DisableLog("rdApp.*")


def generate_smiles(
    n_generation=1000,
    training_dir="",
    device="cpu"
):
    model = build_model(training_dir, device)
    n_success = 0
    generated_smiles = list()
    while n_success < n_generation:
        tmp_smiles = model.generate(sample_size=n_generation)
        for smiles in tmp_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                rings = mol.GetRingInfo().AtomRings()
                if len(rings) > 0:
                    for ring in rings:
                        if len(ring) > 6:
                            raise ValueError
                generated_smiles.append(Chem.MolToSmiles(mol))
            except:
                pass
        n_success += len(generated_smiles)
    generated_smiles = generated_smiles[:n_generation]
    return generated_smiles


def build_model(training_dir, device):
    with open(
        os.path.join(training_dir, "params.yaml"), mode="r"
    ) as params_file:
        parameters = yaml.safe_load(params_file)
    model = SmilesVAE(
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
    return model
