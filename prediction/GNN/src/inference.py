import os
import pickle
import shutil
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

from .token import InferacneToken
from .models.LSTMAttention import LightningModel
from .utils import mean_median_result


RDLogger.DisableLog("rdApp.*")


def main(
    data_name,
    smiles_list=["CC", "CCC", "C=O"],
    augmentation=False,
    outputs_dir="../outputs",
):
    input_dir = os.path.join(outputs_dir, data_name)
    save_dir = os.path.join(input_dir, "inferance")
    os.makedirs(save_dir, exist_ok=True)

    print("***SMILES_X for inference starts...***\n\n")
    print("***Checking the SMILES list for inference***\n")
    smiles_checked = list()
    smiles_rejected = list()

    for i_smiles in smiles_list:
        try:
            mol_tmp = Chem.MolFromSmiles(i_smiles)
            canonical_smiles = Chem.MolToSmiles(mol_tmp)
            smiles_checked.append(canonical_smiles)
        except:
            smiles_rejected.append(i_smiles)

    if len(smiles_rejected) > 0:
        print("The SMILES below are incerrect and could not be verified via RDKit.")
        for i_smiles in smiles_rejected:
            print(i_smiles)

    if len(smiles_checked) == 0:
        print("***Process of inference automatically aborted!***")
        print("The provided SMILES are all incorrect and could not be verified via RDKit.")
        return

    for smiles in smiles_checked:
        if "*" in smiles:
            poly = True
        else:
            poly = False

    inference_dir = os.path.join(save_dir, "inference")

    if os.path.exists(inference_dir):
        shutil.rmtree(inference_dir)
        os.makedirs(inference_dir)
    else:
        os.makedirs(inference_dir)

    with open(
        os.path.join(input_dir, "best_hyper_params.pkl"), mode="rb"
    ) as f:
        best_hyper_params = pickle.load(f)

    token = InferacneToken(
        smiles_list=smiles_checked,
        max_length=best_hyper_params[0],
        augmentation=augmentation,
        input_dir=input_dir,
        poly=poly,
    )
    token.setup()

    model = LightningModel.load_from_checkpoint(
        checkpoint_path=os.path.join(input_dir, "best_weights.ckpt"),
        token_size=best_hyper_params[0],
        learning_rate=best_hyper_params[5],
        lstm_units=best_hyper_params[1],
        dense_units=best_hyper_params[2],
        embedding_dim=best_hyper_params[3],
        log_flag=False,
        map_location=torch.device("cpu"),
    )

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(token.enum_tokens).detach().numpy()
    card = np.array(token.enum_card)
    y_pred_mean, y_pred_std = mean_median_result(card, y_pred)
    df = pd.DataFrame(data=[smiles_checked, y_pred_mean, y_pred_std]).T
    df.columns = ["SMILES", "Predicted Value(mean)", "Predicted Value(std)"]
    print("Prediction Result")
    print(df)
    df.to_csv(os.path.join(inference_dir, "prediction_result.csv"), index=False)
