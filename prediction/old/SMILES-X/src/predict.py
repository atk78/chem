import os
import pickle

import yaml
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

from . import token, augm
from .model import SmilesX
from .utils import mean_std_result


RDLogger.DisableLog("rdApp.*")


def main(
    data_name,
    augmentation=True,
    smiles_list=["CC"],
    outdir="./reports/"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(smiles_list) is str:
        smiles_list = list(smiles_list)
    poly_flag = False
    data_path = os.path.join(outdir, data_name)
    model_path = os.path.join(data_path, "model")
    scaler_path = os.path.join(model_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None
    inference_dir = os.path.join(data_path, "inferance")
    os.makedirs(inference_dir, exist_ok=True)
    try:
        with open(os.path.join(model_path, "all_params.yaml"), mode="r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No {data_name} folder.")
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
    smiles_checked = np.array(smiles_checked)
    if len(smiles_rejected) > 0:
        print("The SMILES below are incerrect and could not be verified via RDKit.")
        for i_smiles in smiles_rejected:
            print(i_smiles)

    if len(smiles_checked) == 0:
        print("***Process of inference automatically aborted!***")
        raise ValueError("The provided SMILES are all incorrect and could not be verified via RDKit.")
    y = np.array([[np.nan]*len(smiles_checked)]).flatten()
    for vocab in config["token"]["vocabulary"]:
        if "*" in vocab:
            poly_flag = True
    smiles, enum_card, y = augm.augment_data(smiles_checked, y, augmentation)
    enum_card = np.array(enum_card)
    tokenized_smiles = token.get_tokens(smiles, poly_flag=poly_flag)
    int_tokens, y = token.convert_to_int_tensor(
        tokenized_smiles,
        y,
        config["token"]["max_length"],
        config["token"]["vocabulary"]
    )

    model = SmilesX(**config["hyper_parameters"]["model"])
    model.load_state_dict(
        torch.load(
            os.path.join(model_path, "best_weights.pth"),
            map_location=torch.device("cpu"),
        )
    )
    model.eval()
    model.to(device)
    with torch.no_grad():
        y_pred = model(int_tokens.to(device)).cpu().detach().numpy()
    y_pred_mean, y_pred_std = mean_std_result(enum_card, y_pred)
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_pred_mean, y_pred_std = mean_std_result(enum_card, y_pred)
    df = pd.DataFrame(data=[smiles_checked, y_pred_mean, y_pred_std]).T
    df.columns = ["SMILES", "Predicted Value(mean)", "Predicted Value(std)"]
    print("Prediction Result")
    print(df)
    df.to_csv(
        os.path.join(inference_dir, "prediction_result.csv"), index=False
    )
