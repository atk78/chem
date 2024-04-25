import random

from rdkit import Chem
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift


def rotate_atoms(li, x):
    return li[x % len(li):] + li[: x % len(li)]


def generate_smiles(smiles, augmentation=True, kekule=False, max_n_augm=0):
    if max_n_augm == 0:
        return [smiles]
    smiles_list = list()
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol is not None:
        n_atoms = mol.GetNumAtoms()
        n_atoms_list = [nat for nat in range(n_atoms)]
        if augmentation:
            canonical = False
            if n_atoms != 0:
                for i_atoms in range(n_atoms):
                    n_atoms_list_tmp = rotate_atoms(n_atoms_list, i_atoms)  # rotate atoms' index
                    nmol = Chem.RenumberAtoms(mol, n_atoms_list_tmp)  # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(
                            nmol,
                            isomericSmiles=True,  # keep isomerism
                            kekuleSmiles=kekule,  # kekulize or not
                            rootedAtAtom=-1,  # default
                            canonical=canonical,  # canonicalize or not
                            allBondsExplicit=False,  #
                            allHsExplicit=False,
                        )  #
                    except:
                        smiles = "None"
                    smiles_list.append(smiles)
            else:
                smiles = "None"
                smiles_list.append(smiles)
        else:
            canonical = True
            try:
                smiles = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=True,
                    kekuleSmiles=kekule,
                    rootedAtAtom=-1,
                    canonical=canonical,
                    allBondsExplicit=False,
                    allHsExplicit=False,
                )
            except:
                smiles = "None"
            smiles_list.append(smiles)
    else:
        smiles = "None"
        smiles_list.append(smiles)

    smiles_list = (
        pd.DataFrame(smiles_list).drop_duplicates().iloc[:, 0].values.tolist()
    )  # duplicates are discarded
    if max_n_augm > 0:
        if len(smiles_list) > max_n_augm:
            smiles_list = random.sample(smiles_list, max_n_augm)

    return smiles_list


def augment_data(smiles_array, prop_array, augmentation=True, max_n_augm=0):
    smiles_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for idx, i_smiles in enumerate(smiles_array):
        enumerated_smiles = generate_smiles(i_smiles, augmentation, max_n_augm=max_n_augm)
        if "None" not in enumerated_smiles:
            smiles_enum_card.append(len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            prop_enum.extend([prop_array[idx]] * len(enumerated_smiles))
    return smiles_enum, smiles_enum_card, prop_enum


def mean_std_result(x_cardinal_tmp: list[int], y_pred_tmp: list[float]):
    x_cardinal_tmp = np.array(x_cardinal_tmp)
    x_card_cumsum = np.cumsum(x_cardinal_tmp)
    x_card_cumsum_shift = shift(x_card_cumsum, 1, cval=0)
    y_mean = np.array(
        [
            np.mean(y_pred_tmp[x_card_cumsum_shift[cenumcard]: ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum)
        ]
    )
    y_std = np.array(
        [
            np.std(y_pred_tmp[x_card_cumsum_shift[cenumcard]: ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum)
        ]
    )
    return y_mean, y_std
