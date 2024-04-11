import random

from rdkit import Chem
import pandas as pd
import numpy as np


def rotate_atoms(li, x):
    return li[x % len(li):] + li[: x % len(li)]


def generate_smiles(smiles, augmentation=True, kekule=False, max_n_augm=10):
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


def augment_data(smiles_array, prop_array, augmentation=True, max_n_augm=10):
    smiles_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for idx, i_smiles in enumerate(smiles_array.tolist()):
        enumerated_smiles = generate_smiles(i_smiles, augmentation, max_n_augm=max_n_augm)
        if "None" not in enumerated_smiles:
            smiles_enum_card.append(len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            prop_enum.extend([prop_array[idx]] * len(enumerated_smiles))

    return np.array(smiles_enum), smiles_enum_card, np.array(prop_enum)