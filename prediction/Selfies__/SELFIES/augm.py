from rdkit import Chem
import pandas as pd
import numpy as np
import selfies as sf


def rotate_atoms(li, x):
    return li[x % len(li) :] + li[: x % len(li)]


def generate_selfies(smiles, kekule=False, canon=True, rotate=False):
    selfies_list = list()
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol != None:
        n_atoms = mol.GetNumAtoms()
        n_atoms_list = [nat for nat in range(n_atoms)]
        if rotate == True:
            canon = False
            if n_atoms != 0:
                for iatoms in range(n_atoms):
                    n_atoms_list_tmp = rotate_atoms(
                        n_atoms_list, iatoms
                    )  # rotate atoms' index
                    nmol = Chem.RenumberAtoms(
                        mol, n_atoms_list_tmp
                    )  # renumber atoms in mol
                    try:
                        smiles = Chem.MolToSmiles(
                            nmol,
                            isomericSmiles=True,  # keep isomerism
                            kekuleSmiles=kekule,  # kekulize or not
                            rootedAtAtom=-1,  # default
                            canonical=canon,  # canonicalize or not
                            allBondsExplicit=False,  #
                            allHsExplicit=False,
                        )  #
                        selfies = sf.encoder(smiles)
                    except:
                        selfies = "None"
                    selfies_list.append(selfies)
            else:
                selfies = "None"
                selfies_list.append(selfies)
        else:
            try:
                smiles = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=True,
                    kekuleSmiles=kekule,
                    rootedAtAtom=-1,
                    canonical=canon,
                    allBondsExplicit=False,
                    allHsExplicit=False,
                )
                selfies = sf.encoder(smiles)
            except:
                selfies = "None"
            selfies_list.append(selfies)
    else:
        selfies = "None"
        selfies_list.append(selfies)

    selfies_list = (
        pd.DataFrame(selfies_list).drop_duplicates().iloc[:, 0].values.tolist()
    )  # duplicates are discarded

    return selfies_list


def Augmentation(smiles_array, prop_array, canon=True, rotate=False):
    selfies_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for idx, ismiles in enumerate(smiles_array.tolist()):
        enumerated_smiles = generate_selfies(ismiles, canon=canon, rotate=rotate)
        if "None" not in enumerated_smiles:
            smiles_enum_card.append(len(enumerated_smiles))
            selfies_enum.extend(enumerated_smiles)
            prop_enum.extend([prop_array[idx]] * len(enumerated_smiles))

    return np.array(selfies_enum), smiles_enum_card, np.array(prop_enum)
