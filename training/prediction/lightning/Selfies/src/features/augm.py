from rdkit import Chem
import pandas as pd
import numpy as np
import selfies as sf


def rotate_atoms(li, x):
    return li[x % len(li):] + li[: x % len(li)]


def generate_selfies(smiles, augmentation=True, kekule=False, poly_flag=False):
    selfies_list = list()
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
                        )
                        if poly_flag:
                            smiles = smiles.replace("*", "[Xe]")
                        selfies = sf.encoder(smiles)
                    except:
                        selfies = "None"
                    selfies_list.append(selfies)
            else:
                selfies = "None"
                selfies_list.append(selfies)
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


def augment_data(smiles_array, prop_array, augmentation=True, poly_flag=False):
    selfies_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for idx, i_smiles in enumerate(smiles_array.tolist()):
        enumerated_selfies = generate_selfies(
            i_smiles, augmentation, poly_flag=poly_flag
        )
        if "None" not in enumerated_selfies:
            smiles_enum_card.append(len(enumerated_selfies))
            selfies_enum.extend(enumerated_selfies)
            prop_enum.extend([prop_array[idx]] * len(enumerated_selfies))

    return np.array(selfies_enum), smiles_enum_card, np.array(prop_enum)
