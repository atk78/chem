import itertools
import os

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit.Chem import rdmolops


main_chein_smarts_file = os.path.join(
    os.path.dirname(__file__), "main_chain_structure.csv"
)
main_chain_smarts_df = pd.read_csv(main_chein_smarts_file)
main_chain_smarts = main_chain_smarts_df["SMARTS"]


def get_graph_vectors(
    smiles_array,
    y_array,
    computable_atoms,
    poly_flag=False,
    use_chirality=False,
    use_stereochemistry=True
):
    graph_vectors = list()
    for i_smiles, y in zip(smiles_array, y_array):
        graph_vector_tmp = calc_graph_vector(
            smiles=i_smiles,
            y=y,
            computable_atoms=computable_atoms,
            poly_flag=poly_flag,
            use_chirality=use_chirality,
            use_stereochemistry=use_stereochemistry)
        graph_vectors.append(graph_vector_tmp)
    return graph_vectors


def calc_graph_vector(
    smiles,
    y,
    computable_atoms,
    poly_flag=False,
    use_chirality=False,
    use_stereochemistry=True
):
    # 初期設定
    computable_atoms = computable_atoms
    poly_bond_indexes = []
    unrelated_smiles = "CC"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    # SMILESからrdkitのmolオブジェクトに変換してグラフ化処理に利用する
    mol = Chem.MolFromSmiles(smiles)
    calc_atom_indexes = [i for i in range(mol.GetNumAtoms())]
    # 対象分子がポリマー（[*]が存在するとき）の動作
    # [*]を[*]の隣の原子とみなして置換する
    if poly_flag:
        poly_bond_indexes = search_poly_bond(mol)  # 結合atom[*]のindexを探索
        poly_bond_nums = len(poly_bond_indexes)  # 結合atom[*]の数を取得
        mol = set_chain_type(
            mol, poly_bond_indexes
        )  # 構造中のatomが主鎖かどうか判別する
        mol = replace_bond_to_atoms(
            mol, poly_bond_indexes
        )  # 結合atomのとなりの原子を付加する
        calc_atom_indexes = calc_atom_indexes[
            : -1 * poly_bond_nums
        ]  # 計算するatomのindex
        unrelated_mol.GetAtomWithIdx(0).SetProp("chain_type", "main_chain")
        # AllChem.ComputeGasteigerCharges(mol)
    else:
        poly_bond_nums = 0
        # for atom in mol.GetAtoms():
        #     atom.SetProp("chain_type", "None")
    # sssr = Chem.GetSymmSSSR(mol)
    # 電荷計算
    ComputeGasteigerCharges(mol, nIter=100, throwOnParamFailure=0)
    ComputeGasteigerCharges(unrelated_mol)
    # 特徴量の数を取得
    n_nodes = len(calc_atom_indexes)
    n_edges = 2 * (mol.GetNumBonds()) - 2 * poly_bond_nums

    # 数を確定させるための前処理
    n_node_features = len(
        get_atom_features(
            unrelated_mol.GetAtomWithIdx(0),
            computable_atoms,
            use_chirality,
            poly_flag
        )
    )
    n_edge_features = len(
        get_bond_features(
            unrelated_mol.GetBondBetweenAtoms(0, 1),
            use_stereochemistry,
            # ff_features=[0, 0, 0, 0],
        )
    )
    # ノードの特徴(X)
    X = np.zeros((n_nodes, n_node_features))
    for idx in calc_atom_indexes:
        atom = mol.GetAtomWithIdx(idx)
        X[idx, :] = get_atom_features(
            atom,
            computable_atoms,
            use_chirality,
            poly_flag
        )
    X = torch.tensor(X, dtype=torch.float32)

    # エッジのindex(E)
    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    new_rows, new_cols = [], []
    for i, j in zip(rows, cols):
        if i in calc_atom_indexes and j in calc_atom_indexes:
            new_rows.append(i)
            new_cols.append(j)
    new_rows, new_cols = np.array(new_rows), np.array(new_cols)

    torch_rows = torch.from_numpy(new_rows).to(torch.int32)
    torch_cols = torch.from_numpy(new_cols).to(torch.int32)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    # エッジの特徴(EF)
    EF = np.zeros((n_edges, n_edge_features))

    for k, (idx1, idx2) in enumerate(zip(new_rows, new_cols)):
        # bond_ff_features = calc_bond_ff_features(mol, idx1, idx2)
        EF[k] = get_bond_features(
            bond=mol.GetBondBetweenAtoms(int(idx1), int(idx2)),
            use_stereochemistry=use_stereochemistry,
            # ff_features=bond_ff_features,
        )
    EF = torch.tensor(EF, dtype=torch.float32)

    Y = torch.tensor(np.array([y]), dtype=torch.float32)
    graph_vector = Data(x=X, edge_index=E, edge_attr=EF, y=Y)
    return graph_vector


def one_hot_encoding(x, encode_list):
    if x not in encode_list:
        x = encode_list[-1]
    # encode_listの各要素に対してatomが適応しているかどうか(True:1, False:0)を判別し
    # それらをエンコードとして返す
    binary_encoding = [
        int(boolian_value)
        for boolian_value in list(map(lambda s: x == s, encode_list))
    ]
    return binary_encoding


def search_poly_bond(mol):
    poly_bond_mol = Chem.MolFromSmiles("*")  # 「*」のmolファイルを生成
    # 結合atomのindexを取得 -> 例: ((1,), (2,))
    poly_bond_indexes = mol.GetSubstructMatches(poly_bond_mol)
    # 結合atomのindexを平坦化する ((1,), (2,)) -> [1, 2]
    poly_bond_indexes = list(itertools.chain.from_iterable(poly_bond_indexes))
    return poly_bond_indexes


def replace_bond_to_atoms(mol, poly_bond_indexes):
    # 結合を隣の原子に置換
    adgacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    next_atoms = []
    for idx in poly_bond_indexes:
        # ポリマー結合[*]の隣の原子を取得
        next_atom_idx = int(np.where(adgacency_matrix[idx] == 1)[0][0])
        next_atom = mol.GetAtomWithIdx(next_atom_idx).GetSymbol()
        if next_atom == "Si":
            next_atom = "[SiH3]"
        next_atoms.append(next_atom)
    next_atoms.reverse()
    # Replace Substructs
    for next_atom in next_atoms:
        replace_from = Chem.MolFromSmiles("*")
        replace_to = Chem.MolFromSmiles(next_atom)
        mol = AllChem.ReplaceSubstructs(mol, replace_from, replace_to)[0]
    mol = Chem.RemoveHs(mol)  # 余計なHを削除
    return mol


def replace_poly_bond_to_h(mol, poly_bond_indexes):
    for _ in poly_bond_indexes:
        replace_from = Chem.MolFromSmiles("*")
        replace_to = Chem.MolFromSmiles("[H]")
        mol = AllChem.ReplaceSubstructs(mol, replace_from, replace_to)[0]
    mol = Chem.RemoveHs(mol)  # たまにHが余計についたりするため
    return mol


def set_chain_type(mol, poly_bond_indexes):
    # 重合の結合[*]ともう一方の重合の結合[*]の最短パスを取得する
    poly_bond_shortest_path = itertools.permutations(poly_bond_indexes, 2)
    poly_bond_path_indexes = []
    for poly_bond_indexes_i in poly_bond_shortest_path:
        poly_bond_path_indexes += rdmolops.GetShortestPath(
            mol, poly_bond_indexes_i[0], poly_bond_indexes_i[1]
        )
    # 最短パスから主鎖構造のindexを取得する
    main_indices = []
    for smarts in main_chain_smarts:
        search_mol = Chem.MolFromSmarts(smarts)
        get_index = mol.GetSubstructMatches(search_mol)
        for index in get_index:
            symmeritc_diff = set(poly_bond_indexes) & set(index)
            if len(symmeritc_diff) > 0:
                main_indices += list(index)
    main_indices = list(set(main_indices))
    for atom in mol.GetAtoms():
        if atom.GetIdx() in main_indices:
            atom.SetProp("chain_type", "main_chain")
        else:
            atom.SetProp("chain_type", "sub_chain")
    return mol


def calc_bond_ff_features(mol, idx1, idx2):
    # rdkitに搭載しているUFF ForceFieldのパラメータを取得
    bond_kb, bond_r0 = rdForceFieldHelpers.GetUFFBondStretchParams(
        mol, int(idx1), int(idx2)
    )
    bond_vdw_x, bond_vdw_d = rdForceFieldHelpers.GetUFFVdWParams(
        mol, int(idx1), int(idx2)
    )
    bond_kb = (bond_kb - 940.610377) / 204.924599
    bond_r0 = (bond_r0 - 1.414844) / 0.089204
    bond_vdw_x = (bond_vdw_x - 3.811359) / 0.079849
    bond_vdw_d = (bond_vdw_d - 0.100566) / 0.017500
    bond_ff_features = [bond_kb, bond_r0, bond_vdw_x, bond_vdw_d]
    return bond_ff_features


def get_atom_features(
    atom,
    computable_atoms,
    use_chirality=False,
    poly_flag=False
):
    # if hydrogens_implicit == False:
    #     computable_atoms = ["H"] + computable_atoms

    # 原子タイプのエンコーディング C -> [0, 0, 1, 0, 0 ...]のように変更
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), computable_atoms)
    # 水素の数
    n_hydrogens_enc = one_hot_encoding(
        int(atom.GetTotalNumHs()),
        [0, 1, 2, 3]
    )
    # 原子に隣接する原子の数
    n_heavy_neighbors_enc = one_hot_encoding(
        int(atom.GetDegree()),
        [1, 2, 3, 4]
    )
    # 電荷
    formal_carge_enc = one_hot_encoding(
        int(atom.GetFormalCharge()),
        [-1, 0, 1]
    )
    # 混成軌道の種類
    hybridization_type_enc = one_hot_encoding(
        atom.GetHybridization(),
        [
            rdkit.Chem.rdchem.HybridizationType.S,
            rdkit.Chem.rdchem.HybridizationType.SP,
            rdkit.Chem.rdchem.HybridizationType.SP2,
            rdkit.Chem.rdchem.HybridizationType.SP3,
            rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        ],
    )
    # 原子の価数
    n_valence_enc = one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3])
    # 環状の原子かどうか
    is_in_a_ring_enc = [int(atom.IsInRing())]

    # 環の大きさ
    if atom.IsInRing():
        ring_size = 7
        for i in range(3, 7):
            if atom.IsInRingSize(i):
                ring_size = i
        if ring_size == 3:
            ring_size_enc = [0, 1, 0, 0, 0, 0]
        elif ring_size == 4:
            ring_size_enc = [0, 0, 1, 0, 0, 0]
        elif ring_size == 5:
            ring_size_enc = [0, 0, 0, 1, 0, 0]
        elif ring_size == 6:
            ring_size_enc = [0, 0, 0, 0, 1, 0]
        else:
            ring_size_enc = [0, 0, 0, 0, 0, 1]
    else:
        ring_size_enc = [1, 0, 0, 0, 0, 0]

    # 芳香族かどうか
    # is_aromatic_enc = [int(atom.GetIsAromatic())]
    if atom.GetIsAromatic():
        neighbor_aromatic_num = [
            na.GetIsAromatic() for na in atom.GetNeighbors()
        ].count(True)
        # 通常の芳香族
        if neighbor_aromatic_num == 2:
            is_aromatic_enc = [0, 1, 0]
        # ナフタレンのような芳香族の場合
        else:
            is_aromatic_enc = [0, 0, 1]
    # 芳香族ではない
    else:
        is_aromatic_enc = [1, 0, 0]
    # 原子の重さ
    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
    # 原子のvdw体積
    vdw_radius_scaled = [
        float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
    ]
    # 共有結合半径
    covalent_radius_scaled = [
        float(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()))
    ]
    # 原子のvdw体積
    # vdw_radius_scaled = [
    #     float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
    # ]
    # 共有結合半径
    # covalent_radius_scaled = [
    #     float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)
    # ]
    # # 原子の電荷
    # atom_charge_scaled = [
    #     (float(atom.GetProp("_GasteigerCharge")) - (-0.043309)) / 0.169341
    # ]

    if poly_flag:
        if atom.GetProp("chain_type") == "main_chain":
            main_chain_atom_enc = [0]
        else:
            main_chain_atom_enc = [1]
    else:
        main_chain_atom_enc = []

    # 原子の特徴量ベクトル
    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + n_hydrogens_enc
        + formal_carge_enc
        + hybridization_type_enc
        + n_valence_enc
        + is_in_a_ring_enc
        + ring_size_enc
        + is_aromatic_enc
        + atomic_mass_scaled
        + vdw_radius_scaled
        + covalent_radius_scaled
        # + atom_charge_scaled
        + main_chain_atom_enc
    )
    # chiralityの選択
    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            atom.GetChiralTag(),
            [
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ],
        )
        atom_feature_vector + chirality_type_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True, ff_features=[]):
    bond_type_enc = one_hot_encoding(
        bond.GetBondType(),
        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    )
    # 共役結合かどうか
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    # 結合が環状かどうか
    bond_is_in_ring_enc = [int(bond.IsInRing())]

    # 結合の特徴量ベクトル
    bond_features_vector = (
        bond_type_enc
        + bond_is_conj_enc
        + bond_is_in_ring_enc
        # + ff_features
    )

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(
            bond.GetStereo(),
            [
                rdkit.Chem.rdchem.BondStereo.STEREONONE,
                rdkit.Chem.rdchem.BondStereo.STEREOZ,
                rdkit.Chem.rdchem.BondStereo.STEREOE,
            ],
        )
        bond_features_vector += stereo_type_enc

    return np.array(bond_features_vector)


def calc_vector(
    mol,
    y,
    n_nodes,
    n_node_features,
    n_edges,
    n_edge_features,
    calc_atom_indexes,
    computable_atoms,
    use_chirality,
    poly_flag=False,
):
    # ノードの特徴(X)
    X = np.zeros((n_nodes, n_node_features))
    for idx in calc_atom_indexes:
        atom = mol.GetAtomWithIdx(idx)
        X[idx, :] = get_atom_features(
            atom,
            computable_atoms,
            use_chirality,
            poly_flag=poly_flag)
    X = torch.tensor(X, dtype=torch.float32)

    # エッジのindex(E)
    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    new_rows, new_cols = [], []
    for i, j in zip(rows, cols):
        if i in calc_atom_indexes and j in calc_atom_indexes:
            new_rows.append(i)
            new_cols.append(j)
    new_rows, new_cols = np.array(new_rows), np.array(new_cols)

    torch_rows = torch.from_numpy(new_rows).to(torch.int32)
    torch_cols = torch.from_numpy(new_cols).to(torch.int32)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    # エッジの特徴(EF)
    EF = np.zeros((n_edges, n_edge_features))

    for k, (idx1, idx2) in enumerate(zip(new_rows, new_cols)):
        # bond_ff_features = calc_bond_ff_features(mol, idx1, idx2)
        EF[k] = get_bond_features(
            bond=mol.GetBondBetweenAtoms(int(idx1), int(idx2)),
            use_stereochemistry=True,
            # ff_features=bond_ff_features,
        )
    EF = torch.tensor(EF, dtype=torch.float32)

    Y = torch.tensor(np.array([y]), dtype=torch.float32)
    graph_vector = Data(x=X, edge_index=E, edge_attr=EF, y=Y)
    return graph_vector
