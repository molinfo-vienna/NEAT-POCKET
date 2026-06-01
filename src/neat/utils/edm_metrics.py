"""Taken without modifications from
https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/configs/datasets_config.py
https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/qm9/bond_analyze.py#L4
https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/qm9/analyze.py#L4
"""

import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

qm9_with_h = {
    "name": "qm9",
    "mapping": {1: 0, 6: 1, 7: 2, 8: 3, 9: 4},
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
    "atom_decoder": ["H", "C", "N", "O", "F"],
    "n_nodes": {
        22: 3393,
        17: 13025,
        23: 4848,
        21: 9970,
        19: 13832,
        20: 9482,
        16: 10644,
        13: 3060,
        15: 7796,
        25: 1506,
        18: 13364,
        12: 1689,
        11: 807,
        24: 539,
        14: 5136,
        26: 48,
        7: 16,
        10: 362,
        8: 49,
        9: 124,
        27: 266,
        4: 4,
        29: 25,
        6: 9,
        5: 5,
        3: 1,
    },
    "max_n_nodes": 29,
    "atom_types": {1: 635559, 2: 101476, 0: 923537, 3: 140202, 4: 2323},
    "distances": [
        903054,
        307308,
        111994,
        57474,
        40384,
        29170,
        47152,
        414344,
        2202212,
        573726,
        1490786,
        2970978,
        756818,
        969276,
        489242,
        1265402,
        4587994,
        3187130,
        2454868,
        2647422,
        2098884,
        2001974,
        1625206,
        1754172,
        1620830,
        1710042,
        2133746,
        1852492,
        1415318,
        1421064,
        1223156,
        1322256,
        1380656,
        1239244,
        1084358,
        981076,
        896904,
        762008,
        659298,
        604676,
        523580,
        437464,
        413974,
        352372,
        291886,
        271948,
        231328,
        188484,
        160026,
        136322,
        117850,
        103546,
        87192,
        76562,
        61840,
        49666,
        43100,
        33876,
        26686,
        22402,
        18358,
        15518,
        13600,
        12128,
        9480,
        7458,
        5088,
        4726,
        3696,
        3362,
        3396,
        2484,
        1988,
        1490,
        984,
        734,
        600,
        456,
        482,
        378,
        362,
        168,
        124,
        94,
        88,
        52,
        44,
        40,
        18,
        16,
        8,
        6,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "colors_dic": ["#FFFFFF99", "C7", "C0", "C3", "C1"],
    "radius_dic": [0.46, 0.77, 0.77, 0.77, 0.77],
    "with_h": True,
}

geom_with_h = {
    "name": "geom",
    "mapping": {
        1: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
        13: 6,
        14: 7,
        15: 8,
        16: 9,
        17: 10,
        33: 11,
        35: 12,
        53: 13,
        80: 14,
        83: 15,
    },
    "atom_encoder": {
        "H": 0,
        "B": 1,
        "C": 2,
        "N": 3,
        "O": 4,
        "F": 5,
        "Al": 6,
        "Si": 7,
        "P": 8,
        "S": 9,
        "Cl": 10,
        "As": 11,
        "Br": 12,
        "I": 13,
        "Hg": 14,
        "Bi": 15,
    },
    "atomic_nb": [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    "atom_decoder": [
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "As",
        "Br",
        "I",
        "Hg",
        "Bi",
    ],
    "max_n_nodes": 181,
    "n_nodes": {
        3: 1,
        4: 3,
        5: 9,
        6: 2,
        7: 8,
        8: 23,
        9: 23,
        10: 50,
        11: 109,
        12: 168,
        13: 280,
        14: 402,
        15: 583,
        16: 597,
        17: 949,
        18: 1284,
        19: 1862,
        20: 2674,
        21: 3599,
        22: 6109,
        23: 8693,
        24: 13604,
        25: 17419,
        26: 25672,
        27: 31647,
        28: 43809,
        29: 56697,
        30: 70400,
        31: 82655,
        32: 104100,
        33: 122776,
        34: 140834,
        35: 164888,
        36: 185451,
        37: 194541,
        38: 218549,
        39: 231232,
        40: 243300,
        41: 253349,
        42: 268341,
        43: 272081,
        44: 276917,
        45: 276839,
        46: 274747,
        47: 272126,
        48: 262709,
        49: 250157,
        50: 244781,
        51: 228898,
        52: 215338,
        53: 203728,
        54: 191697,
        55: 180518,
        56: 163843,
        57: 152055,
        58: 136536,
        59: 120393,
        60: 107292,
        61: 94635,
        62: 83179,
        63: 68384,
        64: 61517,
        65: 48867,
        66: 37685,
        67: 32859,
        68: 27367,
        69: 20981,
        70: 18699,
        71: 14791,
        72: 11921,
        73: 9933,
        74: 9037,
        75: 6538,
        76: 6374,
        77: 4036,
        78: 4189,
        79: 3842,
        80: 3277,
        81: 2925,
        82: 1843,
        83: 2060,
        84: 1394,
        85: 1514,
        86: 1357,
        87: 1346,
        88: 999,
        89: 300,
        90: 390,
        91: 510,
        92: 510,
        93: 240,
        94: 721,
        95: 360,
        96: 360,
        97: 390,
        98: 330,
        99: 540,
        100: 258,
        101: 210,
        102: 60,
        103: 180,
        104: 206,
        105: 60,
        106: 390,
        107: 180,
        108: 180,
        109: 150,
        110: 120,
        111: 360,
        112: 120,
        113: 210,
        114: 60,
        115: 30,
        116: 210,
        117: 270,
        118: 450,
        119: 240,
        120: 228,
        121: 120,
        122: 30,
        123: 420,
        124: 240,
        125: 210,
        126: 158,
        127: 180,
        128: 60,
        129: 30,
        130: 120,
        131: 30,
        132: 120,
        133: 60,
        134: 240,
        135: 169,
        136: 240,
        137: 30,
        138: 270,
        139: 180,
        140: 270,
        141: 150,
        142: 60,
        143: 60,
        144: 240,
        145: 180,
        146: 150,
        147: 150,
        148: 90,
        149: 90,
        151: 30,
        152: 60,
        155: 90,
        159: 30,
        160: 60,
        165: 30,
        171: 30,
        175: 30,
        176: 60,
        181: 30,
    },
    "atom_types": {
        0: 143905848,
        1: 290,
        2: 129988623,
        3: 20266722,
        4: 21669359,
        5: 1481844,
        6: 1,
        7: 250,
        8: 36290,
        9: 3999872,
        10: 1224394,
        11: 4,
        12: 298702,
        13: 5377,
        14: 13,
        15: 34,
    },
    "colors_dic": [
        "#FFFFFF99",
        "C2",
        "C7",
        "C0",
        "C3",
        "C1",
        "C5",
        "C6",
        "C4",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
    ],
    "radius_dic": [
        0.3,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
    ],
    "with_h": True,
}

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}

bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}


def print_table(bonds_dict):
    letters = ["H", "C", "O", "N", "P", "S", "F", "Si", "Cl", "Br", "I"]

    new_letters = []
    for key in letters + list(bonds_dict.keys()):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f"{x} & ", end="")
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f"{y} & ", end="")
            if x in bonds_dict[y]:
                print(f"{bonds_dict[y][x]} & ", end="")
            else:
                print("- & ", end="")
        print()


def check_consistency_bond_dictionaries():
    for bonds_dict in [bonds1, bonds2, bonds3]:
        for atom1 in bonds1:
            for atom2 in bonds_dict[atom1]:
                bond = bonds_dict[atom1][atom2]
                try:
                    bond_check = bonds_dict[atom2][atom1]
                except KeyError:
                    raise ValueError("Not in dict " + str((atom1, atom2)))

                assert (
                    bond == bond_check
                ), f"{bond} != {bond_check} for {atom1}, {atom2}"


margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def single_bond_only(threshold, length, margin1=5):
    if length < threshold + margin1:
        return 1
    return 0


def geom_predictor(atom_pair, bond_length, limit_bonds_to_one=False):
    """atom_pair: atom pair (couple of str)
    bond_length: bond length (float)"""
    bond_order = get_bond_order(
        atom_pair[0], atom_pair[1], bond_length, check_exists=True
    )

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


############################
# Validity and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info["atom_decoder"]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype="int")

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info["name"] == "qm9":
                order = get_bond_order(atom1, atom2, dist)
            elif dataset_info["name"] == "geom":
                order = geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist
                )
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type_i]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type_i], nr_bonds_i)
            )
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error converting molecule to SMILES: {e}")
        return None


def build_molecule(positions, atom_types, dataset_info, bond_orders=None):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info, bond_orders)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(
            bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
        )
    return mol


def build_xae_molecule(positions, atom_types, dataset_info, bond_orders=None):
    """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
    args:
    positions: N x 3  (already masked to keep final number nodes)
    atom_types: N
    returns:
    X: N         (int)
    A: N x N     (bool)                  (binary adjacency matrix)
    E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info["atom_decoder"]
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            if bond_orders is not None:
                order = bond_orders[i, j]
            else:
                pair = sorted([atom_types[i], atom_types[j]])
                if dataset_info["name"] == "qm9":
                    order = get_bond_order(
                        atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j]
                    )
                elif dataset_info["name"] == "geom":
                    order = geom_predictor(
                        (atom_decoder[pair[0]], atom_decoder[pair[1]]),
                        dists[i, j],
                        limit_bonds_to_one=True,
                    )
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


def edm_metrics(x, pos, batch, dataset):
    # atom/mol stability depends on the dataset
    # atom stability depends on the group of molecules - cannot be computed for individual molecules and mean-aggregated
    if dataset == "QM9":
        dataset_info = qm9_with_h
    elif dataset == "GEOM" or dataset == "CROSSDOCKED":
        dataset_info = geom_with_h
    else:
        raise ValueError("Dataset not recognized: " + dataset)

    count_mol_stable = 0
    count_atm_stable = 0
    count_mol_total = batch.max().item() + 1
    count_atm_total = 0

    valid = []
    invalid_idxs = []
    for i in tqdm(range(count_mol_total), desc="EDM metrics calculation"):
        positions = pos[batch == i]
        atom_type = x[batch == i] - 1
        is_stable, nr_stable, total = check_stability(
            positions, atom_type, dataset_info
        )
        count_atm_stable += nr_stable
        count_atm_total += total

        count_mol_stable += int(is_stable)

        mol = build_molecule(positions, atom_type, dataset_info)
        RDLogger.DisableLog("rdApp.*")
        smiles = mol2smiles(mol)
        RDLogger.EnableLog("rdApp.*")
        if smiles is not None:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            valid.append(smiles)
        else:
            invalid_idxs.append(i)

    num_valid = len(valid)
    atom_stability = count_atm_stable / count_atm_total
    mol_stability = count_mol_stable / count_mol_total
    validity = num_valid / count_mol_total
    uniqueness = len(set(valid)) / num_valid if num_valid > 0 else 0

    return atom_stability, mol_stability, validity, uniqueness, invalid_idxs
