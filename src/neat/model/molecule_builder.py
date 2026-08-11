import glob
import logging
import os
from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds, rdmolfiles
from tqdm import tqdm

from neat.model import BondPredictor

import glob
import os
from typing import Optional

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
from tqdm import tqdm



from rdkit import Chem
from rdkit.Chem import Mol

# Bond type mapping: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
RDKIT_BOND_TYPES = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Standard maximum valence lookup table by atomic number
MAX_VALENCE_TABLE = {
    1: 1,   # H
    6: 4,   # C
    7: 4,   # N (allows ammonium / nitro groups)
    8: 3,   # O 
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}

MIN_VALENCE_TABLE = {
    1: 1,   # H
    6: 4,   # C (forces tetravalent carbon)
    7: 3,   # N
    8: 1,   # O
    9: 1,   # F
    15: 3,  # P
    16: 2,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}

BOND_ORDERS = np.array([0.0, 1.0, 2.0, 3.0, 1.5])  # 0=none, 1=single, 2=double, 3=triple, 4=aromatic

# Allowed neutral valence states for main-group elements
NEUTRAL_VALENCE_SETS = {
    1:  {1},             # H
    5:  {3},             # B
    6:  {4},             # C
    7:  {3},             # N
    8:  {2},             # O
    9:  {1},             # F
    14: {4},             # Si
    15: {3, 5},          # P: 3 (phosphines), 5 (phosphine oxides / phosphates)
    16: {2, 4, 6},       # S: 2 (thiols/sulfides), 4 (sulfoxides), 6 (sulfones)
    17: {1},             # Cl 
    35: {1},             # Br
    53: {1},             # I 
}

# Explicit overrides for non-neutral organic species: (atomic_number, explicit_valence) -> formal_charge
CHARGED_STATE_LOOKUP = {
    # Nitrogen
    (7, 4): 1,   # Quaternary ammonium / pyridinium / nitro N+
    (7, 2): -1,  # Amide anion N-
    
    # Oxygen
    (8, 1): -1,  # Carboxylate / alkoxide O-
    (8, 3): 1,   # Oxonium / pyrylium O+
    
    # Sulfur
    (16, 1): -1, # Thiolate S-
    (16, 3): 1,  # Sulfonium S+
    (16, 5): 1,  # Sulfoxonium S+
    
    # Phosphorus
    (15, 4): 1,  # Phosphonium P+
    (15, 6): -1, # Hexafluorophosphate P-
    
    # Boron
    (5, 4): -1,  # Borohydride / Tetraphenylborate B-
}

def assign_rule_based_formal_charge(atom: Chem.Atom, explicit_valence: float) -> int:
    """Assigns formal charges accounting for hypervalent neutral states (P, S, I) 

    and standard ionic organic functional groups.
    """
    z = atom.GetAtomicNum()
    val = int(round(explicit_valence))

    # 1. Check if explicit valence matches ANY valid neutral state for this element
    neutral_valences = NEUTRAL_VALENCE_SETS.get(z)
    if neutral_valences and val in neutral_valences:
        return 0

    # 2. Check explicit ionic lookup table
    if (z, val) in CHARGED_STATE_LOOKUP:
        return CHARGED_STATE_LOOKUP[(z, val)]

    # Default fallback
    return 0


def solve_bond_ilp_scipy(
    probs: np.ndarray,
    edge_pairs: np.ndarray,
    atomic_nums: np.ndarray,
    max_valence_dict: dict = MAX_VALENCE_TABLE,
    min_valence_dict: dict = MIN_VALENCE_TABLE,
    enforce_connectivity: bool = True,
) -> np.ndarray:
    """Solves Integer Linear Program for bond assignment subject to min/max valence

    constraints and single-commodity flow connectivity constraints.
    """
    n_edges, n_classes = probs.shape
    n_atoms = len(atomic_nums)
    if n_edges == 0:
        return np.zeros(0, dtype=int)

    min_valences = np.array(
        [min_valence_dict.get(int(a), 0) for a in atomic_nums], dtype=float
    )
    max_valences = np.array(
        [max_valence_dict.get(int(a), 4) for a in atomic_nums], dtype=float
    )

    # Base cost: Minimize -log(P) for chosen bond types
    c_bonds = -np.log(np.clip(probs, 1e-8, 1.0)).reshape(-1)

    n_bond_vars = n_edges * n_classes
    # Add 2 flow variables per edge (one for each direction: u -> v and v -> u)
    n_flow_vars = (2 * n_edges) if (enforce_connectivity and n_atoms > 1) else 0
    total_vars = n_bond_vars + n_flow_vars

    c = np.zeros(total_vars)
    c[:n_bond_vars] = c_bonds  # Flow variables have zero cost

    # Constraint 1: Exactly 1 bond type chosen per edge
    row_a1 = np.repeat(np.arange(n_edges), n_classes)
    col_a1 = np.arange(n_classes * n_edges)
    data_a1 = np.ones(n_classes * n_edges)

    # Constraint 2: Total valence per atom
    row_a2, col_a2, data_a2 = [], [], []
    for e_idx, (u, v) in enumerate(edge_pairs):
        for k in range(n_classes):
            bo = BOND_ORDERS[k]
            if bo > 0:
                var_idx = n_classes * e_idx + k
                row_a2.extend([u, v])
                col_a2.extend([var_idx, var_idx])
                data_a2.extend([bo, bo])

    rows = list(row_a1) + [r + n_edges for r in row_a2]
    cols = list(col_a1) + col_a2
    data = list(data_a1) + data_a2

    lb_list = [np.ones(n_edges), min_valences]
    ub_list = [np.ones(n_edges), max_valences]
    row_count = n_edges + n_atoms

    # Constraint 3: Graph Connectivity via Single-Commodity Flow
    if enforce_connectivity and n_atoms > 1:
        flow_offset = n_bond_vars

        # 3a. Node Flow Conservation
        # Node 0 generates (N - 1) flow. Nodes 1..N-1 consume 1 unit.
        for i in range(n_atoms):
            for e_idx, (u, v) in enumerate(edge_pairs):
                f_uv = flow_offset + 2 * e_idx
                f_vu = flow_offset + 2 * e_idx + 1

                if u == i:
                    rows.append(row_count)
                    cols.append(f_uv)
                    data.append(1.0)  # Outflow
                    rows.append(row_count)
                    cols.append(f_vu)
                    data.append(-1.0)  # Inflow
                elif v == i:
                    rows.append(row_count)
                    cols.append(f_vu)
                    data.append(1.0)  # Outflow
                    rows.append(row_count)
                    cols.append(f_uv)
                    data.append(-1.0)  # Inflow

            req = float(n_atoms - 1) if i == 0 else -1.0
            lb_list.append([req])
            ub_list.append([req])
            row_count += 1

        # 3b. Edge Flow Capacity
        # Flow can only pass if edge e has a bond (k > 0): f_uv + f_vu <= (N - 1) * sum_{k>0} x_{e,k}
        big_m = float(n_atoms - 1)
        for e_idx in range(n_edges):
            f_uv = flow_offset + 2 * e_idx
            f_vu = flow_offset + 2 * e_idx + 1

            rows.append(row_count)
            cols.append(f_uv)
            data.append(1.0)
            rows.append(row_count)
            cols.append(f_vu)
            data.append(1.0)

            # Subtract Big-M * sum_{k>0} x_{e, k}
            for k in range(1, n_classes):
                rows.append(row_count)
                cols.append(e_idx * n_classes + k)
                data.append(-big_m)

            lb_list.append([-np.inf])
            ub_list.append([0.0])
            row_count += 1

    # Assemble Constraints and Bounds
    A = csc_matrix((data, (rows, cols)), shape=(row_count, total_vars))
    lb = np.concatenate(lb_list)
    ub = np.concatenate(ub_list)

    integrality = np.zeros(total_vars)
    integrality[:n_bond_vars] = 1.0  # Bond variables are binary integers

    bounds_ub = np.ones(total_vars)
    if enforce_connectivity and n_atoms > 1:
        bounds_ub[n_bond_vars:] = float(n_atoms - 1)  # Flow upper bound

    bounds = Bounds(np.zeros(total_vars), bounds_ub)
    constraints = LinearConstraint(A, lb, ub)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": 0.2},
    )

    if res.success:
        sol = res.x[:n_bond_vars].reshape(n_edges, n_classes)
        return np.argmax(sol, axis=1)

    print("MILP solver failed or returned infeasible solution.")
    return None


# def solve_bond_ilp_scipy(
#     probs: np.ndarray,
#     edge_pairs: np.ndarray,
#     atomic_nums: np.ndarray,
#     max_valence_dict: dict = MAX_VALENCE_TABLE,
#     min_valence_dict: dict = MIN_VALENCE_TABLE,  
# ) -> np.ndarray:
#     """Solves Integer Linear Program for bond assignment subject to min/max valence constraints."""
#     n_edges, n_classes = probs.shape
#     n_atoms = len(atomic_nums)
#     if n_edges == 0:
#         return np.zeros(0, dtype=int)

#     # Extract per-atom bounds (defaults fallback to 0 min and 4 max if unmapped)
#     min_valences = np.array(
#         [min_valence_dict.get(int(a), 0) for a in atomic_nums], dtype=float
#     )
#     max_valences = np.array(
#         [max_valence_dict.get(int(a), 4) for a in atomic_nums], dtype=float
#     )

#     # Minimize -log(P) equivalent to maximizing log-probability
#     c = -np.log(np.clip(probs, 1e-8, 1.0)).reshape(-1)  # [5 * E]

#     # Constraint 1: Exactly 1 bond type chosen per edge -> sum_k x_{e, k} = 1
#     row_a1 = np.repeat(np.arange(n_edges), n_classes)
#     col_a1 = np.arange(n_classes * n_edges)
#     data_a1 = np.ones(n_classes * n_edges)

#     # Constraint 2: Total valence per atom
#     row_a2 = []
#     col_a2 = []
#     data_a2 = []

#     for e_idx, (u, v) in enumerate(edge_pairs):
#         for k in range(n_classes):
#             bo = BOND_ORDERS[k]
#             if bo > 0:
#                 var_idx = n_classes * e_idx + k
#                 # Valence contribution to endpoint u
#                 row_a2.append(u)
#                 col_a2.append(var_idx)
#                 data_a2.append(bo)
#                 # Valence contribution to endpoint v
#                 row_a2.append(v)
#                 col_a2.append(var_idx)
#                 data_a2.append(bo)

#     row_ind = np.concatenate([row_a1, np.array(row_a2, dtype=int) + n_edges])
#     col_ind = np.concatenate([col_a1, np.array(col_a2, dtype=int)])
#     data_val = np.concatenate([data_a1, np.array(data_a2, dtype=float)])

#     A = csc_matrix(
#         (data_val, (row_ind, col_ind)),
#         shape=(n_edges + n_atoms, n_classes * n_edges),
#     )

#     # Edges must sum to 1; Atoms must be >= min_valences
#     lb = np.concatenate([np.ones(n_edges), min_valences])
#     ub = np.concatenate([np.ones(n_edges), max_valences])

#     constraints = LinearConstraint(A, lb, ub)
#     integrality = np.ones(n_classes * n_edges)  # Binary / Integer
#     bounds = Bounds(0, 1)

#     res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={'time_limit': 0.1})

#     if res.success:
#         sol = res.x.reshape(n_edges, n_classes)
#         return np.argmax(sol, axis=1)
    
#     # Fallback to standard argmax if MILP solver encounters an numerical anomaly / infeasibility
#     print("MILP solver failed or returned infeasible solution.")
#     return None


class MoleculeBuilder:
    """Build RDKit molecules from tensors.

    Args:
        vocab (str): The vocabulary to use. Options are "QM9" and "GEOM".
    """

    def __init__(self, vocab: str = "QM9") -> None:
        super().__init__()
        self.vocab = vocab
        if str(vocab).upper() == "QM9":
            self.atom_type_to_element = {
                1: "H",
                2: "C",
                3: "N",
                4: "O",
                5: "F",
            }
        elif str(vocab).upper() == "GEOM" or str(vocab).upper() == "CROSSDOCKED" or str(vocab).upper() == "SPINDR":
            self.atom_type_to_element = {
                1: "H",
                2: "B",
                3: "C",
                4: "N",
                5: "O",
                6: "F",
                7: "Al",
                8: "Si",
                9: "P",
                10: "S",
                11: "Cl",
                12: "As",
                13: "Br",
                14: "I",
                15: "Hg",
                16: "Bi",
            }
        else:
            raise ValueError(f"Unsupported vocabulary: {vocab}")

        # Atomic numbers for RDKit (element symbol -> atomic number)
        pt = Chem.GetPeriodicTable()
        self.atom_type_to_atomic_num = {
            k: pt.GetAtomicNumber(v) for k, v in self.atom_type_to_element.items()
        }
        # Precomputed tensor for fast lookup (used in bond predictor path)
        max_atom_type = max(self.atom_type_to_atomic_num.keys())
        self._atomic_num_lookup = torch.zeros(max_atom_type + 1, dtype=torch.long)
        for k, v in self.atom_type_to_atomic_num.items():
            self._atomic_num_lookup[k] = v
        self._atomic_num_lookup[0] = 1  # fallback for unknown

    def load_tensor_from_file(
        self, files_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load atom types, positions, and batch indices from a generated_mols.pt file.

        Args:
            files_path (str): The path to the generated_mols.pt file.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the atom types, positions, and batch indices.
        """
        generated_mols = torch.load(
            os.path.join(files_path, "generated_mols.pt"),
            weights_only=False,
            map_location="cpu",
        )
        return generated_mols.x, generated_mols.pos, generated_mols.batch

    def create_xyz_block(self, x: torch.Tensor, pos: torch.Tensor) -> str:
        """Create an XYZ block from a tensor of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).

        Returns:
            str: An XYZ block.
        """
        xyz_lines = []
        num_atoms = x.size(0)
        xyz_lines.append(f"{num_atoms}")
        xyz_lines.append("")

        for i in range(num_atoms):
            atom_type = x[i].item()
            element = self.atom_type_to_element.get(atom_type, "X")
            x_coord, y_coord, z_coord = pos[i].tolist()
            xyz_lines.append(f"{element}\t{x_coord:.4f}\t{y_coord:.4f}\t{z_coord:.4f}")

        return "\n".join(xyz_lines)

    def generate_rdkit_molecules_via_xyz2mol(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        progress_bar: bool = False,
        break_after_k_mols: int = None,
    ) -> list[Mol]:
        """Generate RDKit molecules from tensors of atom types and positions.

        Args:
            x (torch.Tensor): A tensor of shape (n_atoms,).
            pos (torch.Tensor): A tensor of shape (n_atoms, 3).
            batch (torch.Tensor): A tensor of shape (n_atoms,).

        Returns:
            list[Mol]: A list of RDKit molecules.
        """
        mols = []
        unique_batches = batch.unique().tolist()

        iterator = unique_batches
        if progress_bar:
            iterator = tqdm(unique_batches, desc="Generating RDKit molecules")
        for batch_id in iterator:
            mask = batch == batch_id
            x_mol = x[mask]
            pos_mol = pos[mask]

            xyz_block = self.create_xyz_block(x_mol, pos_mol)
            mol = rdmolfiles.MolFromXYZBlock(xyz_block)
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0, maxIterations=100000)
                Chem.SanitizeMol(mol)
                
            except Exception as e:
                logging.warning(
                    f"An error occurred while determining bonds for molecule in batch {batch_id}: {e}"
                )
                mol = None

            mols.append(mol)

            if break_after_k_mols is not None and len(mols) >= break_after_k_mols:
                break

        return mols
    

    def generate_rdkit_molecules_via_bond_predictor(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        bond_predictor_path: str = None,
        bond_predictor: BondPredictor = None,
        progress_bar: bool = False,
        break_after_k_mols: Optional[int] = None,
    ) -> list[Mol]:
        """Generate RDKit molecules from tensors using ILP bond optimization.

        Args:
            x: Atom types [n_atoms].
            pos: Coordinates [n_atoms, 3].
            batch: Batch indices [n_atoms].
            bond_predictor_path: Path to bond predictor checkpoint or directory.
            bond_predictor: Preloaded BondPredictor model. If provided, bond_predictor_path is ignored.
            progress_bar: Whether to show a progress bar.
            break_after_k_mols: Stop after generating this many molecules.

        Returns:
            list[Mol]: List of RDKit molecules (None for failed conversions).
        """
        if bond_predictor is None and bond_predictor_path is not None:
            ckpt_path = bond_predictor_path
            if os.path.isdir(ckpt_path):
                pattern = os.path.join(ckpt_path, "*.ckpt")
                matches = glob.glob(pattern)
                if not matches:
                    raise FileNotFoundError(
                        f"No bond predictor checkpoint found in {ckpt_path}."
                    )
                ckpt_path = matches[0]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Bond predictor checkpoint not found: {ckpt_path}")

            bond_predictor = BondPredictor.load_from_checkpoint(ckpt_path)

        elif bond_predictor is None:
            raise ValueError(
                "Either bond_predictor or bond_predictor_path must be provided."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, pos, batch = x.to(device), pos.to(device), batch.to(device)
        bond_predictor = bond_predictor.to(device).eval()
        successful_mols = torch.ones(batch.max().item() + 1, dtype=torch.bool, device="cpu")

        with torch.no_grad():
            probs, pair_indices = bond_predictor.predict_bonds(x, pos, batch, device)

            # Precompute local mapping for ILP assembly
            unique_batches, batch_counts = batch.unique(return_counts=True)
            local_idx = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            for m, b in enumerate(unique_batches):
                mask = batch == b
                local_idx[mask] = torch.arange(batch_counts[m].item(), dtype=torch.long, device=device)
            lookup_tensor = self._atomic_num_lookup.to(device)
            atomic_nums = lookup_tensor[
                x.clamp(0, lookup_tensor.shape[0] - 1)
            ]

            ilp_bond_types = torch.zeros(pair_indices.shape[0], dtype=torch.long, device=device)

            # 2. Solve ILP per molecule
            for m, b in enumerate(unique_batches):
                mol_mask = batch == b
                mol_atom_indices = torch.where(mol_mask)[0]
                n_atoms = len(mol_atom_indices)

                # Find edges belonging to current molecule
                mol_edge_mask = (batch[pair_indices[:, 0]] == b) & (batch[pair_indices[:, 1]] == b)
                mol_edge_indices = torch.where(mol_edge_mask)[0]

                if len(mol_edge_indices) > 0:
                    # Map global directed edge indices into local adj matrix
                    adj_edge_idx = torch.full((n_atoms, n_atoms), -1, dtype=torch.long, device=device)
                    src_local = local_idx[pair_indices[mol_edge_indices, 0]]
                    dst_local = local_idx[pair_indices[mol_edge_indices, 1]]
                    adj_edge_idx[src_local, dst_local] = mol_edge_indices

                    # Select undirected unique pairs (u_local < v_local)
                    u_loc, v_loc = torch.where(torch.triu(adj_edge_idx >= 0, diagonal=1))

                    e_fw = adj_edge_idx[u_loc, v_loc]
                    e_bw = adj_edge_idx[v_loc, u_loc]

                    # Symmetrize probabilities across directed edge pairs
                    valid_bw = e_bw >= 0
                    p_fw = probs[e_fw].cpu().numpy()
                    p_bw = np.zeros_like(p_fw)
                    p_bw[valid_bw.cpu().numpy()] = probs[e_bw[valid_bw]].cpu().numpy()

                    p_undirected = np.where(
                        valid_bw.cpu().numpy()[:, None], 0.5 * (p_fw + p_bw), p_fw
                    )

                    edge_pairs_local = torch.stack([u_loc, v_loc], dim=1).cpu().numpy()
                    mol_atomic_nums = atomic_nums[mol_mask].cpu().numpy()
                    mol_pos = pos[mol_mask].cpu().numpy()

                    # Solve ILP
                    chosen_bonds = solve_bond_ilp_scipy(
                        probs=p_undirected,
                        edge_pairs=edge_pairs_local,
                        atomic_nums=mol_atomic_nums,
                        max_valence_dict=MAX_VALENCE_TABLE,
                        min_valence_dict=MIN_VALENCE_TABLE,
                    )
                    if chosen_bonds is None:
                        successful_mols[b] = False
                        continue

                    chosen_bonds_tensor = torch.from_numpy(chosen_bonds).to(device)
                    ilp_bond_types[e_fw] = chosen_bonds_tensor
                    ilp_bond_types[e_bw[valid_bw]] = chosen_bonds_tensor[valid_bw]

        # 4. Transfer outputs to CPU for RDKit construction
        bond_types = ilp_bond_types.cpu()
        pair_indices = pair_indices.cpu()
        x, pos, batch = x.cpu(), pos.cpu(), batch.cpu()
        local_idx = local_idx.cpu()
        atomic_nums = atomic_nums.cpu()

        mols = []
        num_mols = len(unique_batches)
        iterator = range(num_mols)
        if progress_bar:
            iterator = tqdm(iterator, desc="Building molecules (ILP bond predictor)")

        for m in iterator:
            if not successful_mols[m]:
                mols.append(None)
                continue
            # (1) Load molecule-specific data
            b = unique_batches[m].item()
            mol_mask = batch == b
            n = torch.sum(mol_mask).item()

            edge_mask = (batch[pair_indices[:, 0]] == b) & (batch[pair_indices[:, 1]] == b)
            edge_mask = edge_mask & (bond_types > 0)

            if not edge_mask.any():
                bonded_pairs = pair_indices.new_empty(0, 2)
                bonded_types = bond_types.new_empty(0)
            else:
                bonded_pairs = pair_indices[edge_mask]
                bonded_types = bond_types[edge_mask]
                # Keep unique undirected pairs (i < j)
                keep = bonded_pairs[:, 0] < bonded_pairs[:, 1]
                bonded_pairs = bonded_pairs[keep]
                bonded_types = bonded_types[keep]

            i_local = local_idx[bonded_pairs[:, 0]]
            j_local = local_idx[bonded_pairs[:, 1]]
            mol_atomic_nums = atomic_nums[mol_mask]


            # (2) Create RDKit RWMol and add atoms
            rwmol = Chem.RWMol()
            for i in range(n):
                atom = Chem.Atom(int(mol_atomic_nums[i].item()))
                rwmol.AddAtom(atom)

            # (3) Add bonds based from ILP solution
            for idx in range(bonded_pairs.shape[0]):
                bt = bonded_types[idx].item()
                rdkit_bt = RDKIT_BOND_TYPES[bt]
                if rdkit_bt is not None:
                    try:
                        rwmol.AddBond(
                            i_local[idx].item(), j_local[idx].item(), rdkit_bt
                        )
                    except Exception as e:
                        print(f"Error occurred while adding bond: {e}")
                        pass
                    
            rwmol.UpdatePropertyCache(strict=False)
            
            # (4) Compute explicit valence and assign formal charges via lookup
            for atom in rwmol.GetAtoms():
                explicit_val = atom.GetExplicitValence()  # Calculates sum of bond orders added
                charge = assign_rule_based_formal_charge(atom, explicit_val)
                if charge != 0:
                    atom.SetFormalCharge(charge)

            # (5) Create conformer and assign 3D coordinates
            try:
                mol = rwmol.GetMol()

                mol_pos = pos[mol_mask]
                conf = Chem.Conformer(n)
                for i in range(n):
                    x_coord, y_coord, z_coord = mol_pos[i].tolist()
                    conf.SetAtomPosition(
                        i, (float(x_coord), float(y_coord), float(z_coord))
                    )
                mol.AddConformer(conf, assignId=True)

                # (6) Keep only if sanitizable and single fragment
                Chem.SanitizeMol(mol)
                
                if len(Chem.GetMolFrags(mol)) > 1:
                    print(f"Warning: Molecule in batch {b} has multiple fragments.")
                    mols.append(None)
                    continue
                else:
                    mols.append(mol)
                
            except Exception as e:
                print(f"Error occurred while building molecule: {e}")
                mols.append(None)

            if break_after_k_mols is not None and len(mols) >= break_after_k_mols:
                break

        return mols
