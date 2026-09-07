#!/usr/bin/env python3
"""
Extract a SPINDR-style protein pocket and generate molecules conditioned on it.

Inputs are explicit protein and ligand file paths. Optionally pass ``--fragment``
to seed generation from a fragment SDF (same coordinate frame as the target).
Outputs are written under::

  generated_mols/<session>/run_<N>/
    pocket.pdb
    <copy of protein>
    <copy of ligand>
    <copy of fragment>   # only if --fragment is set
    generated_mols.sdf
    generated_mols.pt

``--session`` groups related runs; each invocation creates the next ``run_N``.

Protein protonation via hydride is enabled by default.

Pocket cutting reimplements FlowR's logic (`process_pdb`) without importing
FlowR. Preprocessing uses ``_process_protein_ligand_complex`` /
``SpindrDataSet.collate_pocket_info`` from ``neat.dataset.dataset_spindr``.
The model runs in a ligand-COM-centered frame during sampling, then
``NEAT.generate`` recenters outputs onto the pocket COM. Generated molecules
are translated by that pocket COM back into the target reference frame.

Example
-------
python scripts/generate_from_pdb.py --protein ./1KE6.cif --ligand ./lig.sdf
python scripts/generate_from_pdb.py --protein ./prot.pdb --ligand ./lig.sdf \\
    --session cdk2_screen --num_molecules 50
python scripts/generate_from_pdb.py --protein ./prot.pdb --ligand ./lig.sdf \\
    --fragment ./frag.sdf
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Union

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch
import torch_geometric
import yaml
from Bio.PDB.Polypeptide import is_aa
from biotite.structure import AtomArray, BondList
from lightning import seed_everything
from rdkit import Chem

from neat.dataset.dataset_spindr import (SpindrDataSet,
                                         _process_protein_ligand_complex)
from neat.dataset.dataset_utils import _largest_fragment, _ligand_features
from neat.model import NEAT
from neat.model.bond_predictor import BondPredictor
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils import save_molecules_to_sdf

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(42)
seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.getcwd()
DEFAULT_CONFIG = os.path.join(
    ROOT, "scripts", "config_files", "config_generation_conditional.yaml"
)
DEFAULT_SESSION = "default"
OUTPUT_ROOT = Path(ROOT) / "generated_mols"
PROTEIN_SUFFIXES = {".pdb", ".cif", ".ent"}
LIGAND_SUFFIXES = {".sdf", ".mol"}
RUN_DIR_RE = re.compile(r"^run_(\d+)$")


# ---------------------------------------------------------------------------
# Ligand I/O
# ---------------------------------------------------------------------------


def load_ligand_mol(ligand_path: Union[str, Path]) -> Chem.Mol:
    """Load ligand as RDKit Mol from SDF (OpenBabel fallback)."""
    ligand_path = Path(ligand_path)
    mol = None
    try:
        mol = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
        if mol is None:
            mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)
    except Exception:
        mol = None

    if mol is None:
        try:
            from openbabel import pybel

            mol_ob = next(pybel.readfile("sdf", str(ligand_path)), None)
            if mol_ob is not None:
                mol = Chem.MolFromMolBlock(mol_ob.write("mol"), removeHs=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read ligand from {ligand_path} with RDKit "
                f"or OpenBabel: {exc}"
            ) from exc

    if mol is None:
        raise RuntimeError(f"Could not parse ligand molecule from {ligand_path}")
    return mol


def load_ligand_coords(ligand_path: Union[str, Path]) -> np.ndarray:
    """Load ligand atom coordinates from SDF. Shape (N, 3)."""
    mol = load_ligand_mol(ligand_path)
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    if coords.size == 0:
        raise RuntimeError(f"Ligand {ligand_path} has no atoms / conformer.")
    return coords


# ---------------------------------------------------------------------------
# Protein / pocket extraction (FlowR SPINDR-style)
# ---------------------------------------------------------------------------


def load_protein_structure(
    protein_path: Union[str, Path],
    add_bonds: bool = True,
    add_hs: bool = True,
) -> AtomArray:
    """Load a protein structure from PDB or mmCIF via biotite."""
    protein_path = Path(protein_path)
    suffix = protein_path.suffix.lower()

    if suffix == ".cif":
        file_obj = pdbx.CIFFile.read(str(protein_path))
        read_fn = pdbx.get_structure
    elif suffix in {".pdb", ".ent"}:
        file_obj = pdb.PDBFile.read(str(protein_path))
        read_fn = pdb.get_structure
    else:
        raise ValueError(
            f"Unsupported protein format '{suffix}'. Use .pdb or .cif."
        )

    try:
        structure = read_fn(
            file_obj, model=1, extra_fields=["charge"], include_bonds=add_bonds
        )
    except Exception:
        structure = read_fn(file_obj, model=1, include_bonds=add_bonds)

    if add_hs:
        try:
            import hydride
        except ImportError as exc:
            raise ImportError(
                "hydride is required for hydrogen addition (default). "
                "Install hydride or pass --no_add_hs."
            ) from exc
        structure, _mask = hydride.add_hydrogen(structure)
        structure.coord = hydride.relax_hydrogen(structure)

    if structure.bonds is None:
        bonds = struc.connect_via_residue_names(structure, inter_residue=True)
        structure.bonds = bonds
        if structure.bonds.as_array().shape[0] == 0:
            raise ValueError(f"No bonds found in structure {protein_path}")

    return structure


def cut_pocket(
    structure: AtomArray,
    ligand_coords: np.ndarray,
    pocket_cutoff: float = 6.0,
) -> AtomArray:
    """Cut a SPINDR/FlowR-style binding pocket around a ligand.

    Algorithm (identical to FlowR ``process_pdb`` with ``cut_pocket=True``):
      1. Keep chains that have any atom within ``pocket_cutoff`` of the ligand.
      2. Within those chains, keep whole standard amino-acid residues whose
         closest atom is within ``pocket_cutoff`` of the ligand.
    """
    distances = np.linalg.norm(
        structure.coord[:, None, :] - ligand_coords[None, :, :],
        axis=-1,
    )
    atoms_in_pocket = (distances < pocket_cutoff).any(axis=1)
    chains_in_pocket = list(structure.chain_id[atoms_in_pocket])
    chains_in_pocket = [chain for chain in chains_in_pocket if len(chain) > 0]

    structure = structure[np.isin(structure.chain_id, chains_in_pocket)]

    chain_res_pairs = set(zip(structure.chain_id, structure.res_id))
    res_filter_mask = np.zeros(len(structure), dtype=bool)

    for chain_id, res_id in chain_res_pairs:
        res_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
        res = structure[res_mask]

        if (
            is_aa(res.res_name[0], standard=True)
            and (
                np.linalg.norm(
                    res.coord[:, None, :] - ligand_coords[None, :, :],
                    axis=-1,
                )
            ).min()
            < pocket_cutoff
        ):
            res_filter_mask |= res_mask

    return structure[res_filter_mask]


# ---------------------------------------------------------------------------
# Minimal pocket container + PDB writer (FlowR ProteinPocket subset)
# ---------------------------------------------------------------------------


class ProteinPocket:
    """Minimal pocket holder with biotite PDB writer (FlowR-compatible)."""

    def __init__(self, atoms: AtomArray, bonds: Optional[BondList] = None):
        annotations = atoms.get_annotation_categories()
        if "res_name" not in annotations:
            raise RuntimeError("atom array must contain key res_name")
        if "element" not in annotations:
            raise RuntimeError("atom array must contain key element")

        if "charge" not in annotations:
            atoms.add_annotation("charge", np.float32)

        self.atoms = atoms
        self.bonds = bonds if bonds is not None else atoms.bonds

    def __len__(self) -> int:
        return len(self.atoms)

    @staticmethod
    def from_pocket_atoms(
        atoms: AtomArray,
        infer_res_bonds: bool = False,
    ) -> "ProteinPocket":
        if infer_res_bonds:
            bonds = struc.connect_via_residue_names(atoms, inter_residue=True)
        else:
            bonds = atoms.bonds
        return ProteinPocket(atoms, bonds)

    def write_pdb(self, filepath: Union[str, Path], include_bonds: bool = True) -> None:
        """Write pocket as PDB. Multi-char chain IDs are truncated to 1 char."""
        if len(self.atoms) > 0 and len(self.atoms.chain_id[0]) > 1:
            self.atoms.chain_id = np.array([cid[-1] for cid in self.atoms.chain_id])

        if include_bonds:
            if self.bonds is None:
                raise AssertionError(
                    "Bonds must be provided to include them as CONECT entries"
                )
            self.atoms.bonds = self.bonds
        else:
            self.atoms.bonds = None

        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, self.atoms)
        pdb_file.write(Path(filepath))

        if include_bonds:
            self.atoms.bonds = None


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def extract_spindr_pocket(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    pocket_cutoff: float = 6.0,
    add_hs_to_protein: bool = True,
    min_pocket_atoms: int = 10,
) -> ProteinPocket:
    """Build a SPINDR-style pocket from protein + ligand files."""
    ligand_coords = load_ligand_coords(ligand_path)
    structure = load_protein_structure(
        protein_path,
        add_bonds=True,
        add_hs=add_hs_to_protein,
    )
    structure = cut_pocket(structure, ligand_coords, pocket_cutoff=pocket_cutoff)
    pocket = ProteinPocket.from_pocket_atoms(structure)

    if len(pocket) < min_pocket_atoms:
        raise RuntimeError(
            f"Pocket too small after cutting ({len(pocket)} atoms; "
            f"threshold={min_pocket_atoms}). Check ligand placement / cutoff."
        )
    return pocket


def validate_input_paths(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    fragment_path: Union[str, Path, None] = None,
) -> tuple[Path, Path, Path | None]:
    """Validate protein/ligand/(optional) fragment paths and suffixes."""
    protein_path = Path(protein_path).expanduser().resolve()
    ligand_path = Path(ligand_path).expanduser().resolve()

    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")
    if protein_path.suffix.lower() not in PROTEIN_SUFFIXES:
        raise ValueError(
            f"Unsupported protein format '{protein_path.suffix}'. "
            f"Expected one of {sorted(PROTEIN_SUFFIXES)}."
        )

    if not ligand_path.is_file():
        raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
    if ligand_path.suffix.lower() not in LIGAND_SUFFIXES:
        raise ValueError(
            f"Unsupported ligand format '{ligand_path.suffix}'. "
            f"Expected one of {sorted(LIGAND_SUFFIXES)}."
        )

    resolved_fragment = None
    if fragment_path is not None:
        resolved_fragment = Path(fragment_path).expanduser().resolve()
        if not resolved_fragment.is_file():
            raise FileNotFoundError(f"Fragment file not found: {resolved_fragment}")
        if resolved_fragment.suffix.lower() not in LIGAND_SUFFIXES:
            raise ValueError(
                f"Unsupported fragment format '{resolved_fragment.suffix}'. "
                f"Expected one of {sorted(LIGAND_SUFFIXES)}."
            )

    return protein_path, ligand_path, resolved_fragment


def create_run_dir(session: str = DEFAULT_SESSION) -> Path:
    """Create ``generated_mols/<session>/run_<N>`` with the next free index."""
    session_dir = OUTPUT_ROOT / session
    session_dir.mkdir(parents=True, exist_ok=True)

    existing: list[int] = []
    for path in session_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.match(path.name)
        if match:
            existing.append(int(match.group(1)))

    run_idx = max(existing) + 1 if existing else 0
    run_dir = session_dir / f"run_{run_idx}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def prepare_run_dir(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    session: str = DEFAULT_SESSION,
    pocket_cutoff: float = 6.0,
    add_hs: bool = True,
    fragment_path: Union[str, Path, None] = None,
) -> tuple[Path, Path, Path, Path | None, torch.Tensor]:
    """Create a run folder, copy inputs, and write the extracted pocket.

    Returns
    -------
    (run_dir, pocket_pdb_path, ligand_copy_path, fragment_copy_path_or_none,
     pocket_com)
        ``pocket_com`` is the pocket geometric center in the target frame.
        ``model.generate`` returns coordinates centered on the pocket COM, so
        this vector maps generated molecules back into the target frame.
    """
    protein_path, ligand_path, fragment_path = validate_input_paths(
        protein_path, ligand_path, fragment_path=fragment_path
    )
    run_dir = create_run_dir(session=session)

    protein_copy = run_dir / protein_path.name
    ligand_copy = run_dir / ligand_path.name
    shutil.copy2(protein_path, protein_copy)
    shutil.copy2(ligand_path, ligand_copy)

    fragment_copy = None
    if fragment_path is not None:
        fragment_name = fragment_path.name
        if fragment_name in {protein_path.name, ligand_path.name}:
            fragment_name = f"fragment{fragment_path.suffix.lower()}"
        fragment_copy = run_dir / fragment_name
        shutil.copy2(fragment_path, fragment_copy)

    pocket = extract_spindr_pocket(
        protein_path=protein_path,
        ligand_path=ligand_path,
        pocket_cutoff=pocket_cutoff,
        add_hs_to_protein=add_hs,
    )
    pocket_path = run_dir / "pocket.pdb"
    pocket.write_pdb(pocket_path, include_bonds=True)
    pocket_com = torch.tensor(
        np.asarray(pocket.atoms.coord, dtype=np.float64).mean(axis=0),
        dtype=torch.float32,
    )

    n_res = len(set(zip(pocket.atoms.chain_id, pocket.atoms.res_id)))
    print(
        f"Protein: {protein_path.name}  |  Ligand: {ligand_path.name}"
        + (f"  |  Fragment: {fragment_path.name}" if fragment_path else "")
        + f"\nSession: {session}  |  Run: {run_dir.name}\n"
        f"Wrote pocket ({len(pocket)} atoms, {n_res} residues, "
        f"cutoff={pocket_cutoff} Å, add_hs={add_hs})\n"
        f"  -> {run_dir.resolve()}"
    )
    return run_dir, pocket_path, ligand_copy, fragment_copy, pocket_com


def compute_preprocess_center(ligand_path: Path) -> torch.Tensor:
    """Ligand COM used by SPINDR preprocessing (target-frame origin for the model)."""
    suppl = Chem.SDMolSupplier(str(ligand_path), sanitize=True, removeHs=False)
    if suppl is None or len(suppl) == 0 or suppl[0] is None:
        raise RuntimeError(f"Could not load ligand for centering from {ligand_path}")
    rdmol = _largest_fragment(suppl[0])
    _lig_x, lig_pos, _lig_charge = _ligand_features(rdmol, get_charge=True)
    if lig_pos is None:
        raise RuntimeError(f"Could not get ligand coordinates from {ligand_path}")
    return lig_pos.mean(dim=0)


def load_fragment_mol(fragment_path: Path) -> Chem.Mol:
    """Load a fragment SDF for seeded generation (target reference frame)."""
    supplier = Chem.SDMolSupplier(
        str(fragment_path), removeHs=False, sanitize=False
    )
    if supplier is None or len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Failed to read fragment from {fragment_path}.")
    return supplier[0]


def translate_mol(mol: Chem.Mol, shift: torch.Tensor | np.ndarray) -> Chem.Mol:
    """Return a copy of ``mol`` with all atom coordinates translated by ``shift``."""
    mol = Chem.Mol(mol)
    shift = np.asarray(shift, dtype=np.float64).reshape(3)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(
            i, (float(pos.x + shift[0]), float(pos.y + shift[1]), float(pos.z + shift[2]))
        )
    return mol


def prepare_fragment_info(fragment: Chem.Mol, num_molecules: int) -> dict:
    """Build fragment tensors for model.generate() (single pocket, N samples)."""
    x, pos = _ligand_features(fragment, get_charge=False)
    if x is None or pos is None:
        raise RuntimeError("Could not extract fragment features for generation.")
    return {
        "fragment_x": torch.cat([x for _ in range(num_molecules)], dim=0),
        "fragment_pos": torch.cat([pos for _ in range(num_molecules)], dim=0),
        "fragment_batch": torch.cat(
            [
                torch.ones(len(x), dtype=torch.long) * j
                for j in range(num_molecules)
            ],
            dim=0,
        ),
    }


def preprocess_pocket(pocket_path: Path, ligand_path: Path):
    """Preprocess pocket/ligand with SPINDR dataset logic into a PyG Data object."""
    data = _process_protein_ligand_complex(
        pocket_path=pocket_path,
        ligand_path=ligand_path,
        split_name="inference",
    )
    if data is None:
        raise RuntimeError(
            f"SPINDR preprocessing failed for pocket={pocket_path} "
            f"ligand={ligand_path}. Check that the ligand can be sanitized "
            "and all atom types are in the vocabulary."
        )
    return data


def load_model_and_bond_predictor(params: dict):
    """Load NEAT checkpoint and optional bond predictor from config params."""
    checkpoints_dir = os.path.join(ROOT, params["checkpoints_path"], "checkpoints")
    pt_files = [
        f
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".ckpt") and f.startswith("periodic-epoch")
    ]
    if not pt_files:
        pt_files = [
            f
            for f in os.listdir(checkpoints_dir)
            if f.endswith(".ckpt") and "best-val-loss" in f
        ]
    if not pt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoints_dir}")

    checkpoints_path = os.path.join(checkpoints_dir, pt_files[0])
    print(f"Using checkpoint file: {checkpoints_path}")
    model = NEAT.load_from_checkpoint(checkpoints_path, map_location=DEVICE)

    bond_predictor_dir = params.get("bond_predictor_dir", None)
    if bond_predictor_dir is not None:
        bond_predictor_dir = os.path.join(ROOT, bond_predictor_dir, "checkpoints")
        bp_files = [f for f in os.listdir(bond_predictor_dir) if f.endswith(".ckpt")]
        if not bp_files:
            raise FileNotFoundError(f"No .ckpt files found in {bond_predictor_dir}")
        bond_predictor_path = os.path.join(bond_predictor_dir, bp_files[0])
        print(f"Using bond predictor: {bond_predictor_path}")
        bond_predictor = BondPredictor.load_from_checkpoint(
            bond_predictor_path, map_location=DEVICE
        )
    else:
        bond_predictor = None

    return model, bond_predictor


def generate_molecules(
    pocket_data,
    num_molecules: int,
    params: dict,
    out_dir: Path,
    frame_center: torch.Tensor,
    fragment_info: dict | None = None,
) -> None:
    """Generate ligands conditioned on a preprocessed pocket.

    ``model.generate`` returns coordinates centered on the pocket COM (see
    step 5 in ``NEAT.generate``). ``frame_center`` should therefore be the
    pocket geometric center in the target frame so outputs land on the
    original protein/ligand coordinates.
    """
    model, bond_predictor = load_model_and_bond_predictor(params)

    pocket_info = SpindrDataSet.collate_pocket_info(
        [pocket_data], samples_per_pocket=num_molecules, device=DEVICE
    )

    mode = "fragment-seeded" if fragment_info is not None else "pocket-conditioned"
    print(f"Generating {num_molecules} molecules ({mode})...")
    with torch.no_grad():
        model.eval()
        cfg_factor = params.get("cfg_factor", 0.0)
        generated_mols = model.generate(
            batch_size=pocket_info["pocket_batch"].max().item() + 1,
            max_atoms=params["max_atoms"],
            num_time_steps=params["num_time_steps"],
            time_step_spacing=params["time_step_spacing"],
            integration_method=params["integration_method"],
            pocket_info=pocket_info,
            fragment_info=fragment_info,
            cfg_factor=cfg_factor,
            device=DEVICE,
        )

    generated_mols.batch -= generated_mols.batch.min()
    # Map pocket-COM-centered model coords back to the target frame.
    center = frame_center.to(
        device=generated_mols.pos.device, dtype=generated_mols.pos.dtype
    )
    generated_mols.pos = generated_mols.pos + center
    torch.save(generated_mols, out_dir / "generated_mols.pt")

    builder = MoleculeBuilder(vocab=params["data_set"])
    if bond_predictor is not None:
        rdkit_mols = builder.generate_rdkit_molecules_via_bond_predictor(
            generated_mols.x,
            generated_mols.pos,
            generated_mols.batch,
            bond_predictor=bond_predictor,
            progress_bar=True,
        )
    else:
        rdkit_mols = builder.generate_rdkit_molecules_via_xyz2mol(
            generated_mols.x,
            generated_mols.pos,
            generated_mols.batch,
            progress_bar=True,
        )
    save_molecules_to_sdf(rdkit_mols, out_dir / "generated_mols.sdf")
    print(
        f"Wrote generated molecules (target reference frame):\n"
        f"  -> {(out_dir / 'generated_mols.pt').resolve()}\n"
        f"  -> {(out_dir / 'generated_mols.sdf').resolve()}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut a SPINDR-style pocket from a protein + ligand, preprocess it, "
            "and generate molecules into generated_mols/<session>/run_<N>/."
        )
    )
    parser.add_argument(
        "--protein",
        type=str,
        required=True,
        help="Path to the protein structure (.pdb / .cif / .ent).",
    )
    parser.add_argument(
        "--ligand",
        type=str,
        required=True,
        help="Path to the reference ligand (.sdf / .mol).",
    )
    parser.add_argument(
        "--fragment",
        type=str,
        default=None,
        help=(
            "Optional fragment SDF/MOL in the same coordinate frame as the "
            "protein/ligand. If set, all molecules are seeded from this fragment."
        ),
    )
    parser.add_argument(
        "--session",
        type=str,
        default=DEFAULT_SESSION,
        help=(
            "Session folder under generated_mols/ for grouping related runs "
            f"(default: {DEFAULT_SESSION})."
        ),
    )
    parser.add_argument(
        "--pocket_cutoff",
        type=float,
        default=6.0,
        help="Distance cutoff in Angstroms for pocket residue selection (default: 6.0).",
    )
    parser.add_argument(
        "--no_add_hs",
        action="store_true",
        help="Disable protein protonation with hydride (enabled by default).",
    )
    parser.add_argument(
        "--num_molecules",
        type=int,
        default=100,
        help="Number of molecules to generate (default: 100).",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help=(
            "Config file for generation (default: "
            "scripts/config_files/config_generation_conditional.yaml)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir, pocket_path, ligand_path, fragment_path, pocket_com = prepare_run_dir(
        protein_path=args.protein,
        ligand_path=args.ligand,
        session=args.session,
        pocket_cutoff=args.pocket_cutoff,
        add_hs=not args.no_add_hs,
        fragment_path=args.fragment,
    )

    # Ligand COM subtracted during SPINDR preprocess (model input frame).
    ligand_com = compute_preprocess_center(ligand_path)
    pocket_data = preprocess_pocket(pocket_path, ligand_path)
    print(
        f"Preprocessed pocket: {pocket_data.pocket_x.numel()} atoms, "
        f"ligand SMILES={pocket_data.smiles}\n"
        f"Ligand COM (preprocess): {ligand_com.tolist()}\n"
        f"Pocket COM (output frame): {pocket_com.tolist()}"
    )

    fragment_info = None
    if fragment_path is not None:
        fragment_mol = load_fragment_mol(fragment_path)
        # Pocket tensors are ligand-COM centered; put the fragment in that frame.
        fragment_model_frame = translate_mol(fragment_mol, -ligand_com.numpy())
        fragment_info = prepare_fragment_info(
            fragment_model_frame, args.num_molecules
        )
        print(
            f"Using fragment seed: {fragment_path.name} "
            f"({fragment_mol.GetNumAtoms()} atoms)"
        )

    config_path = args.config_file or DEFAULT_CONFIG
    print(f"Using config: {config_path}")
    params = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # model.generate recenters outputs onto the pocket COM; add it back so
    # generated molecules sit in the target / pocket.pdb reference frame.
    generate_molecules(
        pocket_data=pocket_data,
        num_molecules=args.num_molecules,
        params=params,
        out_dir=run_dir,
        frame_center=pocket_com,
        fragment_info=fragment_info,
    )


if __name__ == "__main__":
    main()
