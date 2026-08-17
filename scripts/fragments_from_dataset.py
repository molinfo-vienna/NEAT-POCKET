f"""Precompute BRICS fragments from CrossDocked or SPINDR test ligands for FBDD generation.

For each test pocket:
  1. Center the pocket and shift the reference ligand into that frame.
  2. Break the ligand with BRICS and keep the largest, second-largest,
     and smallest fragments (dummy atoms stripped).
  3. Write fragments/DATASET/FRAGMENT_TYPE/PDB_CODE.sdf.

Also writes:
  - fragments/DATASET/fragment_statistics.png
  - fragments/DATASET/fragments_logs.log

Run this before generation_conditional.py when fragment_type is set.
"""

import argparse
import logging
import os
import shutil
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from rdkit import Chem
from rdkit.Chem import BRICS, SDWriter

from neat.dataset import DataModule
from neat.dataset.dataset_crossdocked import _largest_fragment
from neat.utils import center_pdb, cif_2_pdb

plt.rcParams["font.size"] = 18

ROOT = os.getcwd()
FRAGMENTS_DIR = os.path.join(ROOT, "fragments")
FRAGMENT_TYPES = ("largest", "second_largest", "smallest")
PALETTE = sns.color_palette("rocket")
FRAGMENT_COLORS = {
    "largest": PALETTE[1],
    "second_largest": PALETTE[2],
    "smallest": PALETTE[4],
}


def plot_fragment_statistics(
    fragments_by_type: dict[str, List[Chem.Mol]],
    ligands_by_type: dict[str, List[Chem.Mol]],
    output_path: str,
) -> None:
    """Plot absolute and relative fragment-size histograms (3 rows x 2 cols).

    Rows are largest / second_largest / smallest. Absolute plots share one
    heavy-atom x-axis; relative plots share [0, 1]. Dashed lines mark means.

    Args:
        fragments_by_type: Successfully written fragments per rank.
        ligands_by_type: Matching full ligands (same length per rank).
        output_path: Destination PNG path.
    """
    # Precompute counts so absolute bins can be shared across all rows.
    counts_by_type = {}
    relative_by_type = {}
    global_max_atoms = 0

    for fragment_type in FRAGMENT_TYPES:
        fragment_list = fragments_by_type[fragment_type]
        ligand_list = ligands_by_type[fragment_type]
        if len(fragment_list) != len(ligand_list):
            raise ValueError(
                f"{fragment_type}: fragment and ligand lists must be the same length."
            )
        if not fragment_list:
            counts_by_type[fragment_type] = None
            relative_by_type[fragment_type] = None
            continue

        frag_atom_counts = np.array(
            [mol.GetNumHeavyAtoms() for mol in fragment_list], dtype=float
        )
        ligand_atom_counts = np.array(
            [mol.GetNumHeavyAtoms() for mol in ligand_list], dtype=float
        )
        counts_by_type[fragment_type] = frag_atom_counts
        relative_by_type[fragment_type] = frag_atom_counts / (ligand_atom_counts + 1e-9)
        global_max_atoms = max(global_max_atoms, int(np.max(frag_atom_counts)))

    abs_bins = np.arange(0, global_max_atoms + 2)
    abs_xlim = (-0.5, global_max_atoms + 0.5)
    abs_xticks = np.arange(0, global_max_atoms + 1, max(1, global_max_atoms // 10))
    rel_bins = np.linspace(0.0, 1.0, 21)

    fig, axes = plt.subplots(3, 2, figsize=(30, 18))

    for row, fragment_type in enumerate(FRAGMENT_TYPES):
        frag_atom_counts = counts_by_type[fragment_type]
        relative_sizes = relative_by_type[fragment_type]

        logging.info(f"Fragment type: {fragment_type}")
        logging.info(f"Average fragment heavy atom count: {frag_atom_counts.mean()}")
        logging.info(f"Average fragment relative size: {relative_sizes.mean()}")

        if frag_atom_counts is None:
            axes[row, 0].set_visible(False)
            axes[row, 1].set_visible(False)
            continue

        ax_abs, ax_rel = axes[row, 0], axes[row, 1]
        row_label = fragment_type.replace("_", " ")
        color = FRAGMENT_COLORS[fragment_type]

        ax_abs.hist(
            frag_atom_counts,
            bins=abs_bins,
            color=color,
            rwidth=0.8,
            align="left",
        )
        ax_abs.axvline(
            frag_atom_counts.mean(), color=color, linestyle="--", linewidth=3
        )
        ax_abs.set_xlim(*abs_xlim)
        ax_abs.set_xticks(abs_xticks)
        ax_abs.set_xlabel("Number of heavy atoms")
        ax_abs.set_ylabel("Frequency")
        ax_abs.set_title(f"{row_label.capitalize()}: absolute size")

        ax_rel.hist(
            relative_sizes,
            bins=rel_bins,
            color=color,
            rwidth=0.8,
        )
        ax_rel.axvline(relative_sizes.mean(), color=color, linestyle="--", linewidth=3)
        ax_rel.set_xlim(0.0, 1.0)
        ax_rel.set_xlabel("Relative size (fraction of ligand heavy atoms)")
        ax_rel.set_ylabel("Frequency")
        ax_rel.set_title(f"{row_label.capitalize()}: relative size")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote fragment statistics to {output_path}.")


def get_fragments_with_brics(mol: Chem.Mol) -> dict[str, Chem.Mol]:
    """COM-center a ligand, BRICS-fragment it, and return size-ranked pieces.

    Steps:
      1. Shift the conformer so the unweighted center of mass is at the origin
         (in-place). Coordinates then match the frame used at generation time.
      2. Break BRICS bonds (dummy atoms mark cut points).
      3. Strip dummies and rank fragments by heavy-atom count.

    If BRICS yields fewer than two pieces, the (centered) full molecule is
    returned for every rank.

    Args:
        mol: Ligand already in the pocket-centered frame, with hydrogens.

    Returns:
        Dict with keys largest, second_largest, and smallest.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "The input molecule must have a 3D conformation to calculate the center of mass."
        )

    # Center on the ligand COM so fragment coords are ligand-COM-relative.
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    com = np.mean(coords, axis=0)
    for i in range(num_atoms):
        original_pos = conf.GetAtomPosition(i)
        centered_pos = original_pos - com
        conf.SetAtomPosition(i, centered_pos)

    fragmented_mol = BRICS.BreakBRICSBonds(mol)
    fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)

    # No cleavable BRICS bond: treat the whole ligand as every rank.
    if not fragments or len(fragments) <= 1:
        print("No fragments found; returning the full molecule for all ranks.")
        # return {fragment_type: mol for fragment_type in FRAGMENT_TYPES}
        return None

    clean_frags = [
        Chem.DeleteSubstructs(frag, Chem.MolFromSmiles("*")) for frag in fragments
    ]
    order = np.argsort([frag.GetNumHeavyAtoms() for frag in clean_frags])
    largest = clean_frags[order[-1]]
    smallest = clean_frags[order[0]]
    second_largest = clean_frags[order[-2]] if len(order) >= 2 else largest

    return {
        "largest": largest,
        "second_largest": second_largest,
        "smallest": smallest,
    }


def write_fragment(fragment: Chem.Mol, out_sdf_file: str) -> None:
    """Write a single fragment molecule to an SDF file."""
    writer = SDWriter(out_sdf_file)
    writer.SetKekulize(False)
    writer.write(fragment)
    writer.close()


def extract_fragments(dataset: str) -> None:
    """Extract and save BRICS fragments from the CrossDocked or SPINDR test set.

    Args:
        dataset: Dataset name (CrossDocked or SPINDR).
    """
    # --- Data ---

    datamodule = DataModule(
        os.path.join(ROOT, "data"),
        dataset,
    )
    datamodule.setup()
    test_data = datamodule.test_data

    # Output layout: fragments/{largest,second_largest,smallest}/{pdb}.sdf
    fragments_dir = os.path.join(FRAGMENTS_DIR, dataset)
    os.makedirs(fragments_dir, exist_ok=True)
    for fragment_type in FRAGMENT_TYPES:
        os.makedirs(os.path.join(fragments_dir, fragment_type), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(fragments_dir, f"fragments_logs.log"),
                mode="w",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    # Temporary dir for centered pocket PDBs (discarded after the run).
    tmp_dir = os.path.join(fragments_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    fragments_by_type = {fragment_type: [] for fragment_type in FRAGMENT_TYPES}
    ligands_by_type = {fragment_type: [] for fragment_type in FRAGMENT_TYPES}
    n_failed = {fragment_type: 0 for fragment_type in FRAGMENT_TYPES}
    n_samples = len(test_data)

    for data_idx in range(n_samples):
        pdb_code = test_data.get_pdb_code_from_data_point(test_data[data_idx])
        in_pdb_file = test_data.get_pocket_path_from_data_point(test_data[data_idx])
        out_pdb_file = os.path.join(tmp_dir, f"{pdb_code}_pocket.pdb")

        if in_pdb_file.endswith(".cif"):
            pocket_center = cif_2_pdb(in_pdb_file, out_pdb_file, return_center=True)
        elif in_pdb_file.endswith(".pdb"):
            pocket_center = center_pdb(in_pdb_file, out_pdb_file, return_center=True)
        else:
            raise ValueError(f"Unsupported pocket file format: {in_pdb_file}")

        # Mirror ligand prep used at generation time so fragment coords match.
        in_sdf_file = test_data.get_ligand_path_from_data_point(test_data[data_idx])
        supplier = Chem.SDMolSupplier(in_sdf_file, removeHs=False, sanitize=False)
        rdmol = supplier[0]
        rdmol = _largest_fragment(rdmol)
        rdmol = Chem.AddHs(rdmol, addCoords=True)
        Chem.Kekulize(rdmol, clearAromaticFlags=True)

        conformer = rdmol.GetConformer()
        for i in range(rdmol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            new_pos = np.array([pos.x, pos.y, pos.z]) - pocket_center
            conformer.SetAtomPosition(i, new_pos)

        fragments = get_fragments_with_brics(rdmol)
        if fragments is None:
            logging.warning(f"No fragments found for {pdb_code}; skipping.")
            continue
        for fragment_type in FRAGMENT_TYPES:
            out_sdf_file = os.path.join(fragments_dir, fragment_type, f"{pdb_code}.sdf")
            try:
                write_fragment(fragments[fragment_type], out_sdf_file)
            except Exception as e:
                # Common failure: kekulization errors on awkward BRICS cuts.
                n_failed[fragment_type] += 1
                logging.error(
                    "Failed to write %s fragment for %s to %s: %s",
                    fragment_type,
                    pdb_code,
                    out_sdf_file,
                    e,
                )
                if os.path.exists(out_sdf_file):
                    os.remove(out_sdf_file)
                continue

            fragments_by_type[fragment_type].append(fragments[fragment_type])
            ligands_by_type[fragment_type].append(rdmol)

    for fragment_type in FRAGMENT_TYPES:
        logging.info(
            "Wrote %d %s fragments to %s (%d failed).",
            len(fragments_by_type[fragment_type]),
            fragment_type,
            os.path.join(fragments_dir, fragment_type),
            n_failed[fragment_type],
        )
        print(
            f"Wrote {len(fragments_by_type[fragment_type])} {fragment_type} fragments."
        )

    if any(fragments_by_type[fragment_type] for fragment_type in FRAGMENT_TYPES):
        plot_fragment_statistics(
            fragments_by_type,
            ligands_by_type,
            os.path.join(fragments_dir, f"fragment_statistics.png"),
        )
    else:
        logging.warning("No fragments were written; skipping statistics plot.")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="SPINDR",
        type=str,
        help="Dataset name (CrossDocked or SPINDR). Default: SPINDR.",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    extract_fragments(args.dataset)
    end_time = datetime.now()
    print(f"Total time: {end_time - start_time}")
