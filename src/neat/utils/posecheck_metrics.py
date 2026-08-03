import numpy as np
from posecheck import PoseCheck
from rdkit.Chem import Mol
from posecheck.utils.chem import remove_radicals


def compute_posecheck_metrics_from_mols(mols: list[Mol], pocket_path: str) -> dict[str, float]:
    # Initialize the PoseCheck object
    pc = PoseCheck()

    # Load a protein from a PDB file (will run reduce in the background)
    pc.load_protein_from_pdb(pocket_path)

    # Load ligands from RDKitmolecule objects
    pc_mols = [remove_radicals(mol) for mol in mols if mol is not None]
    pc.load_ligands_from_mols(pc_mols)

    # Calculate clashes
    clashes = pc.calculate_clashes()
    pose_check_results = {}

    # Calculate strain energies
    strain = pc.calculate_strain_energy()

    # Calculate interactions
    interactions = pc.calculate_interactions()

    # Calculate the mean number of clashes, strain energy, and interactions per ligand
    # Remove potential None values before calculating the mean
    clashes = [c for c in clashes if c is not None]
    pose_check_results["clashes"] = np.array(clashes).mean()

    strain = [s for s in strain if s is not None]
    pose_check_results["strain"] = np.array(strain).mean()

    interaction_types = [
        "HBAcceptor",
        "HBDonor",
        "Hydrophobic",
        "VdWContact",
    ]
    for i_type in interaction_types:
        cols = [col for col in interactions.columns if col[2] == i_type]
        i_sum = interactions[cols].sum(axis=1)
        pose_check_results[i_type] = float(i_sum.mean())

    return pose_check_results
