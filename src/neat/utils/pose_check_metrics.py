import numpy as np
from posecheck import PoseCheck
from posecheck.utils.chem import remove_radicals


def compute_pose_check_metrics(mols_xyz2mol, pocket_path):
    # Initialize the PoseCheck object
    pc = PoseCheck()

    # Load a protein from a PDB file (will run reduce in the background)
    pc.load_protein_from_pdb(pocket_path)

    # Load ligands from an SDF file
    # pc.load_ligands_from_sdf("data/examples/1a2g_ligand.sdf")
    # Alternatively, load RDKit molecules directly
    pc_mols = [remove_radicals(mol) for mol in mols_xyz2mol if mol is not None]
    pc.load_ligands_from_mols(pc_mols)

    # Check for clashes
    clashes = pc.calculate_clashes()
    pose_check_results = {}
    pose_check_results["clashes"] = np.array(clashes).mean()

    # Check for strain
    # strain = pc.calculate_strain_energy()
    # print(f"Strain energy of example molecule: {strain[0]}")

    # Check for interactions
    interactions = pc.calculate_interactions()
    print(f"Interactions of example molecule: {interactions}")
    interaction_types = [
        "HBAcceptor",
        "HBDonor",
        "Hydrophobic",
        "PiStacking",
    ]
    n_lig_atoms = [lig.GetNumAtoms() for lig in mols_xyz2mol if lig is not None]
    for i_type in interaction_types:
        cols = [col for col in interactions.columns if col[2] == i_type]
        i_sum = interactions[cols].sum(axis=1)
        pose_check_results[i_type] = np.array(
            [n_interactions for (n_interactions, n_atoms) in zip(i_sum, n_lig_atoms)]
        ).mean()
    return pose_check_results
