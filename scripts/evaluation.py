import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import py3Dmol
import rdkit
import yaml
from posebusters import PoseBusters
from posecheck import (
    PoseCheck,
)  # Leave import, posecheck will creat segmentation fault otherwise
from rdkit.Chem import AllChem, Draw, MolToSmiles, rdDepictor, SDWriter

from neat.dataset import DataModule
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils.edm_metrics import edm_metrics
from neat.utils.sbdd_metrics import GninaEvalulator
from neat.utils.pose_check_metrics import compute_pose_check_metrics

NUM_MOLECULES_PLOTTED = 100
NUM_MOLECULES_PER_ROW = 5
RESOLUTION = 400
ROOT = os.getcwd()


def compute_validity_uniqueness_novelty(
    smiles: list[str], reference_smiles: list[str] = None
) -> tuple[float, float, float]:
    """Compute validity, uniqueness and novelty ratio of generated molecules.

    Args:
        smiles (List[str]): generated molecules as SMILES strings
        reference_smiles (List[str]): reference canonical SMILES strings for novelty computation

    Returns:
        Tuple[float, float, float]: validity, uniqueness, and novelty ratios
    """
    unique_smiles = set()
    num_valid = 0

    for smile in smiles:
        if smile is None:
            continue
        num_valid += 1
        unique_smiles.add(smile)
    num_unique = len(unique_smiles)

    p_valid = num_valid / len(smiles)
    p_valid_unique = num_unique / len(smiles)
    p_valid_unique_novel = None

    if reference_smiles is not None:
        ref_set = set(reference_smiles)
        num_novel = len(unique_smiles - ref_set)
        p_valid_unique_novel = num_novel / len(smiles)

    return p_valid, p_valid_unique, p_valid_unique_novel


def compute_mean_and_95_ci(data: list[float]) -> tuple[float, float]:
    """Compute mean and 95% confidence interval for a list of data.

    Args:
        data (List[float]): list of data points.

    Returns:
        Tuple[float, float]: mean and 95% confidence interval.
    """
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin_of_error = 1.96 * std_err
    return mean, margin_of_error


def save_molecules_to_sdf(mols: list[rdkit.Chem.Mol], file_path: str) -> None:
    """Save a list of RDKit molecules to an SDF file.

    Args:
        mols (List[rdkit.Chem.Mol]): list of RDKit molecule objects.
        file_path (str): path to save the SDF file.

    Returns:
        None
    """
    writer = rdkit.Chem.SDWriter(file_path)
    for mol in mols:
        if mol is not None:
            try:
                writer.write(mol)
            except Exception:
                continue
    writer.close()


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate generated molecules using various metrics.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    # Load config file
    if args.config_file is not None:
        config_file_path = args.config_file
        print(f"Using config file: {config_file_path}")
    else:
        config_file_path = os.path.join(ROOT, "scripts", "config_evaluation.yaml")
        print(f"Using default config file: {config_file_path}")
    params = yaml.load(
        open(config_file_path, "r"),
        Loader=yaml.FullLoader,
    )

    # Load preprocessed training data for computing novelty
    if params["compute_novelty"]:
        data_root = os.path.join(ROOT, "data")
        datamodule = DataModule(data_root, data_set=params["data_set"].upper())
        datamodule.setup()
        reference_smiles = datamodule.training_data.smiles
    else:
        reference_smiles = None

    # Evaluate generated molecules across all available seeds or prefixes
    edm_atom_stability_lst = []
    edm_molecule_stability_lst = []
    edm_valid_lst = []
    edm_valid_x_unique_lst = []
    xyz2mol_valid_lst = []
    xyz2mol_valid_x_unique_lst = []
    xyz2mol_valid_x_unique_x_novel_lst = []
    bp_valid_lst = []
    bp_valid_x_unique_lst = []
    bp_valid_x_unique_x_novel_lst = []
    posebusters_metrics_list = []
    posecheck_metrics_list = []
    use_bond_predictor = params.get("bond_predictor_path") is not None
    compute_posebusters = bool(params.get("compute_posebusters", False))
    data_path = Path(os.path.join(ROOT, params["data_path"], params["data_subdir"]))

    gnina_evaluator = GninaEvalulator()

    for subdir in data_path.iterdir():
        if subdir.is_dir() and (
            subdir.name.startswith("seed")
            or subdir.name.startswith("prefix")
            or subdir.name.startswith("pocket")
        ):
            subdata_path = os.path.join(data_path, subdir.name)
            builder = MoleculeBuilder(vocab=params["data_set"])
            x, pos, batch = builder.load_tensor_from_file(subdata_path)

            # Compute EDM-based metrics
            (
                atom_stability,
                mol_stability,
                edm_valid,
                edm_unique,
                edm_invalid_idxs,
            ) = edm_metrics(x, pos, batch, params["data_set"].upper())
            edm_atom_stability_lst.append(atom_stability)
            edm_molecule_stability_lst.append(mol_stability)
            edm_valid_lst.append(edm_valid)
            edm_valid_x_unique = edm_valid * edm_unique
            edm_valid_x_unique_lst.append(edm_valid_x_unique)

            # Compute xyz2mol-based metrics
            mols_xyz2mol = builder.generate_rdkit_molecules_via_xyz2mol(
                x, pos, batch, progress_bar=True
            )
            smiles_xyz2mol = [
                MolToSmiles(mol, canonical=True) if mol is not None else None
                for mol in mols_xyz2mol
            ]
            if (
                datamodule.data_set == "CROSSDOCKED"
                and params["data_subdir"] == "conditional"
            ):
                # 1. Initialize the SDWriter with the output file path
                writer = SDWriter(os.path.join(subdata_path, "generated_mols.sdf"))
                try:
                    # 2. Loop through each molecule and write it to the file
                    for mol in mols_xyz2mol:
                        if mol is not None:  # Ensure the molecule object is valid
                            writer.write(mol)
                        else:
                            print(
                                "Warning: Encountered a 'None' molecule object. Skipping."
                            )
                finally:
                    # 3. Always close the writer to ensure all data is flushed and saved properly
                    writer.close()

                pocket_path = os.path.join(subdata_path, "pocket.pdb")
                pose_check_results = compute_pose_check_metrics(
                    mols_xyz2mol, pocket_path
                )
                posecheck_metrics_list.append(pose_check_results)

                # scores = []
                # for mol in mols_xyz2mol:
                #     if mol is not None:
                #         gnina_score = gnina_evaluator.evaluate(mol, protein=pocket_path)
                #         print(f"Gnina score for molecule: {gnina_score}")
                #         scores.append(gnina_score)
                #     else:
                #         print(
                #             "Warning: Encountered a 'None' molecule object. Skipping Gnina evaluation."
                #         )

            xyz2mol_valid, xyz2mol_valid_x_unique, xyz2mol_valid_x_unique_x_novel = (
                compute_validity_uniqueness_novelty(smiles_xyz2mol, reference_smiles)
            )
            xyz2mol_valid_lst.append(xyz2mol_valid)
            xyz2mol_valid_x_unique_lst.append(xyz2mol_valid_x_unique)
            xyz2mol_valid_x_unique_x_novel_lst.append(xyz2mol_valid_x_unique_x_novel)

            # Compute bond predictor-based metrics if bond predictor is available
            if use_bond_predictor:
                mols_bp = builder.generate_rdkit_molecules_via_bond_predictor(
                    x,
                    pos,
                    batch,
                    bond_predictor_path=params["bond_predictor_path"],
                    progress_bar=True,
                )
                smiles_bp = [
                    MolToSmiles(mol, canonical=True) if mol is not None else None
                    for mol in mols_bp
                ]
                (
                    bp_valid,
                    bp_valid_x_unique,
                    bp_valid_x_unique_x_novel,
                ) = compute_validity_uniqueness_novelty(smiles_bp, reference_smiles)
                bp_valid_lst.append(bp_valid)
                bp_valid_x_unique_lst.append(bp_valid_x_unique)
                bp_valid_x_unique_x_novel_lst.append(bp_valid_x_unique_x_novel)

                # Compute PoseBusters metrics for this seed/prefix and save results (optional)
                if compute_posebusters:
                    buster = PoseBusters(config="mol")
                    pred_file = os.path.join(
                        subdata_path, "generated_molecules_bond_predictor.sdf"
                    )
                    save_molecules_to_sdf(mols_xyz2mol, pred_file)
                    df = buster.bust([pred_file], None, None, full_report=False)
                    df.to_csv(
                        os.path.join(
                            subdata_path, "posebusters_report_bond_predictor.csv"
                        ),
                        index=False,
                    )
                    pb_metrics = {}
                    for column in df.columns:
                        pb_metrics[column] = df[column].mean().item()
                    posebusters_metrics_list.append(pb_metrics)

            # Save evaluation results and generated molecule images for this seed/prefix
            with open(os.path.join(subdata_path, "evaluation_results.txt"), "w") as f:
                f.write(f"Data set: {params['data_set']}\n")
                f.write(f"RDKit version: {rdkit.__version__}\n")
                f.write("\nEDM metrics:\n")
                f.write(f"Atom stable: {atom_stability*100:.2f}%\n")
                f.write(f"Molecule stable: {mol_stability*100:.2f}%\n")
                f.write(f"Valid: {edm_valid*100:.2f}%\n")
                f.write(f"Valid x unique: { edm_valid_x_unique * 100:.2f}%\n")
                f.write("\nxyz2mol metrics:\n")
                f.write(f"Valid: {xyz2mol_valid*100:.2f}%\n")
                f.write(f"Valid x unique: {xyz2mol_valid_x_unique * 100:.2f}%\n")
                if params["compute_novelty"]:
                    f.write(
                        f"Valid x unique x novel: { xyz2mol_valid_x_unique_x_novel *100:.2f}%\n"
                    )
                if use_bond_predictor:
                    f.write("\nBond predictor metrics:\n")
                    f.write(f"Valid: {bp_valid*100:.2f}%\n")
                    f.write(f"Valid x unique: {bp_valid_x_unique*100:.2f}%\n")
                    if params["compute_novelty"]:
                        f.write(
                            f"Valid x unique x novel: {bp_valid_x_unique_x_novel*100:.2f}%\n"
                        )
                if compute_posebusters and posebusters_metrics_list:
                    f.write("\nPoseBusters metrics:\n")
                    for metric, value in posebusters_metrics_list[0].items():
                        f.write(f"{metric}: {value*100:.2f}%\n")
                if (
                    datamodule.data_set == "CROSSDOCKED"
                    and params["data_subdir"] == "conditional"
                ):
                    f.write("\nPoseCheck metrics:\n")
                    for metric, value in pose_check_results.items():
                        f.write(f"{metric}: {value:.2f}\n")

            mols_plotting_subset = mols_xyz2mol[:NUM_MOLECULES_PLOTTED]
            for mol in mols_plotting_subset:
                if mol is not None:
                    # Optimize 2D coordinates for better visualization
                    # This operation is in place
                    rdDepictor.Compute2DCoords(mol)
            img = Draw.MolsToGridImage(
                mols_plotting_subset,
                molsPerRow=NUM_MOLECULES_PER_ROW,
                subImgSize=(RESOLUTION, RESOLUTION),
            )
            img.save(os.path.join(subdata_path, "generated_molecules.png"))

            mols_2d = []
            for mol in mols_plotting_subset:
                if mol is None:
                    mols_2d.append(None)
                else:
                    mol = AllChem.RemoveHs(mol)
                    rdDepictor.Compute2DCoords(mol)
                    mols_2d.append(mol)

            img = Draw.MolsToGridImage(mols_2d, molsPerRow=5, subImgSize=(400, 400))
            img.save(os.path.join(subdata_path, "generated_molecules_2d.png"))

            print(f"Saved generated molecules images to {os.path.join(subdata_path)}.")

            view = py3Dmol.view(
                width=NUM_MOLECULES_PER_ROW * RESOLUTION,
                height=NUM_MOLECULES_PLOTTED * RESOLUTION,
                viewergrid=(NUM_MOLECULES_PLOTTED, NUM_MOLECULES_PER_ROW),
            )

            for i in range(NUM_MOLECULES_PLOTTED):
                row = i // NUM_MOLECULES_PER_ROW
                col = i % NUM_MOLECULES_PER_ROW

                x_sub = x[batch == i]
                pos_sub = pos[batch == i]

                xyz = builder.create_xyz_block(x_sub, pos_sub)

                view.addModel(xyz, "xyz", viewer=(row, col))
                view.setStyle(
                    {"model": -1},
                    {"stick": {"radius": 0.2}, "sphere": {"scale": 0.3}},
                    viewer=(row, col),
                )

            view.zoomTo()

            with open(
                os.path.join(subdata_path, "generated_molecules_3d.html"), "w"
            ) as f:
                f.write(view._make_html())

            print(
                f"Saved 3D visualization to {os.path.join(subdata_path, 'generated_molecules_3d.html')}"
            )

    # Compute overall mean and confidence intervals across all seeds/prefixes and save summary
    atom_stability_mean, atom_stability_ci = compute_mean_and_95_ci(
        edm_atom_stability_lst
    )
    molecule_stability_mean, molecule_stability_ci = compute_mean_and_95_ci(
        edm_molecule_stability_lst
    )
    lookup_valid_mean, lookup_valid_ci = compute_mean_and_95_ci(edm_valid_lst)
    lookup_valid_x_unique_mean, lookup_valid_x_unique_ci = compute_mean_and_95_ci(
        edm_valid_x_unique_lst
    )
    xyz2mol_valid_mean, xyz2mol_valid_ci = compute_mean_and_95_ci(xyz2mol_valid_lst)
    xyz2mol_valid_x_unique_mean, xyz2mol_valid_x_unique_ci = compute_mean_and_95_ci(
        xyz2mol_valid_x_unique_lst
    )
    if params["compute_novelty"]:
        xyz2mol_valid_x_unique_x_novel_mean, xyz2mol_valid_x_unique_x_novel_ci = (
            compute_mean_and_95_ci(xyz2mol_valid_x_unique_x_novel_lst)
        )
    if use_bond_predictor:
        bond_predictor_valid_mean, bond_predictor_valid_ci = compute_mean_and_95_ci(
            bp_valid_lst
        )
        bond_predictor_valid_x_unique_mean, bond_predictor_valid_x_unique_ci = (
            compute_mean_and_95_ci(bp_valid_x_unique_lst)
        )
        if params["compute_novelty"]:
            (
                bond_predictor_valid_x_unique_x_novel_mean,
                bond_predictor_valid_x_unique_x_novel_ci,
            ) = compute_mean_and_95_ci(bp_valid_x_unique_x_novel_lst)

    with open(os.path.join(data_path, "evaluation_summary.txt"), "w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")
        f.write("\nEDM metrics:\n")
        f.write(
            f"Atom stable: {atom_stability_mean*100:.2f}% ± {atom_stability_ci*100:.2f}%\n"
        )
        f.write(
            f"Molecule stable: {molecule_stability_mean*100:.2f}% ± {molecule_stability_ci*100:.2f}%\n"
        )
        f.write(f"Valid: {lookup_valid_mean*100:.2f}% ± {lookup_valid_ci*100:.2f}%\n")
        f.write(
            f"Valid x unique: {lookup_valid_x_unique_mean*100:.2f}% ± {lookup_valid_x_unique_ci*100:.2f}%\n"
        )
        f.write("\nxyz2mol metrics:\n")
        f.write(f"Valid: {xyz2mol_valid_mean*100:.2f}% ± {xyz2mol_valid_ci*100:.2f}%\n")
        f.write(
            f"Valid x unique: {xyz2mol_valid_x_unique_mean*100:.2f}% ± {xyz2mol_valid_x_unique_ci*100:.2f}%\n"
        )
        if params["compute_novelty"]:
            f.write(
                f"Valid x unique x novel: {xyz2mol_valid_x_unique_x_novel_mean*100:.2f}% ± {xyz2mol_valid_x_unique_x_novel_ci*100:.2f}%\n"
            )
        if use_bond_predictor:
            f.write("\nBond predictor metrics:\n")
            f.write(
                f"Valid: {bond_predictor_valid_mean*100:.2f}% ± {bond_predictor_valid_ci*100:.2f}%\n"
            )
            f.write(
                f"Valid x unique: {bond_predictor_valid_x_unique_mean*100:.2f}% ± {bond_predictor_valid_x_unique_ci*100:.2f}%\n"
            )
            if params["compute_novelty"]:
                f.write(
                    f"Valid x unique x novel: {bond_predictor_valid_x_unique_x_novel_mean*100:.2f}% ± {bond_predictor_valid_x_unique_x_novel_ci*100:.2f}%\n"
                )
            if compute_posebusters and posebusters_metrics_list:
                f.write("\nPoseBusters metrics:\n")
                metric_names = list(posebusters_metrics_list[0].keys())
                for metric_name in metric_names:
                    values = [
                        metrics[metric_name] for metrics in posebusters_metrics_list
                    ]
                    mean, ci = compute_mean_and_95_ci(values)
                    f.write(f"{metric_name}: {mean*100:.2f}% ± {ci*100:.2f}%\n")
        if (
            datamodule.data_set == "CROSSDOCKED"
            and params["data_subdir"] == "conditional"
        ):
            f.write("\nPoseCheck metrics:\n")
            metric_names = list(posecheck_metrics_list[0].keys())
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in posecheck_metrics_list]
                mean, ci = compute_mean_and_95_ci(values)
                f.write(f"{metric_name}: {mean:.2f} ± {ci:.2f}\n")


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for evaluation.",
    )

    args = parser.parse_args()

    evaluate(args)

    end_time = datetime.now()
    print(f"Total evaluation time: {end_time - start_time}")
