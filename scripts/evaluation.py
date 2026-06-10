"""Evaluate generated molecules."""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import logging

import numpy as np
import py3Dmol
import rdkit
import yaml
from posebusters import PoseBusters
from posecheck import (
    PoseCheck,
)  # Required import; omitting posecheck can cause a segmentation fault.
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Draw,
    Mol,
    MolToSmiles,
    QED,
    rdDepictor,
    SanitizeMol,
    SDMolSupplier,
    SDWriter,
)
from rdkit.Contrib.SA_Score import sascorer

from neat.dataset import DataModule
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils.edm_metrics import compute_edm_metrics_from_tensors
from neat.utils.pose_check_metrics import compute_pose_check_metrics_from_mols
from neat.utils.sbdd_metrics import ClashEvaluator, GninaEvaluator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_MOLECULES_PLOTTED = 100
NUM_MOLECULES_PER_ROW = 5
PLOT_RESOLUTION = 400

ROOT = Path(os.getcwd())
DEFAULT_CONFIG = ROOT / "scripts" / "config_evaluation.yaml"
RESULT_SUBDIR_PREFIXES = ("seed", "prefix", "pocket")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def resolve_config_path(config_file: str | None) -> Path:
    if config_file is not None:
        path = Path(config_file)
        print(f"Using config file: {path}")
        return path
    print(f"Using default config file: {DEFAULT_CONFIG}")
    return DEFAULT_CONFIG


def load_evaluation_config(config_file: str | None) -> dict:
    path = resolve_config_path(config_file)
    with path.open() as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_reference_smiles(params: dict) -> list[str] | None:
    data_root = ROOT / "data"
    datamodule = DataModule(data_root, data_set=params["data_set"].upper())
    datamodule.setup()
    return datamodule.training_data.smiles


def iter_result_subdirs(data_path: Path) -> Iterator[Path]:
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith(RESULT_SUBDIR_PREFIXES):
            yield subdir


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_2d_molecules_visualizations_to_png(subdir: Path, mols: list) -> None:
    subset = mols[:NUM_MOLECULES_PLOTTED]

    for mol in subset:
        if mol is not None:
            rdDepictor.Compute2DCoords(mol)

    img = Draw.MolsToGridImage(
        subset,
        molsPerRow=NUM_MOLECULES_PER_ROW,
        subImgSize=(PLOT_RESOLUTION, PLOT_RESOLUTION),
    )
    img.save(subdir / "generated_molecules_2d.png")

    mols_2d_no_h = []
    for mol in subset:
        if mol is None:
            mols_2d_no_h.append(None)
            continue
        mol_no_h = AllChem.RemoveHs(mol)
        rdDepictor.Compute2DCoords(mol_no_h)
        mols_2d_no_h.append(mol_no_h)

    img_2d = Draw.MolsToGridImage(
        mols_2d_no_h,
        molsPerRow=NUM_MOLECULES_PER_ROW,
        subImgSize=(PLOT_RESOLUTION, PLOT_RESOLUTION),
    )
    img_2d.save(subdir / "generated_molecules_2d_no_h.png")
    print(f"Saved generated molecule images to {subdir}.")


def save_3d_molecules_visualizations_to_html(
    subdir: Path,
    builder: MoleculeBuilder,
    x,
    pos,
    batch,
) -> None:
    view = py3Dmol.view(
        width=NUM_MOLECULES_PER_ROW * PLOT_RESOLUTION,
        height=NUM_MOLECULES_PLOTTED * PLOT_RESOLUTION,
        viewergrid=(NUM_MOLECULES_PLOTTED, NUM_MOLECULES_PER_ROW),
    )

    for i in range(NUM_MOLECULES_PLOTTED):
        row = i // NUM_MOLECULES_PER_ROW
        col = i % NUM_MOLECULES_PER_ROW
        xyz = builder.create_xyz_block(x[batch == i], pos[batch == i])
        view.addModel(xyz, "xyz", viewer=(row, col))
        view.setStyle(
            {"model": -1},
            {"stick": {"radius": 0.2}, "sphere": {"scale": 0.3}},
            viewer=(row, col),
        )

    view.zoomTo()
    html_path = subdir / "generated_molecules_3d.html"
    html_path.write_text(view._make_html())
    print(f"Saved 3D visualization to {html_path}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_validity_uniqueness_novelty(
    mols: list[Mol],
    reference_smiles: list[str] | None = None,
) -> tuple[float, float, float | None]:
    """Return validity, uniqueness, and optional novelty ratios."""

    smiles: list[str] = []
    num_valid = 0

    # Validity computed as number of valid molecules / total number of molecules
    # Valid molecules are those that are not None, can be sanitized, and can be converted to a canonical SMILES string
    for mol in mols:
        if mol is not None:
            mol_copy = copy.deepcopy(mol)
            sanitization_flag = SanitizeMol(mol_copy)
            if int(sanitization_flag) != 0:
                continue
            smile = MolToSmiles(mol, canonical=True)
            if smiles is not None:
                smiles.append(smile)
                num_valid += 1

    p_valid = num_valid / len(mols)

    # Uniqueness computed as number of unique canonical SMILES strings / total number of molecules
    unique_smiles = set[str](smiles)
    p_valid_unique = len(unique_smiles) / len(mols)

    # Novelty computed as number of unique canonical SMILES strings that are not in the reference set / total number of molecules
    if reference_smiles is None:
        return p_valid, p_valid_unique, None

    ref_set = set(reference_smiles)
    num_novel = len(unique_smiles - ref_set)

    p_valid_unique_novel = num_novel / len(mols)

    return p_valid, p_valid_unique, p_valid_unique_novel


def compute_mean_and_95_ci(data: list[float]) -> tuple[float, float]:
    mean = float(np.mean(data))
    std_err = float(np.std(data) / np.sqrt(len(data)))
    return mean, 1.96 * std_err


def canonical_smiles_from_mols(mols: list) -> list[str | None]:
    return [
        MolToSmiles(mol, canonical=True) if mol is not None else None for mol in mols
    ]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def molecule_pipeline_label(params: dict, *, use_bond_predictor: bool) -> str:
    if use_bond_predictor:
        return f"bond predictor ({params['bond_predictor_path']})"
    return "xyz2mol"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_molecules_to_sdf(mols: list, file_path: Path) -> None:
    writer = SDWriter(str(file_path))
    try:
        for mol in mols:
            if mol is None:
                print("Warning: Encountered a 'None' molecule object. Skipping.")
                continue
            writer.write(mol)
    finally:
        writer.close()


def _write_edm_metrics(
    f,
    title: str,
    metrics: EdmMetrics,
) -> None:
    f.write(f"\n{title}:\n")
    f.write(f"Atom stable: {pct(metrics.atom_stability)}\n")
    f.write(f"Molecule stable: {pct(metrics.molecule_stability)}\n")
    f.write(f"Valid: {pct(metrics.valid)}\n")
    f.write(f"Valid x unique: {pct(metrics.valid_x_unique)}\n")


def _write_rdkit_metrics(
    f,
    title: str,
    metrics: RdkitMetrics,
    *,
    include_novelty: bool,
) -> None:
    f.write(f"\n{title}:\n")
    f.write(f"Valid: {pct(metrics.valid)}\n")
    f.write(f"Valid x unique: {pct(metrics.valid_x_unique)}\n")
    if include_novelty and metrics.valid_x_unique_x_novel is not None:
        f.write(f"Valid x unique x novel: {pct(metrics.valid_x_unique_x_novel)}\n")


def _write_dict_metrics(
    f,
    title: str,
    metrics: dict[str, float],
    *,
    as_percent: bool,
    percent_keys: set[str] | None = None,
) -> None:
    percent_keys = percent_keys or set()
    f.write(f"\n{title}:\n")
    for name, value in metrics.items():
        if as_percent or name in percent_keys:
            f.write(f"{name}: {pct(value)}\n")
        else:
            f.write(f"{name}: {value:.2f}\n")


def write_subdir_results(
    subdir: Path,
    params: dict,
    run: SubdirRunResult,
    compute_novelty: bool,
    molecule_pipeline: str,
) -> None:
    with (subdir / "evaluation_results.txt").open("w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")
        f.write(f"Molecule construction: {molecule_pipeline}\n")

        if run.edm is not None:
            _write_edm_metrics(f, "EDM metrics", run.edm)

        if run.rdkit is not None:
            _write_rdkit_metrics(
                f, "RDKit metrics", run.rdkit, include_novelty=compute_novelty
            )

        if run.posebusters is not None:
            _write_dict_metrics(
                f, "PoseBusters metrics", run.posebusters, as_percent=True
            )

        if run.posecheck is not None:
            _write_dict_metrics(f, "PoseCheck metrics", run.posecheck, as_percent=False)

        if run.drugflow is not None:
            _write_dict_metrics(
                f, 
                "DrugFlow metrics", 
                run.drugflow, 
                as_percent=False, 
                percent_keys={"no_clashes"}
            )

        if run.vina is not None:
            _write_dict_metrics(f, "Vina metrics", run.vina, as_percent=False)

        if run.physchem is not None:
            _write_dict_metrics(
                f,
                "Physchem metrics",
                run.physchem,
                as_percent=False,
                percent_keys={"Lipinski"},
            )


def _write_aggregate_edm_metrics(
    f,
    title: str,
    atom_stability: list[float],
    molecule_stability: list[float],
    valid: list[float],
    valid_x_unique: list[float],
) -> None:
    atom_mean, atom_ci = compute_mean_and_95_ci(atom_stability)
    molecule_mean, molecule_ci = compute_mean_and_95_ci(molecule_stability)
    valid_mean, valid_ci = compute_mean_and_95_ci(valid)
    unique_mean, unique_ci = compute_mean_and_95_ci(valid_x_unique)
    f.write(f"\n{title}:\n")
    f.write(f"Atom stable: {pct(atom_mean)} ± {pct(atom_ci)}\n")
    f.write(f"Molecule stable: {pct(molecule_mean)} ± {pct(molecule_ci)}\n")
    f.write(f"Valid: {pct(valid_mean)} ± {pct(valid_ci)}\n")
    f.write(f"Valid x unique: {pct(unique_mean)} ± {pct(unique_ci)}\n")


def _write_aggregate_rdkit_metrics(
    f,
    title: str,
    valid: list[float],
    valid_x_unique: list[float],
    valid_x_unique_x_novel: list[float] | None,
    *,
    include_novelty: bool,
) -> None:
    valid_mean, valid_ci = compute_mean_and_95_ci(valid)
    unique_mean, unique_ci = compute_mean_and_95_ci(valid_x_unique)
    f.write(f"\n{title}:\n")
    f.write(f"Valid: {pct(valid_mean)} ± {pct(valid_ci)}\n")
    f.write(f"Valid x unique: {pct(unique_mean)} ± {pct(unique_ci)}\n")
    if include_novelty and valid_x_unique_x_novel:
        novel_mean, novel_ci = compute_mean_and_95_ci(valid_x_unique_x_novel)
        f.write(f"Valid x unique x novel: {pct(novel_mean)} ± {pct(novel_ci)}\n")


def _write_aggregate_dict_metrics(
    f,
    title: str,
    runs: list[dict[str, float]],
    *,
    as_percent: bool,
    percent_keys: set[str] | None = None,
) -> None:
    if not runs:
        return
    percent_keys = percent_keys or set()
    f.write(f"\n{title}:\n")
    for metric_name in runs[0]:
        values = [run[metric_name] for run in runs]
        mean, ci = compute_mean_and_95_ci(values)
        if as_percent or metric_name in percent_keys:
            f.write(f"{metric_name}: {pct(mean)} ± {pct(ci)}\n")
        else:
            f.write(f"{metric_name}: {mean:.2f} ± {ci:.2f}\n")


def write_summary(
    data_path: Path,
    params: dict,
    aggregate: AggregateResults,
    *,
    compute_edm: bool,
    compute_novelty: bool,
    compute_posebusters: bool,
    compute_posecheck: bool,
    compute_physchem: bool,
    compute_drugflow_clashes: bool,
    compute_vina: bool,
    molecule_pipeline: str,
) -> None:
    with (data_path / "evaluation_summary.txt").open("w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")
        f.write(f"Molecule construction: {molecule_pipeline}\n")

        if compute_edm:
            if aggregate.edm_atom_stability:
                _write_aggregate_edm_metrics(
                    f,
                    "EDM metrics",
                    aggregate.edm_atom_stability,
                    aggregate.edm_molecule_stability,
                    aggregate.edm_valid,
                    aggregate.edm_valid_x_unique,
                )
            else:
                f.write("EDM metrics: No data available\n")

        if aggregate.rdkit_valid:
            _write_aggregate_rdkit_metrics(
                f,
                "RDKit metrics",
                aggregate.rdkit_valid,
                aggregate.rdkit_valid_x_unique,
                aggregate.rdkit_valid_x_unique_x_novel or None,
                include_novelty=compute_novelty,
            )
        else:
            f.write("RDKit metrics: No data available\n")

        if compute_posebusters:
            if aggregate.posebusters:
                _write_aggregate_dict_metrics(
                    f, "PoseBusters metrics", aggregate.posebusters, as_percent=True
                )
            else:
                f.write("PoseBusters metrics: No data available\n")

        if compute_posecheck:
            if aggregate.posecheck:
                _write_aggregate_dict_metrics(
                    f, "PoseCheck metrics", aggregate.posecheck, as_percent=False
                )
            else:
                f.write("PoseCheck metrics: No data available\n")

        if compute_physchem:
            if aggregate.physchem:
                _write_aggregate_dict_metrics(
                    f,
                    "Physchem metrics",
                    aggregate.physchem,
                    as_percent=False,
                    percent_keys={"Lipinski"},
                )
            else:
                f.write("Physchem metrics: No data available\n")

        if compute_drugflow_clashes:
            if aggregate.drugflow:
                _write_aggregate_dict_metrics(
                    f, 
                    "DrugFlow metrics", 
                    aggregate.drugflow, 
                    as_percent=False, 
                    percent_keys={"no_clashes"}
                )
            else:
                f.write("DrugFlow metrics: No data available\n")

        if compute_vina:
            if aggregate.vina:
                _write_aggregate_dict_metrics(
                    f, "Vina metrics", aggregate.vina, as_percent=False
                )
            else:
                f.write("Vina metrics: No data available\n")


# ---------------------------------------------------------------------------
# Per-run and aggregate results
# ---------------------------------------------------------------------------


@dataclass
class EdmMetrics:
    atom_stability: float
    molecule_stability: float
    valid: float
    valid_x_unique: float


@dataclass
class RdkitMetrics:
    valid: float
    valid_x_unique: float
    valid_x_unique_x_novel: float | None = None


@dataclass
class SubdirRunResult:
    edm: EdmMetrics | None = None
    rdkit: RdkitMetrics | None = None
    posebusters: dict[str, float] | None = None
    posecheck: dict[str, float] | None = None
    physchem: dict[str, float] | None = None
    drugflow: dict[str, float] | None = None
    vina: dict[str, float] | None = None


@dataclass
class AggregateResults:
    edm_atom_stability: list[float] = field(default_factory=list)
    edm_molecule_stability: list[float] = field(default_factory=list)
    edm_valid: list[float] = field(default_factory=list)
    edm_valid_x_unique: list[float] = field(default_factory=list)
    posebusters: list[dict[str, float]] = field(default_factory=list)
    posecheck: list[dict[str, float]] = field(default_factory=list)
    physchem: list[dict[str, float]] = field(default_factory=list)
    drugflow: list[dict[str, float]] = field(default_factory=list)
    vina: list[dict[str, float]] = field(default_factory=list)
    rdkit_valid: list[float] = field(default_factory=list)
    rdkit_valid_x_unique: list[float] = field(default_factory=list)
    rdkit_valid_x_unique_x_novel: list[float] = field(default_factory=list)

    def record(self, run: SubdirRunResult) -> None:
        if run.edm is not None:
            self.edm_atom_stability.append(run.edm.atom_stability)
            self.edm_molecule_stability.append(run.edm.molecule_stability)
            self.edm_valid.append(run.edm.valid)
            self.edm_valid_x_unique.append(run.edm.valid_x_unique)

        if run.posebusters is not None:
            self.posebusters.append(run.posebusters)

        if run.posecheck is not None:
            self.posecheck.append(run.posecheck)

        if run.physchem is not None:
            self.physchem.append(run.physchem)

        if run.drugflow is not None:
            self.drugflow.append(run.drugflow)

        if run.vina is not None:
            self.vina.append(run.vina)

        if run.rdkit is not None:
            self.rdkit_valid.append(run.rdkit.valid)
            self.rdkit_valid_x_unique.append(run.rdkit.valid_x_unique)
            if run.rdkit.valid_x_unique_x_novel is not None:
                self.rdkit_valid_x_unique_x_novel.append(
                    run.rdkit.valid_x_unique_x_novel
                )


def _compute_rdkit_metrics(
    mols: list[Mol], reference_smiles: list[str] | None
) -> RdkitMetrics:
    valid, valid_x_unique, valid_x_unique_x_novel = compute_validity_uniqueness_novelty(
        mols, reference_smiles
    )
    return RdkitMetrics(valid, valid_x_unique, valid_x_unique_x_novel)


# ---------------------------------------------------------------------------
# Subdirectory evaluation
# ---------------------------------------------------------------------------


def compute_physchem_properties_from_mols(mols: list) -> dict[str, float] | None:
    mol_weights: list[float] = []
    num_heavy_atoms: list[int] = []
    sa_scores: list[float] = []
    qed_scores: list[float] = []
    lipinski_pass: list[float] = []

    for mol in mols:
        if mol is None:
            continue

        mol_weight = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        lipinski_bool = (
            mol_weight < 500 and logp < 5 and num_h_donors < 5 and num_h_acceptors < 10
        )

        mol_weights.append(mol_weight)
        num_heavy_atoms.append(mol.GetNumHeavyAtoms())
        sa_scores.append(sascorer.calculateScore(mol))
        qed_scores.append(QED.qed(mol))
        lipinski_pass.append(float(lipinski_bool))

    if not sa_scores or not qed_scores or not lipinski_pass:
        return None

    return {
        "MW": float(np.mean(mol_weights)),
        "NHA": float(np.mean(num_heavy_atoms)),
        "SA": float(np.mean(sa_scores)),
        "QED": float(np.mean(qed_scores)),
        "Lipinski": float(np.mean(lipinski_pass)),
    }


def run_posebusters(subdir: Path) -> dict[str, float]:
    buster = PoseBusters(config="mol")
    pred_file = Path(subdir / "generated_mols.sdf")
    cond_file = None
    for file in subdir.iterdir():
        if file.name.endswith(".pdb"):
            cond_file = Path(file)
            break
    if cond_file is not None:
        buster = PoseBusters(config="dock")
    df = buster.bust(
        mol_pred=[pred_file], mol_true=None, mol_cond=cond_file, full_report=False
    )
    df.to_csv(subdir / "posebusters_report.csv", index=False)
    try:
        return {column: df[column].mean() for column in df.columns}
    except:
        return 0


def evaluate_subdirectory(
    subdir: Path,
    params: dict,
    *,
    compute_drugflow_clashes: bool,
    compute_edm: bool,
    compute_novelty: bool,
    compute_posebusters: bool,
    compute_posecheck: bool,
    compute_physchem: bool,
    compute_vina: bool,
    reference_smiles: list[str] | None,
    molecule_pipeline: str,
    use_bond_predictor: bool,
    use_sdf: bool,
) -> SubdirRunResult:

    result = SubdirRunResult()

    if use_sdf:
        supplier = SDMolSupplier(str(subdir / "generated_mols.sdf"))
        mols = [mol for mol in supplier]
    else:
        builder = MoleculeBuilder(vocab=params["data_set"])
        x, pos, batch = builder.load_tensor_from_file(subdir)

        if use_bond_predictor:
            mols = builder.generate_rdkit_molecules_via_bond_predictor(
                x,
                pos,
                batch,
                bond_predictor_path=params["bond_predictor_path"],
                progress_bar=True,
            )

        else:
            mols = builder.generate_rdkit_molecules_via_xyz2mol(
                x, pos, batch, progress_bar=True
            )

        save_molecules_to_sdf(mols, subdir / "generated_mols.sdf")

    result.rdkit = _compute_rdkit_metrics(
        mols, reference_smiles
    )

    if compute_edm:
        atom_stability, mol_stability, edm_valid, edm_unique, _ = (
            compute_edm_metrics_from_tensors(x, pos, batch, params["data_set"].upper())
        )
        result.edm = EdmMetrics(
            atom_stability=atom_stability,
            molecule_stability=mol_stability,
            valid=edm_valid,
            valid_x_unique=edm_valid * edm_unique,
        )

    if compute_posebusters:
        result.posebusters = run_posebusters(subdir)

    if compute_posecheck:
        pocket_path = subdir / "pocket.pdb"
        result.posecheck = compute_pose_check_metrics_from_mols(mols, str(pocket_path))

    if compute_drugflow_clashes:
        clash_evaluator = ClashEvaluator()
        pocket_path = subdir / "pocket.pdb"
        result.drugflow = clash_evaluator.evaluate_mols(mols, str(pocket_path))

    if compute_vina:
        gnina_evaluator = GninaEvaluator()
        pocket_path = subdir / "pocket.pdb"
        mol_sdf_path = subdir / "generated_mols.sdf"
        ligand_sdf_path = subdir / "ligand.sdf"
        vina_results = gnina_evaluator.evaluate_mols(
            mol_sdf_path, str(pocket_path), str(ligand_sdf_path), minimize=True
        )
        vin_min_results = {f"{key}_min": value for key, value in vina_results.items()}
        vina_results = gnina_evaluator.evaluate_mols(
            mol_sdf_path, str(pocket_path), str(ligand_sdf_path), minimize=False
        )
        result.vina = vina_results | vin_min_results

    if compute_physchem:
        result.physchem = compute_physchem_properties_from_mols(mols)

    write_subdir_results(
        subdir,
        params,
        result,
        compute_novelty,
        molecule_pipeline,
    )
    save_2d_molecules_visualizations_to_png(subdir, mols)
    if not use_sdf:
        save_3d_molecules_visualizations_to_html(subdir, builder, x, pos, batch)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def eval_worker(
    subdir,
    params,
    compute_drugflow_clashes,
    compute_edm,
    compute_novelty,
    compute_posebusters,
    compute_posecheck,
    compute_physchem,
    compute_vina,
    reference_smiles,
    molecule_pipeline,
    use_bond_predictor,
    use_sdf,
    log_filename="evaluation.log",
):

    # 1. Disable all standard Python logging for this process
    logging.getLogger().setLevel(logging.CRITICAL)

    # 2. Open the log file
    with open(log_filename, "a") as log_file:
        # Save copies of the original low-level system streams
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            # Redirect system stdout (1) and stderr (2) to the log file descriptor
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)

            # Execute your function in complete stealth mode
            return evaluate_subdirectory(
                subdir,
                params,
                compute_drugflow_clashes=compute_drugflow_clashes,
                compute_edm=compute_edm,
                compute_novelty=compute_novelty,
                compute_posebusters=compute_posebusters,
                compute_posecheck=compute_posecheck,
                compute_physchem=compute_physchem,
                compute_vina=compute_vina,
                molecule_pipeline=molecule_pipeline,
                reference_smiles=reference_smiles,
                use_bond_predictor=use_bond_predictor,
                use_sdf=use_sdf,
            )
        finally:
            # Always restore the low-level streams back to the terminal
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def evaluate(args: argparse.Namespace) -> None:
    params = load_evaluation_config(args.config_file)

    compute_drugflow_clashes = bool(params.get("compute_drugflow_clashes", False))
    compute_edm = bool(params.get("compute_edm", False))
    compute_novelty = bool(params.get("compute_novelty", False))
    compute_posebusters = bool(params.get("compute_posebusters", False))
    compute_posecheck = bool(params.get("compute_posecheck", False))
    compute_physchem = bool(params.get("compute_physchem", False))
    compute_vina = bool(params.get("compute_vina", False))
    use_bond_predictor = params.get("bond_predictor_path") is not None
    use_sdf = bool(params.get("use_sdf", False))
    molecule_pipeline = molecule_pipeline_label(
        params, use_bond_predictor=use_bond_predictor
    )

    if use_sdf:
        compute_edm = False
        print("EDM metrics are not supported for SDF files. Setting compute_edm to False.")

    if compute_novelty:
        reference_smiles = load_reference_smiles(params)
    else:
        reference_smiles = None

    data_path = ROOT / params["data_path"] / params["data_subdir"]
    aggregate = AggregateResults()

    num_workers = params.get("num_workers", 1)
    runs = []

    if num_workers > 1:
        subdirs = list(iter_result_subdirs(data_path))
        worker_func = partial(
            eval_worker,
            params=params,
            compute_drugflow_clashes=compute_drugflow_clashes,
            compute_edm=compute_edm,
            compute_novelty=compute_novelty,
            compute_posebusters=compute_posebusters,
            compute_posecheck=compute_posecheck,
            compute_physchem=compute_physchem,
            compute_vina=compute_vina,
            molecule_pipeline=molecule_pipeline,
            reference_smiles=reference_smiles,
            use_bond_predictor=use_bond_predictor,
            use_sdf=use_sdf,
        )
        runs_dict = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker_func, subdir): i
                for i, subdir in enumerate(subdirs)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating subdirectories",
            ):
                original_index = futures[future]
                runs_dict[original_index] = future.result()

        runs = [runs_dict[i] for i in sorted(runs_dict.keys())]
    
    else:
        for subdir in iter_result_subdirs(data_path):
            print(f"Evaluating {subdir.name}...")
            run = evaluate_subdirectory(
                subdir,
                params,
                compute_drugflow_clashes=compute_drugflow_clashes,
                compute_edm=compute_edm,
                compute_novelty=compute_novelty,
                compute_posebusters=compute_posebusters,
                compute_posecheck=compute_posecheck,
                compute_physchem=compute_physchem,
                compute_vina=compute_vina,
                molecule_pipeline=molecule_pipeline,
                reference_smiles=reference_smiles,
                use_bond_predictor=use_bond_predictor,
                use_sdf=use_sdf,
            )
            runs.append(run)

    for run in runs:
        aggregate.record(run)

    write_summary(
        data_path,
        params,
        aggregate,
        compute_drugflow_clashes=compute_drugflow_clashes,
        compute_edm=compute_edm,
        compute_novelty=compute_novelty,
        compute_posebusters=compute_posebusters,
        compute_posecheck=compute_posecheck,
        compute_physchem=compute_physchem,
        compute_vina=compute_vina,
        molecule_pipeline=molecule_pipeline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated molecules.")
    parser.add_argument(
        "--config",
        dest="config_file",
        required=False,
        metavar="<file>",
        help="Config file for evaluation.",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    evaluate(args)
    print(f"Total evaluation time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
