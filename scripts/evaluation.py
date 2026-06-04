"""Evaluate generated molecules from NEAT model outputs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import py3Dmol
import rdkit
import yaml
from posebusters import PoseBusters
from posecheck import (
    PoseCheck,
)  # Required import; omitting posecheck can cause a segmentation fault.
from rdkit.Chem import AllChem, Draw, FindPotentialStereo, MolToSmiles, rdDepictor, SDWriter

from neat.dataset import DataModule
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils.edm_metrics import edm_metrics
from neat.utils.pose_check_metrics import compute_pose_check_metrics

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


def is_conditional_crossdocked(params: dict) -> bool:
    return (
        params["data_set"].upper() == "CROSSDOCKED"
        and params["data_subdir"] == "conditional"
    )


def iter_result_subdirs(data_path: Path) -> Iterator[Path]:
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith(RESULT_SUBDIR_PREFIXES):
            yield subdir


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_validity_uniqueness_novelty(
    smiles: list[str | None],
    reference_smiles: list[str] | None = None,
) -> tuple[float, float, float | None]:
    """Return validity, uniqueness, and optional novelty ratios."""
    unique_smiles: set[str] = set()
    num_valid = 0

    for smile in smiles:
        if smile is None:
            continue
        num_valid += 1
        unique_smiles.add(smile)

    p_valid = num_valid / len(smiles)
    p_valid_unique = len(unique_smiles) / len(smiles)

    if reference_smiles is None:
        return p_valid, p_valid_unique, None

    ref_set = set(reference_smiles)
    num_novel = len(unique_smiles - ref_set)
    return p_valid, p_valid_unique, num_novel / len(smiles)


def compute_mean_and_95_ci(data: list[float]) -> tuple[float, float]:
    mean = float(np.mean(data))
    std_err = float(np.std(data) / np.sqrt(len(data)))
    return mean, 1.96 * std_err


def smiles_from_mols(mols: list) -> list[str | None]:
    return [
        MolToSmiles(mol, canonical=True) if mol is not None else None for mol in mols
    ]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_molecules_to_sdf(mols: list, file_path: Path) -> None:
    writer = SDWriter(str(file_path))
    try:
        for mol in mols:
            if mol is None:
                continue
            try:
                writer.write(mol)
            except Exception:
                continue
    finally:
        writer.close()


def write_xyz2mol_sdf(mols: list, file_path: Path) -> None:
    writer = SDWriter(str(file_path))
    try:
        for mol in mols:
            if mol is None:
                print("Warning: Encountered a 'None' molecule object. Skipping.")
                continue
            writer.write(mol)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Per-run and aggregate results
# ---------------------------------------------------------------------------


@dataclass
class ValidityMetrics:
    valid: float
    valid_x_unique: float
    valid_x_unique_x_novel: float | None = None


@dataclass
class EdmMetrics:
    atom_stability: float
    molecule_stability: float
    valid: float
    valid_x_unique: float


@dataclass
class SubdirRunResult:
    edm: EdmMetrics | None = None
    xyz2mol: ValidityMetrics | None = None
    bond_predictor: ValidityMetrics | None = None
    posebusters: dict[str, float] | None = None
    posecheck: dict[str, float] | None = None


@dataclass
class AggregateResults:
    edm_atom_stability: list[float] = field(default_factory=list)
    edm_molecule_stability: list[float] = field(default_factory=list)
    edm_valid: list[float] = field(default_factory=list)
    edm_valid_x_unique: list[float] = field(default_factory=list)
    xyz2mol_valid: list[float] = field(default_factory=list)
    xyz2mol_valid_x_unique: list[float] = field(default_factory=list)
    xyz2mol_valid_x_unique_x_novel: list[float] = field(default_factory=list)
    bp_valid: list[float] = field(default_factory=list)
    bp_valid_x_unique: list[float] = field(default_factory=list)
    bp_valid_x_unique_x_novel: list[float] = field(default_factory=list)
    posebusters: list[dict[str, float]] = field(default_factory=list)
    posecheck: list[dict[str, float]] = field(default_factory=list)

    def record(self, run: SubdirRunResult) -> None:
        if run.edm is not None:
            self.edm_atom_stability.append(run.edm.atom_stability)
            self.edm_molecule_stability.append(run.edm.molecule_stability)
            self.edm_valid.append(run.edm.valid)
            self.edm_valid_x_unique.append(run.edm.valid_x_unique)
        if run.xyz2mol is not None:
            self.xyz2mol_valid.append(run.xyz2mol.valid)
            self.xyz2mol_valid_x_unique.append(run.xyz2mol.valid_x_unique)
            if run.xyz2mol.valid_x_unique_x_novel is not None:
                self.xyz2mol_valid_x_unique_x_novel.append(
                    run.xyz2mol.valid_x_unique_x_novel
                )
        if run.bond_predictor is not None:
            self.bp_valid.append(run.bond_predictor.valid)
            self.bp_valid_x_unique.append(run.bond_predictor.valid_x_unique)
            if run.bond_predictor.valid_x_unique_x_novel is not None:
                self.bp_valid_x_unique_x_novel.append(
                    run.bond_predictor.valid_x_unique_x_novel
                )
        if run.posebusters is not None:
            self.posebusters.append(run.posebusters)
        if run.posecheck is not None:
            self.posecheck.append(run.posecheck)


def _validity_metrics_from_smiles(
    smiles: list[str | None], reference_smiles: list[str] | None
) -> ValidityMetrics:
    valid, valid_x_unique, valid_x_unique_x_novel = compute_validity_uniqueness_novelty(
        smiles, reference_smiles
    )
    return ValidityMetrics(valid, valid_x_unique, valid_x_unique_x_novel)


def _write_validity_section(
    f,
    title: str,
    metrics: ValidityMetrics,
    *,
    include_novelty: bool,
) -> None:
    f.write(f"\n{title}:\n")
    f.write(f"Valid: {pct(metrics.valid)}\n")
    f.write(f"Valid x unique: {pct(metrics.valid_x_unique)}\n")
    if include_novelty and metrics.valid_x_unique_x_novel is not None:
        f.write(f"Valid x unique x novel: {pct(metrics.valid_x_unique_x_novel)}\n")


def _write_dict_metrics(
    f, title: str, metrics: dict[str, float], *, as_percent: bool
) -> None:
    f.write(f"\n{title}:\n")
    for name, value in metrics.items():
        if as_percent:
            f.write(f"{name}: {pct(value)}\n")
        else:
            f.write(f"{name}: {value:.2f}\n")


def write_subdir_results(
    subdir: Path,
    params: dict,
    run: SubdirRunResult,
    compute_novelty: bool,
) -> None:
    with (subdir / "evaluation_results.txt").open("w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")

        if run.edm is not None:
            f.write("\nEDM metrics:\n")
            f.write(f"Atom stable: {pct(run.edm.atom_stability)}\n")
            f.write(f"Molecule stable: {pct(run.edm.molecule_stability)}\n")
            f.write(f"Valid: {pct(run.edm.valid)}\n")
            f.write(f"Valid x unique: {pct(run.edm.valid_x_unique)}\n")

        if run.xyz2mol is not None:
            _write_validity_section(
                f, "xyz2mol metrics", run.xyz2mol, include_novelty=compute_novelty
            )

        if run.bond_predictor is not None:
            _write_validity_section(
                f,
                "Bond predictor metrics",
                run.bond_predictor,
                include_novelty=compute_novelty,
            )

        if run.posebusters is not None:
            _write_dict_metrics(f, "PoseBusters metrics", run.posebusters, as_percent=True)

        if run.posecheck is not None:
            _write_dict_metrics(f, "PoseCheck metrics", run.posecheck, as_percent=False)


def _write_aggregate_validity(
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
        f.write(
            f"Valid x unique x novel: {pct(novel_mean)} ± {pct(novel_ci)}\n"
        )


def _write_aggregate_dict_metrics(
    f,
    title: str,
    runs: list[dict[str, float]],
    *,
    as_percent: bool,
) -> None:
    if not runs:
        return
    f.write(f"\n{title}:\n")
    for metric_name in runs[0]:
        values = [run[metric_name] for run in runs]
        mean, ci = compute_mean_and_95_ci(values)
        if as_percent:
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
    run_posecheck: bool,
    use_bond_predictor: bool,
) -> None:
    with (data_path / "evaluation_summary.txt").open("w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")

        if compute_edm and aggregate.edm_valid:
            atom_mean, atom_ci = compute_mean_and_95_ci(aggregate.edm_atom_stability)
            mol_mean, mol_ci = compute_mean_and_95_ci(aggregate.edm_molecule_stability)
            valid_mean, valid_ci = compute_mean_and_95_ci(aggregate.edm_valid)
            unique_mean, unique_ci = compute_mean_and_95_ci(aggregate.edm_valid_x_unique)
            f.write("\nEDM metrics:\n")
            f.write(f"Atom stable: {pct(atom_mean)} ± {pct(atom_ci)}\n")
            f.write(f"Molecule stable: {pct(mol_mean)} ± {pct(mol_ci)}\n")
            f.write(f"Valid: {pct(valid_mean)} ± {pct(valid_ci)}\n")
            f.write(f"Valid x unique: {pct(unique_mean)} ± {pct(unique_ci)}\n")

        _write_aggregate_validity(
            f,
            "xyz2mol metrics",
            aggregate.xyz2mol_valid,
            aggregate.xyz2mol_valid_x_unique,
            aggregate.xyz2mol_valid_x_unique_x_novel or None,
            include_novelty=compute_novelty,
        )

        if use_bond_predictor and aggregate.bp_valid:
            _write_aggregate_validity(
                f,
                "Bond predictor metrics",
                aggregate.bp_valid,
                aggregate.bp_valid_x_unique,
                aggregate.bp_valid_x_unique_x_novel or None,
                include_novelty=compute_novelty,
            )

        if compute_posebusters:
            _write_aggregate_dict_metrics(
                f, "PoseBusters metrics", aggregate.posebusters, as_percent=True
            )

        if run_posecheck:
            _write_aggregate_dict_metrics(
                f, "PoseCheck metrics", aggregate.posecheck, as_percent=False
            )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_molecule_plots(subdir: Path, mols: list) -> None:
    subset = mols[:NUM_MOLECULES_PLOTTED]

    for mol in subset:
        if mol is not None:
            rdDepictor.Compute2DCoords(mol)

    img = Draw.MolsToGridImage(
        subset,
        molsPerRow=NUM_MOLECULES_PER_ROW,
        subImgSize=(PLOT_RESOLUTION, PLOT_RESOLUTION),
    )
    img.save(subdir / "generated_molecules.png")

    mols_2d = []
    for mol in subset:
        if mol is None:
            mols_2d.append(None)
            continue
        mol_h = AllChem.RemoveHs(mol)
        rdDepictor.Compute2DCoords(mol_h)
        mols_2d.append(mol_h)

    img_2d = Draw.MolsToGridImage(
        mols_2d,
        molsPerRow=NUM_MOLECULES_PER_ROW,
        subImgSize=(PLOT_RESOLUTION, PLOT_RESOLUTION),
    )
    img_2d.save(subdir / "generated_molecules_2d.png")
    print(f"Saved generated molecule images to {subdir}.")


def save_3d_html(
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
# Per-subdirectory evaluation
# ---------------------------------------------------------------------------


def run_posebusters(subdir: Path, mols: list) -> dict[str, float]:
    buster = PoseBusters(config="mol")
    pred_file = subdir / "generated_molecules_bond_predictor.sdf"
    save_molecules_to_sdf(mols, pred_file)
    df = buster.bust([str(pred_file)], None, None, full_report=False)
    df.to_csv(subdir / "posebusters_report_bond_predictor.csv", index=False)
    return {column: df[column].mean().item() for column in df.columns}


def evaluate_subdirectory(
    subdir: Path,
    params: dict,
    reference_smiles: list[str] | None,
    *,
    compute_edm: bool,
    compute_novelty: bool,
    compute_posebusters: bool,
    run_posecheck: bool,
    use_bond_predictor: bool,
) -> SubdirRunResult:

    builder = MoleculeBuilder(vocab=params["data_set"])
    x, pos, batch = builder.load_tensor_from_file(subdir)
    result = SubdirRunResult()

    mols_xyz2mol = builder.generate_rdkit_molecules_via_xyz2mol(
        x, pos, batch, progress_bar=True
    )
    result.xyz2mol = _validity_metrics_from_smiles(
        smiles_from_mols(mols_xyz2mol), reference_smiles
    )
    write_xyz2mol_sdf(mols_xyz2mol, subdir / "generated_mols.sdf")

    if compute_edm:
        atom_stability, mol_stability, edm_valid, edm_unique, _ = edm_metrics(
            x, pos, batch, params["data_set"].upper()
        )
        result.edm = EdmMetrics(
            atom_stability=atom_stability,
            molecule_stability=mol_stability,
            valid=edm_valid,
            valid_x_unique=edm_valid * edm_unique,
        )

    if compute_posebusters:
        result.posebusters = run_posebusters(subdir, mols_xyz2mol)

    if run_posecheck:
        pocket_path = subdir / "pocket.pdb"
        result.posecheck = compute_pose_check_metrics(mols_xyz2mol, str(pocket_path))

    if use_bond_predictor:
        mols_bp = builder.generate_rdkit_molecules_via_bond_predictor(
            x,
            pos,
            batch,
            bond_predictor_path=params["bond_predictor_path"],
            progress_bar=True,
        )
        result.bond_predictor = _validity_metrics_from_smiles(
            smiles_from_mols(mols_bp), reference_smiles
        )

    write_subdir_results(
        subdir,
        params,
        result,
        compute_novelty,
    )
    save_molecule_plots(subdir, mols_xyz2mol)
    save_3d_html(subdir, builder, x, pos, batch)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    params = load_evaluation_config(args.config_file)

    compute_edm = bool(params.get("compute_edm", False))
    compute_novelty = bool(params.get("compute_novelty", False))
    compute_posebusters = bool(params.get("compute_posebusters", False))
    run_posecheck = is_conditional_crossdocked(params)
    use_bond_predictor = params.get("bond_predictor_path") is not None

    if compute_novelty:
        reference_smiles = load_reference_smiles(params)
    else:
        reference_smiles = None

    data_path = ROOT / params["data_path"] / params["data_subdir"]
    aggregate = AggregateResults()

    for subdir in iter_result_subdirs(data_path):
        print(f"Evaluating {subdir.name}...")
        run = evaluate_subdirectory(
            subdir,
            params,
            reference_smiles,
            compute_edm=compute_edm,
            compute_novelty=compute_novelty,
            compute_posebusters=compute_posebusters,
            run_posecheck=run_posecheck,
            use_bond_predictor=use_bond_predictor,
        )
        aggregate.record(run)

    write_summary(
        data_path,
        params,
        aggregate,
        compute_edm=compute_edm,
        compute_novelty=compute_novelty,
        compute_posebusters=compute_posebusters,
        run_posecheck=run_posecheck,
        use_bond_predictor=use_bond_predictor,
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
