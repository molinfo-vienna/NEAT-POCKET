"""Evaluate generated molecules from NEAT (and related) generation runs.

Workflow:
  1. Load scripts/config_evaluation.yaml (or --config).
  2. Iterate result subdirectories under data_path/data_subdir whose names
     start with seed_, prefix_, or pocket_ (see RESULT_SUBDIR_PREFIXES).
  3. For each subdirectory: build RDKit mols from generated_mols.pt (or load 
     from .sdf), compute selected metrics, write evaluation_results.txt and
     visualization images.
  4. Aggregate per-subdir scores into evaluation_summary.txt (mean ± 95% CI).

Subdirectories missing generated_mols.pt / generated_mols.sdf (e.g. pockets
skipped during fragment-conditioned generation) are skipped rather than
raising.

Metric flags in the config (compute_edm, compute_posebusters, …) control
which scores are computed. Molecule construction uses xyz2mol by default,
or a bond predictor if bond_predictor_path is set.
"""

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
from rdkit import Chem
from tqdm import tqdm
import logging
import tempfile


import numpy as np
import py3Dmol
import rdkit
import yaml
from scipy import stats
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
)
from rdkit.Contrib.SA_Score import sascorer

from neat.dataset import DataModule
from neat.model.molecule_builder import MoleculeBuilder
from neat.utils import save_molecules_to_sdf
from neat.utils.edm_metrics import compute_edm_metrics_from_tensors
from neat.utils.posecheck_metrics import compute_posecheck_metrics_from_mols
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
    """Return the evaluation config path, defaulting to config_evaluation.yaml."""
    if config_file is not None:
        path = Path(config_file)
        print(f"Using config file: {path}")
        return path
    print(f"Using default config file: {DEFAULT_CONFIG}")
    return DEFAULT_CONFIG


def load_evaluation_config(config_file: str | None) -> dict:
    """Load and parse the evaluation YAML config."""
    path = resolve_config_path(config_file)
    with path.open() as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_reference_smiles(params: dict) -> list[str] | None:
    """Load training-set SMILES used as the novelty reference distribution."""
    data_root = ROOT / "data"
    datamodule = DataModule(data_root, data_set=params["data_set"].upper())
    datamodule.setup()
    return datamodule.training_data.smiles


def iter_result_subdirs(data_path: Path) -> Iterator[Path]:
    """Yield seed_/prefix_/pocket_ result directories under data_path, sorted."""
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith(RESULT_SUBDIR_PREFIXES):
            yield subdir


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_2d_molecules_visualizations_to_png(
    subdir: Path, 
    mols: list
) -> None:
    """Save 2D grid images (with and without hydrogens) for the first N mols."""
    subset = mols[:NUM_MOLECULES_PLOTTED]

    for mol in subset:
        if mol is not None:
            try:
                rdDepictor.Compute2DCoords(mol)
            except Exception as e:
                print(f"Warning: Failed to compute 2D coordinates for a molecule: {e}")
                # If 2D coordinates computation fails, we can skip this molecule or handle it as needed.
                continue

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
        try:
            mol_no_h = AllChem.RemoveHs(mol)
            rdDepictor.Compute2DCoords(mol_no_h)
        except Exception as e:
            print(f"Warning: Failed to compute 2D coordinates for a molecule: {e}")
            continue
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
    """Save an interactive py3Dmol HTML grid of the first N generated molecules."""
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
) -> tuple[float, float, float | None, list[bool | None]]:
    """Compute RDKit validity / uniqueness / novelty fractions.

    Validity: sanitizable mols with a canonical SMILES, over all mols.
    Uniqueness: unique valid canonical SMILES, over all mols.
    Novelty (optional): unique valid SMILES absent from reference_smiles,
    over all mols.

    Args:
        mols: RDKit molecules (None entries count as invalid).
        reference_smiles: Training SMILES for novelty; None skips novelty.

    Returns:
        (valid, valid_x_unique, valid_x_unique_x_novel_or_None, validity_flags)
        where validity_flags[i] is True iff mols[i] passed validity checks.
    """

    smiles: list[str] = []
    num_valid = 0
    total_mols = len(mols)

    # Validity computed as number of valid molecules / total number of molecules
    # Valid molecules are those that are not None, can be sanitized, and can be converted to a canonical SMILES string
    
    validity_flag_list = [False] * total_mols
    for i, mol in enumerate(mols):
        if mol is not None:
            mol_copy = copy.deepcopy(mol)
            sanitization_flag = SanitizeMol(mol_copy, catchErrors=True)
            if int(sanitization_flag) != 0:
                continue
            smile = MolToSmiles(mol, canonical=True)
            if smiles is not None:
                smiles.append(smile)
                num_valid += 1
                validity_flag_list[i] = True

    p_valid = num_valid / total_mols

    # Uniqueness computed as number of unique canonical SMILES strings / total number of molecules
    unique_smiles = set[str](smiles)
    num_unique = len(unique_smiles)
    p_valid_unique = num_unique / total_mols

    # Novelty computed as number of unique canonical SMILES strings that are not in the reference set / number of valid molecules
    # Validity x uniqueness x novelty computed as number of unique canonical SMILES strings that are not in the reference set / total number of molecules
    if reference_smiles is None:
        return p_valid, p_valid_unique, None, validity_flag_list

    ref_set = set(reference_smiles)
    num_novel = len(unique_smiles - ref_set)

    p_valid_unique_novel = num_novel / total_mols

    return p_valid, p_valid_unique, p_valid_unique_novel, validity_flag_list


def compute_mean_and_95_ci(data: list[float]) -> tuple[float, float]:
    """Return (mean, half-width of a 95% t-interval) over scalar metric values.

    Uses the sample standard deviation (ddof=1) and the Student-t critical
    value with ``len(data) - 1`` degrees of freedom. For a single observation
    the half-width is returned as 0 (CI undefined).
    """
    n = len(data)
    mean = float(np.mean(data))
    if n < 2:
        return mean, 0.0
    std_err = float(np.std(data, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, t_crit * std_err


def canonical_smiles_from_mols(mols: list) -> list[str | None]:
    """Convert molecules to canonical SMILES (None stays None)."""
    return [
        MolToSmiles(mol, canonical=True) if mol is not None else None for mol in mols
    ]


def pct(value: float) -> str:
    """Format a fraction in [0, 1] as a percentage string with two decimals."""
    return f"{value * 100:.2f}%"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _write_edm_metrics(
    f,
    title: str,
    metrics: EdmMetrics,
) -> None:
    """Write a single-run EDM metric block to an open text file."""
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
    """Write a single-run RDKit validity/uniqueness(/novelty) block."""
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
    """Write a dict of named scalar metrics; optionally format as percentages."""
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
) -> None:
    """Write evaluation_results.txt for one seed/prefix/pocket subdirectory."""
    with (subdir / "evaluation_results.txt").open("w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")

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

        if run.rdkit_x_posebusters is not None:
            f.write(f"\nRDKit x PoseBusters: {pct(run.rdkit_x_posebusters)}\n")

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
    """Write mean ± 95% CI for EDM metrics across subdirectories."""
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
    """Write mean ± 95% CI for RDKit metrics across subdirectories."""
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
    """Write mean ± 95% CI for each key shared across per-subdir metric dicts."""
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
) -> None:
    """Write evaluation_summary.txt aggregating all subdirectory runs."""
    with (data_path / "evaluation_summary.txt").open("w") as f:
        f.write(f"Data set: {params['data_set']}\n")
        f.write(f"RDKit version: {rdkit.__version__}\n")

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

        if aggregate.rdkit_x_posebusters:
            mean, ci = compute_mean_and_95_ci(aggregate.rdkit_x_posebusters)
            f.write(f"\nRDKit x PoseBusters: {pct(mean)} ± {pct(ci)}\n")

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
    """EDM atom/molecule stability and validity metrics for one subdirectory."""

    atom_stability: float
    molecule_stability: float
    valid: float
    valid_x_unique: float


@dataclass
class RdkitMetrics:
    """RDKit validity / uniqueness / optional novelty for one subdirectory."""

    valid: float
    valid_x_unique: float
    valid_x_unique_x_novel: float | None = None


@dataclass
class SubdirRunResult:
    """All metric blocks computed for a single seed/prefix/pocket directory."""

    edm: EdmMetrics | None = None
    rdkit: RdkitMetrics | None = None
    posebusters: dict[str, float] | None = None
    rdkit_x_posebusters: float | None = None
    posecheck: dict[str, float] | None = None
    physchem: dict[str, float] | None = None
    drugflow: dict[str, float] | None = None
    vina: dict[str, float] | None = None


@dataclass
class AggregateResults:
    """Accumulates per-subdirectory metric values for the summary report."""

    edm_atom_stability: list[float] = field(default_factory=list)
    edm_molecule_stability: list[float] = field(default_factory=list)
    edm_valid: list[float] = field(default_factory=list)
    edm_valid_x_unique: list[float] = field(default_factory=list)
    posebusters: list[dict[str, float]] = field(default_factory=list)
    rdkit_x_posebusters: list[float] = field(default_factory=list)
    posecheck: list[dict[str, float]] = field(default_factory=list)
    physchem: list[dict[str, float]] = field(default_factory=list)
    drugflow: list[dict[str, float]] = field(default_factory=list)
    vina: list[dict[str, float]] = field(default_factory=list)
    rdkit_valid: list[float] = field(default_factory=list)
    rdkit_valid_x_unique: list[float] = field(default_factory=list)
    rdkit_valid_x_unique_x_novel: list[float] = field(default_factory=list)

    def record(self, run: SubdirRunResult) -> None:
        """Append non-None metric blocks from one subdirectory run."""
        if run.edm is not None:
            self.edm_atom_stability.append(run.edm.atom_stability)
            self.edm_molecule_stability.append(run.edm.molecule_stability)
            self.edm_valid.append(run.edm.valid)
            self.edm_valid_x_unique.append(run.edm.valid_x_unique)

        if run.posebusters is not None:
            self.posebusters.append(run.posebusters)

        if run.rdkit_x_posebusters is not None:
            self.rdkit_x_posebusters.append(run.rdkit_x_posebusters)

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
) -> tuple[RdkitMetrics, list[bool | None]]:
    """Wrap validity/uniqueness/novelty into an RdkitMetrics dataclass."""
    valid,valid_x_unique, valid_x_unique_x_novel, validity_flag_list = compute_validity_uniqueness_novelty(
        mols, reference_smiles
    )
    return RdkitMetrics(valid, valid_x_unique, valid_x_unique_x_novel), validity_flag_list


# ---------------------------------------------------------------------------
# Subdirectory evaluation
# ---------------------------------------------------------------------------


def compute_physchem_properties_from_mols(mols: list) -> dict[str, float] | None:
    """Mean MW, heavy-atom count, SA, QED, and Lipinski pass rate over valid mols.

    Returns None if no usable molecules are present.
    """
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


def run_posebusters(cond_file: Path, pred_file: Path) -> dict[str, float] | None:
    """Run PoseBusters on predicted mols; use dock mode if a pocket PDB exists.

    Writes posebusters_report.csv next to cond_file and returns per-check pass
    rates plus ``all`` (fraction of molecules that pass every check).
    """
    buster = PoseBusters(config="mol")

    if os.path.exists(cond_file):
        buster = PoseBusters(config="dock")
    df = buster.bust(
        mol_pred=[pred_file], mol_true=None, mol_cond=cond_file, full_report=False
    )
    subdir = cond_file.parent
    df.to_csv(subdir / "posebusters_report.csv", index=False)
    try:
        metrics = {column: float(df[column].mean()) for column in df.columns}
        metrics["all"] = float(df.all(axis=1).mean())
        return metrics
    except Exception:
        return None


def evaluate_subdirectory(
    subdir: Path,
    params: dict,
    *,
    compute_drugflow_clashes: bool,
    compute_edm: bool,
    compute_novelty: bool,
    compute_posebusters: bool,
    compute_posecheck: bool,
    compute_strain: bool,
    compute_physchem: bool,
    compute_vina: bool,
    reference_smiles: list[str] | None,
    add_hydrogens: bool,
) -> SubdirRunResult | None:
    """Evaluate one seed/prefix/pocket directory and write local artifacts.

    Loads generated_mols.pt (or .sdf), builds RDKit molecules, computes the
    enabled metric suites, and writes evaluation_results.txt plus 2D/3D plots.

    Args:
        subdir: Path to a result directory containing generated molecules.
        params: Evaluation config dict (dataset name, bond predictor path, …).
        compute_*: Flags selecting which metric families to run.
        reference_smiles: Training SMILES for novelty, or None.
        add_hydrogens: Whether to add hydrogens to the generated molecules.

    Returns:
        SubdirRunResult on success, or None if generated molecules are missing.
    """
    result = SubdirRunResult()

    # Pockets skipped during FBDD generation may leave an empty directory
    # (no generated_mols.*). Skip those rather than crashing.
    generated_file = subdir / "generated_mols.sdf" 
    
    if not generated_file.exists():
        print(f"Skipping {subdir.name}: missing {generated_file.name}")
        return None

    supplier = SDMolSupplier(str(generated_file), removeHs=False, sanitize=False)
    mols = []
    for mol in supplier:
        try:
            Chem.SanitizeMol(mol)
            if add_hydrogens:
                mol = Chem.AddHs(mol, addCoords=True)
            mols.append(mol)
        except Exception as e:
            print(f"Warning: Failed to sanitize a molecule: {e}")
            continue
    if len(mols) < 100:
        mols += [None] * (100 - len(mols))  # Pad with None if fewer than 100 molecules
    
    if add_hydrogens:
        save_molecules_to_sdf(mols, subdir / "generated_mols_with_hs.sdf")

    tensor_file_available = True
    try:
        builder = MoleculeBuilder(vocab=params["data_set"])
        x, pos, batch = builder.load_tensor_from_file(subdir)
    except FileNotFoundError as e:
        print(f"Skipping {subdir.name}: {e}")
        tensor_file_available = False
        
    if compute_edm and tensor_file_available:
        atom_stability, mol_stability, edm_valid, edm_unique, _ = (
            compute_edm_metrics_from_tensors(x, pos, batch, params["data_set"].upper())
        )
        result.edm = EdmMetrics(
            atom_stability=atom_stability,
            molecule_stability=mol_stability,
            valid=edm_valid,
            valid_x_unique=edm_valid * edm_unique,
        )
    
    result.rdkit, validity_flag_list = _compute_rdkit_metrics(
        mols, reference_smiles
    )
    
    mols = [mol for mol, is_valid in zip(mols, validity_flag_list) if is_valid]
    
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        mol_sdf_path = tmp.name
        
    writer = Chem.SDWriter(mol_sdf_path)
    for mol in mols:
        if mol is not None:
            writer.write(mol)
    writer.close()
    
    pocket_path = subdir / "pocket.pdb"
    
    if compute_posebusters:
        result.posebusters = run_posebusters(pocket_path, mol_sdf_path)

    if (
        result.rdkit is not None
        and result.posebusters is not None
        and "all" in result.posebusters
    ):
        result.rdkit_x_posebusters = result.rdkit.valid * result.posebusters["all"]

    if compute_posecheck:
        result.posecheck = compute_posecheck_metrics_from_mols(mols, str(pocket_path), compute_strain)

    if compute_drugflow_clashes:
        clash_evaluator = ClashEvaluator()
        pocket_path = subdir / "pocket.pdb"
        result.drugflow = clash_evaluator.evaluate_mols(mols, str(pocket_path))

    if compute_vina:
        gnina_evaluator = GninaEvaluator()
        pocket_path = subdir / "pocket.pdb"
        ligand_sdf_path = subdir / "ligand.sdf"
        vina_results = gnina_evaluator.evaluate_mols(
            mol_sdf_path, str(pocket_path), str(ligand_sdf_path), minimize=True
        )
        vin_min_results = {f"{key}_min": value for key, value in vina_results.items()}
        vina_results = gnina_evaluator.evaluate_mols(
            mol_sdf_path, str(pocket_path), str(ligand_sdf_path), minimize=False
        )
        result.vina = vina_results | vin_min_results

        mean_nha = float(
            np.mean([mol.GetNumHeavyAtoms() for mol in mols if mol is not None])
        )
        if mean_nha > 0:
            # Insert efficiency next to the corresponding vina scores for readability.
            ordered_vina: dict[str, float] = {}
            for key, value in result.vina.items():
                ordered_vina[key] = value
                if key == "vina_score":
                    ordered_vina["vina_efficiency"] = value / mean_nha
                elif key == "vina_score_min":
                    ordered_vina["vina_efficiency_min"] = value / mean_nha
            result.vina = ordered_vina

    if compute_physchem:
        result.physchem = compute_physchem_properties_from_mols(mols)

    write_subdir_results(
        subdir,
        params,
        result,
        compute_novelty,
    )
    save_2d_molecules_visualizations_to_png(subdir, mols)
    if tensor_file_available:
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
    compute_strain,
    compute_physchem,
    compute_vina,
    reference_smiles,
    add_hydrogens,
    log_filename="evaluation.log",
):
    """Process-pool worker: evaluate one subdirectory with stdout/stderr redirected.

    Logging noise from PoseCheck / docking tools is appended to log_filename
    so the parent process progress bar stays readable.
    """
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
                compute_strain=compute_strain,
                compute_physchem=compute_physchem,
                compute_vina=compute_vina,
                reference_smiles=reference_smiles,
                add_hydrogens=add_hydrogens
            )
        finally:
            # Always restore the low-level streams back to the terminal
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation over all result subdirectories and write the summary.

    Uses a process pool when num_workers > 1. Skipped or failing subdirectories
    are omitted from the aggregate rather than aborting the whole run.
    """
    params = load_evaluation_config(args.config_file)

    compute_drugflow_clashes = bool(params.get("compute_drugflow_clashes", False))
    compute_edm = bool(params.get("compute_edm", False))
    compute_novelty = bool(params.get("compute_novelty", False))
    compute_posebusters = bool(params.get("compute_posebusters", False))
    compute_posecheck = bool(params.get("compute_posecheck", False))
    compute_strain = bool(params.get("compute_strain", False))
    compute_physchem = bool(params.get("compute_physchem", False))
    compute_vina = bool(params.get("compute_vina", False))
    add_hydrogens = bool(params.get("add_hydrogens", False))

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
            compute_strain=compute_strain,
            compute_physchem=compute_physchem,
            compute_vina=compute_vina,
            reference_smiles=reference_smiles,
            add_hydrogens=add_hydrogens,
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
                try:
                    runs_dict[original_index] = future.result()
                except Exception as e:
                    subdir = subdirs[original_index]
                    print(f"Skipping {subdir.name}: {e}")
                    runs_dict[original_index] = None

        runs = [
            runs_dict[i]
            for i in sorted(runs_dict.keys())
            if runs_dict[i] is not None
        ]
    
    else:
        for subdir in iter_result_subdirs(data_path):
            print(f"Evaluating {subdir.name}...")
            try:
                run = evaluate_subdirectory(
                    subdir,
                    params,
                    compute_drugflow_clashes=compute_drugflow_clashes,
                    compute_edm=compute_edm,
                    compute_novelty=compute_novelty,
                    compute_posebusters=compute_posebusters,
                    compute_posecheck=compute_posecheck,
                    compute_strain=compute_strain,
                    compute_physchem=compute_physchem,
                    compute_vina=compute_vina,
                    reference_smiles=reference_smiles,
                    add_hydrogens=params.get("add_hydrogens", False)
                )
            except Exception as e:
                print(f"Skipping {subdir.name}: {e}")
                continue
            if run is not None:
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
    )


def main() -> None:
    """CLI entry point for scripts/evaluation.py."""
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
